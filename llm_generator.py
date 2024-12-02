# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma2ForCausalLM
from tqdm import tqdm


def remove_padded_tokens(
    generated_ids: torch.Tensor, attention_mask: torch.Tensor, padding_side: str
) -> List[torch.Tensor]:
    """
    Remove padded tokens from generated_ids based on the attention_mask and padding_side.

    Args:
        generated_ids (torch.Tensor): Tensor of generated token IDs (batch_size, seq_length).
        attention_mask (torch.Tensor): Tensor indicating non-padded tokens (batch_size, seq_length).
        padding_side (str): 'left' or 'right' padding.

    Returns:
        List[torch.Tensor]: List of cleaned token ID tensors for each sequence.
    """
    cleaned_ids = []
    input_length = attention_mask.size(1)
    for gen_id, mask in zip(generated_ids, attention_mask):
        if padding_side == "right":
            non_padded_length = mask.sum().item()
            # Preserve generated tokens beyond the input mask
            cleaned_id = torch.cat([gen_id[:non_padded_length], gen_id[input_length:]])
        elif padding_side == "left":
            padded_length = (1 - mask).sum().item()
            cleaned_id = gen_id[padded_length:]
        else:
            raise ValueError(f"Unsupported padding side: {padding_side}")
        cleaned_ids.append(cleaned_id)
    return cleaned_ids


class DualLLMGenerator:
    def __init__(
        self,
        original_model_name: str,
        uncensored_model_name: str,
        device_ids: Optional[List[int]] = None
    ):
        """
        Initializes the DualLLMGenerator with two language models.

        Args:
            original_model_name (str): Hugging Face model name for the original LLM.
            uncensored_model_name (str): Hugging Face model name for the uncensored LLM.
            device_ids (List[int], optional): List of GPU device IDs to use. Defaults to all available GPUs.

        Raises:
            ValueError: If no GPUs are available.
            RuntimeError: If there's an error loading the models.
        """
        self.original_model_name = original_model_name
        self.uncensored_model_name = uncensored_model_name

        # Determine device IDs
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = device_ids

        if not self.device_ids:
            raise ValueError("No GPUs available. Please provide GPU device IDs or use CPU.")

        # Load tokenizer
        padding_side = self.get_padding_side(self.original_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.original_model_name,
            padding_side=padding_side
        )
        self.uncensored_tokenizer = AutoTokenizer.from_pretrained(
            self.uncensored_model_name,
            padding_side=padding_side
        )

        # Validate vocabularies match
        self._validate_vocabularies()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not have an EOS token defined.")

        self.eos_token_id = self.tokenizer.eos_token_id

        # Determine dtype based on device availability
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Load models with device_map='auto'
        try:
            self.original_model = AutoModelForCausalLM.from_pretrained(
                self.original_model_name,
                device_map='auto',
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_cache=True,
            )
        except Exception as e:
            raise RuntimeError(f"Error loading original model: {e}") from e

        try:
            self.uncensored_model = AutoModelForCausalLM.from_pretrained(
                self.uncensored_model_name,
                device_map='auto',
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_cache=True,
            )
        except Exception as e:
            raise RuntimeError(f"Error loading uncensored model: {e}") from e

        # Set models to evaluation mode and disable gradients
        self.original_model.eval()
        self.uncensored_model.eval()
        for param in self.original_model.parameters():
            param.requires_grad = False
        for param in self.uncensored_model.parameters():
            param.requires_grad = False

    def get_padding_side(self, model_name: str) -> str:
        """
        Determines the padding side for the tokenizer based on the model name.

        Args:
            model_name (str): The name of the model.

        Returns:
            str: 'left' or 'right' padding side.
        """
        # Currently hardcoded to 'left'; can be customized for specific models
        return 'left'

    def complete(
        self,
        texts: List[str],
        batch_size: int = 1,
        max_new_tokens: int = 100,
        guided_topk: int = 3,
        temperature: float = 1.0,
        **kwargs,
    ) -> List[str]:
        """
        Generates completions for a list of input texts.

        Args:
            texts (List[str]): List of input texts already formatted with the chat template, including BOS.
            batch_size (int): Number of texts to process in parallel.
            max_new_tokens (int): Maximum number of tokens to generate.
            guided_topk (int): Number of top tokens to consider from the uncensored model.
            temperature (float): Temperature parameter for sampling.

        Returns:
            List[str]: List of generated completions corresponding to each input text.
        """
        if isinstance(self.original_model, Gemma2ForCausalLM):
            return self.complete_no_cache(texts, batch_size, max_new_tokens, guided_topk, temperature)
        else:
            return self.complete_cached(texts, batch_size, max_new_tokens, guided_topk, temperature)

    def complete_cached(
        self,
        texts: List[str],
        batch_size: int = 1,
        max_new_tokens: int = 100,
        guided_topk: int = 3,
        temperature: float = 1.0
    ) -> List[str]:
        """
        Generates completions using cached past_key_values for efficiency.

        Args:
            texts (List[str]): List of input texts already formatted with the chat template, including BOS.
            batch_size (int): Number of texts to process in parallel.
            max_new_tokens (int): Maximum number of tokens to generate.
            guided_topk (int): Number of top tokens to consider from the uncensored model.
            temperature (float): Temperature parameter for sampling.

        Returns:
            List[str]: List of generated completions corresponding to each input text.
        """
        if not texts:
            return []

        completions = []
        total_batches = math.ceil(len(texts) / batch_size)

        with torch.no_grad():
            for batch_num in tqdm(range(total_batches), desc="Generating completions (cached)"):
                batch_texts = texts[batch_num * batch_size: (batch_num + 1) * batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                )
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                # Validate input
                self._validate_inputs(input_ids)

                # Separate context_ids and last_token_ids
                context_ids, last_token_ids, attention_mask = self._split_input(input_ids, attention_mask)

                # Initialize generated_ids with context_ids and last_token_ids
                generated_ids = torch.cat((context_ids, last_token_ids), dim=1).clone()

                # Initialize past_key_values for both models using context_ids
                past_original = self.original_model(
                    context_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True
                ).past_key_values

                past_uncensored = self.uncensored_model(
                    context_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True
                ).past_key_values

                # Track which sequences have finished generation
                finished = [False] * len(batch_texts)

                extended_attention_mask = attention_mask.clone()

                for _ in range(max_new_tokens):
                    # Update extended_attention_mask to include new tokens
                    extended_attention_mask = torch.cat(
                        (
                            extended_attention_mask,
                            torch.ones(
                                (attention_mask.size(0), 1),
                                device=attention_mask.device,
                                dtype=attention_mask.dtype
                            )
                        ),
                        dim=1
                    )

                    # Uncensored model forward pass
                    outputs_uncensored = self.uncensored_model(
                        last_token_ids,
                        past_key_values=past_uncensored,
                        attention_mask=extended_attention_mask,
                        use_cache=True,
                        return_dict=True
                    )
                    logits_uncensored = outputs_uncensored.logits  # (batch_size, 1, vocab_size)
                    past_uncensored = outputs_uncensored.past_key_values

                    next_token_logits_uncensored = logits_uncensored[:, -1, :]  # (batch_size, vocab_size)

                    # Exclude EOS token
                    next_token_logits_uncensored[:, self.eos_token_id] = -float('inf')

                    # Get top-k tokens
                    topk_logits, topk_indices = torch.topk(
                        next_token_logits_uncensored, guided_topk, dim=-1
                    )  # (batch_size, k)

                    # Original model forward pass
                    outputs_original = self.original_model(
                        last_token_ids,
                        past_key_values=past_original,
                        attention_mask=extended_attention_mask,
                        use_cache=True,
                        return_dict=True
                    )
                    logits_original = outputs_original.logits  # (batch_size, 1, vocab_size)
                    past_original = outputs_original.past_key_values

                    next_token_logits_original = logits_original[:, -1, :]  # (batch_size, vocab_size)

                    # Gather logits for top-k tokens
                    topk_logits_original = torch.gather(
                        next_token_logits_original, dim=-1, index=topk_indices
                    )  # (batch_size, k)

                    # Compute probabilities
                    topk_probs_original = F.log_softmax(
                        topk_logits_original / temperature, dim=-1
                    ).exp()  # (batch_size, k)

                    # Sample tokens from top-k
                    selected_token = torch.multinomial(
                        topk_probs_original, num_samples=1
                    )  # (batch_size, 1)

                    # Retrieve token IDs
                    selected_token_id = torch.gather(
                        topk_indices, dim=-1, index=selected_token
                    )  # (batch_size, 1)

                    # Append to generated_ids
                    generated_ids = torch.cat([generated_ids, selected_token_id], dim=-1)

                    # Update last_token_ids
                    last_token_ids = selected_token_id

                    # Check for EOS tokens
                    for i, token_id in enumerate(selected_token_id.squeeze(-1).tolist()):
                        if not finished[i] and token_id == self.eos_token_id:
                            finished[i] = True

                    # If all sequences have finished, break
                    if all(finished):
                        break

                # Decode the generated tokens
                cleaned_generated_ids = remove_padded_tokens(
                    generated_ids, inputs['attention_mask'], self.tokenizer.padding_side
                )
                generated_texts = self.tokenizer.batch_decode(
                    cleaned_generated_ids, skip_special_tokens=False
                )
                completions.extend(generated_texts)

        return completions

    def complete_no_cache(
        self,
        texts: List[str],
        batch_size: int = 1,
        max_new_tokens: int = 100,
        guided_topk: int = 3,
        temperature: float = 1.0
    ) -> List[str]:
        """
        Generates completions without using cached past_key_values.

        Args:
            texts (List[str]): List of input texts already formatted with the chat template, including BOS.
            batch_size (int): Number of texts to process in parallel.
            max_new_tokens (int): Maximum number of tokens to generate.
            guided_topk (int): Number of top tokens to consider from the uncensored model.
            temperature (float): Temperature parameter for sampling.

        Returns:
            List[str]: List of generated completions corresponding to each input text.
        """
        if not texts:
            return []

        completions = []
        total_batches = math.ceil(len(texts) / batch_size)

        with torch.no_grad():
            for batch_num in tqdm(range(total_batches), desc="Generating completions (no cache)"):
                batch_texts = texts[batch_num * batch_size: (batch_num + 1) * batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                )
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                # Validate input
                self._validate_inputs(input_ids)

                # Separate context_ids and last_token_ids
                context_ids, last_token_ids, attention_mask = self._split_input(input_ids, attention_mask)

                # Initialize generated_ids with context_ids and last_token_ids
                generated_ids = torch.cat((context_ids, last_token_ids), dim=1).clone()

                # Track which sequences have finished generation
                finished = [False] * len(batch_texts)

                extended_attention_mask = attention_mask.clone()

                for _ in range(max_new_tokens):
                    # Update extended_attention_mask to include new tokens
                    extended_attention_mask = torch.cat(
                        (
                            extended_attention_mask,
                            torch.ones(
                                (attention_mask.size(0), 1),
                                device=attention_mask.device,
                                dtype=attention_mask.dtype
                            )
                        ),
                        dim=1
                    )

                    # Uncensored model forward pass
                    outputs_uncensored = self.uncensored_model(
                        generated_ids,
                        attention_mask=extended_attention_mask,
                        return_dict=True,
                        use_cache=False,
                    )
                    logits_uncensored = outputs_uncensored.logits  # (batch_size, seq_length, vocab_size)

                    next_token_logits_uncensored = logits_uncensored[:, -1, :]  # (batch_size, vocab_size)

                    # Exclude EOS token
                    next_token_logits_uncensored[:, self.eos_token_id] = -float('inf')

                    # Get top-k tokens
                    topk_logits, topk_indices = torch.topk(
                        next_token_logits_uncensored, guided_topk, dim=-1
                    )  # (batch_size, k)

                    # Original model forward pass
                    outputs_original = self.original_model(
                        generated_ids,
                        attention_mask=extended_attention_mask,
                        return_dict=True,
                        use_cache=False,
                    )
                    logits_original = outputs_original.logits  # (batch_size, seq_length, vocab_size)

                    next_token_logits_original = logits_original[:, -1, :]  # (batch_size, vocab_size)

                    # Gather logits for top-k tokens
                    topk_logits_original = torch.gather(
                        next_token_logits_original, dim=-1, index=topk_indices
                    )  # (batch_size, k)

                    # Compute probabilities
                    topk_probs_original = F.log_softmax(
                        topk_logits_original / temperature, dim=-1
                    ).exp()  # (batch_size, k)

                    # Sample tokens from top-k
                    selected_token = torch.multinomial(
                        topk_probs_original, num_samples=1
                    )  # (batch_size, 1)

                    # Retrieve token IDs
                    selected_token_id = torch.gather(
                        topk_indices, dim=-1, index=selected_token
                    )  # (batch_size, 1)

                    # Append to generated_ids
                    generated_ids = torch.cat([generated_ids, selected_token_id], dim=-1)

                    # Update last_token_ids
                    last_token_ids = selected_token_id

                    # Check for EOS tokens
                    for i, token_id in enumerate(selected_token_id.squeeze(-1).tolist()):
                        if not finished[i] and token_id == self.eos_token_id:
                            finished[i] = True

                    # If all sequences have finished, break
                    if all(finished):
                        break

                # Decode the generated tokens
                cleaned_generated_ids = remove_padded_tokens(
                    generated_ids, inputs['attention_mask'], self.tokenizer.padding_side
                )
                generated_texts = self.tokenizer.batch_decode(cleaned_generated_ids, skip_special_tokens=False)
                completions.extend(generated_texts)

        return completions

    def _validate_vocabularies(self):
        """
        Validates that both models use the same vocabulary.
        Raises ValueError if vocabularies don't match.
        """
        # Compare vocabulary sizes
        if len(self.tokenizer.get_vocab()) != len(self.uncensored_tokenizer.get_vocab()):
            raise ValueError(
                f"Vocabulary sizes don't match! Original model: {len(self.tokenizer.get_vocab())}, "
                f"Uncensored model: {len(self.uncensored_tokenizer.get_vocab())}"
            )

        # Compare special tokens
        orig_special = {
            'eos_token': self.tokenizer.eos_token,
            'bos_token': self.tokenizer.bos_token,
        }
        uncens_special = {
            'eos_token': self.uncensored_tokenizer.eos_token,
            'bos_token': self.uncensored_tokenizer.bos_token,
        }
        if orig_special != uncens_special:
            raise ValueError(
                f"Special tokens don't match!\nOriginal model: {orig_special}\n"
                f"Uncensored model: {uncens_special}"
            )

        # Compare entire vocabulary dictionaries at once
        orig_vocab = self.tokenizer.get_vocab()
        uncens_vocab = self.uncensored_tokenizer.get_vocab()

        if orig_vocab != uncens_vocab:
            # Find mismatched tokens for error message
            mismatched = []
            for token in orig_vocab:
                if token not in uncens_vocab:
                    mismatched.append(f"Missing in uncensored: {token}")
                elif orig_vocab[token] != uncens_vocab[token]:
                    mismatched.append(f"ID mismatch for '{token}': {orig_vocab[token]} vs {uncens_vocab[token]}")
                if len(mismatched) >= 5:  # Limit to first 5 mismatches
                    break
            raise ValueError(f"Vocabulary mismatch! First few differences:\n" + "\n".join(mismatched))

    def _validate_inputs(self, input_ids: torch.Tensor) -> None:
        """
        Validates the input_ids for correct BOS token placement.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs (batch_size, seq_length).

        Raises:
            AssertionError: If the input does not start with a single BOS token.
        """
        assert (input_ids[0] == self.tokenizer.bos_token_id).sum() == 1, \
            f"BOS error: {input_ids[0]}!"
        assert input_ids[0, 0] in [
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id
        ], f"Input not starting with BOS! Found {self.tokenizer.decode(input_ids[0, 0])}"

    def _split_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits the input_ids and attention_mask into context and last tokens.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs (batch_size, seq_length).
            attention_mask (torch.Tensor): Tensor indicating non-padded tokens (batch_size, seq_length).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: context_ids, last_token_ids, updated attention_mask
        """
        if self.tokenizer.padding_side == 'left':
            context_ids = input_ids[:, :-1]  # (batch_size, seq_length - 1)
            last_token_ids = input_ids[:, -1].unsqueeze(-1)  # (batch_size, 1)
            attention_mask = attention_mask[:, :-1]
        else:
            indices = -1 - (attention_mask == 0).sum(1)
            mask = torch.ones_like(input_ids, dtype=torch.bool)
            mask[torch.arange(input_ids.size(0)), indices] = False
            context_ids = input_ids[mask].view(input_ids.size(0), -1)
            attention_mask = attention_mask[mask].view(attention_mask.size(0), -1)
            last_token_ids = input_ids[torch.arange(input_ids.size(0)), indices].unsqueeze(-1)
        return context_ids, last_token_ids, attention_mask


class SingleLLMGenerator:
    def __init__(
        self,
        model_name: str,
        device_ids: Optional[List[int]] = None
    ):
        """
        Initializes the SingleLLMGenerator with a single language model.

        Args:
            model_name (str): Hugging Face model name for the LLM.
            device_ids (List[int], optional): List of GPU device IDs to use. Defaults to all available GPUs.

        Raises:
            ValueError: If no GPUs are available.
            RuntimeError: If there's an error loading the model.
        """
        self.model_name = model_name

        # Determine device IDs
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = device_ids

        if not self.device_ids:
            raise ValueError("No GPUs available. Please provide GPU device IDs or use CPU.")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left'  # Assuming 'left' padding; adjust as needed
        )

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # if self.tokenizer.pad_token_id is None:
        #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not have an EOS token defined.")

        self.eos_token_id = self.tokenizer.eos_token_id

        # Determine dtype based on device availability
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Load model with device_map='auto'
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map='auto',
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_cache=True,
            )
        except Exception as e:
            raise RuntimeError(f"Error loading model '{self.model_name}': {e}") from e

        # Set model to evaluation mode and disable gradients
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def complete(
        self,
        texts: List[str],
        batch_size: int = 1,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
        """
        Generates completions for a list of input texts using greedy or sampling decoding.

        Args:
            texts (List[str]): List of input texts.
            batch_size (int): Number of texts to process in parallel.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Temperature parameter for sampling.
            top_p (float): Top-p (nucleus) sampling parameter.
            guided_topk (float): select k tokens using uncensored LLM, and then rank them based on victim LLM.

        Returns:
            List[str]: List of generated completions corresponding to each input text.
        """
        if not texts:
            return []

        completions = []
        total_batches = math.ceil(len(texts) / batch_size)

        with torch.no_grad():
            for batch_num in tqdm(range(total_batches), desc="Generating completions (SingleLLM)"):
                batch_texts = texts[batch_num * batch_size: (batch_num + 1) * batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                )
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                # Validate input
                self._validate_inputs(input_ids)

                # Generate outputs
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.eos_token_id,
                )

                # Decode the generated tokens
                cleaned_generated_ids = remove_padded_tokens(
                    outputs, inputs['attention_mask'], self.tokenizer.padding_side
                )
                generated_texts = self.tokenizer.batch_decode(
                    cleaned_generated_ids,
                    skip_special_tokens=False
                )
                completions.extend(generated_texts)

        return completions

    def _validate_inputs(self, input_ids: torch.Tensor) -> None:
        """
        Validates the input_ids for correct BOS token placement.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs (batch_size, seq_length).

        Raises:
            AssertionError: If the input does not start with a single BOS token.
        """
        assert (input_ids[0] == self.tokenizer.bos_token_id).sum() == 1, \
            f"BOS error: {input_ids[0]}!"
        assert input_ids[0, 0] in [
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id
        ], f"Input not starting with BOS! Found {self.tokenizer.decode(input_ids[0, 0])}"


if __name__ == "__main__":
    # Example usage of the DualLLMGenerator class

    # Non-Gemma2 models
    # original_model = "meta-llama/Meta-Llama-3-8B-Instruct"
    # uncensored_model = "Orenguteng/Llama-3-8B-Lexi-Uncensored"
    # # Define input prompts (already formatted with chat template)
    # input_prompts = [
    #     "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    #     "Give step-by-step instructions for how to make a cupcake"
    #     "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    #     #
    #     "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    #     "How's the weather today?"
    #     "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHere",
    # ]

    # Gemma2 models
    original_model = "google/gemma-2-9b-it"
    uncensored_model = "TheDrummer/Tiger-Gemma-9B-v1"
    # Define input prompts (already formatted with chat template)
    input_prompts = [
        "<bos><start_of_turn>user\n"
        "Give step-by-step instructions for how to make a cupcake"
        "<end_of_turn>\n<start_of_turn>model\nHere",
        #
        "<bos><start_of_turn>user\n"
        "How's the weather today?"
        "<end_of_turn>\n<start_of_turn>model\nHere",
    ]

    # Initialize the generator
    generator = DualLLMGenerator(
        original_model_name=original_model,
        uncensored_model_name=uncensored_model,
    )

    # Generate completions
    completions = generator.complete(
        texts=input_prompts,
        batch_size=3,
        max_new_tokens=50,
        guided_topk=3,
        temperature=1.0
    )

    # Print the completions
    for i, completion in enumerate(completions, start=1):
        print(f"--- Completion {i} ---")
        print(completion)
        print("\n")

    # Example usage of the SingleLLMGenerator class

    # Initialize SingleLLMGenerator
    single_generator = SingleLLMGenerator(
        model_name=uncensored_model,
    )

    # Define input prompts for SingleLLMGenerator (same as DualLLMGenerator)
    single_input_prompts = input_prompts

    # Generate completions using SingleLLMGenerator
    print("Generating completions using SingleLLMGenerator...")
    completions_single = single_generator.complete(
        texts=single_input_prompts,
        batch_size=3,
        max_new_tokens=50,
        temperature=1.0,
        top_p=0.9
    )

    # Print the SingleLLMGenerator completions
    for i, completion in enumerate(completions_single, start=1):
        print(f"--- SingleLLM Completion {i} ---")
        print(completion)
        print("\n")
