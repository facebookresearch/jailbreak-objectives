# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import logging


@dataclass
class CECalculatorConfig:
    """Configuration for cross-entropy calculation"""
    model_id: str
    batch_size: int = 32
    padding_side: Optional[str] = None  # If None, will be determined based on model
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    surrogate_attack_prompt: str = ""

    def __post_init__(self):
        if self.padding_side is None:
            # Set default padding side based on model
            if self.model_id in ['google/gemma-2-9b-it', 'meta-llama/Llama-2-7b-chat-hf']:
                self.padding_side = 'right'
            else:
                self.padding_side = 'left'


class PrefixCECalculator:
    """
    Calculate cross-entropy loss for prefixes.

    This class handles the computation of negative log-likelihood (cross-entropy loss)
    for generated prefixes, using the victim language model.
    """

    def __init__(self, config: CECalculatorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_model()

    def _setup_model(self):
        """Initialize model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            padding_side=self.config.padding_side
        )
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map
        )

    def _clear_gpu_memory(self):
        """Clear GPU memory and ensure synchronization"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        del self.model
        del self.tokenizer

    def calculate_ce(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-entropy loss for all prefixes in the dataframe.

        Args:
            df: DataFrame with 'goal' and 'prefix' columns

        Returns:
            DataFrame with additional 'prefix_nll' column containing the loss values
        """
        self.logger.info(f"Calculating CE for {len(df)} prefixes")

        result_df = df.copy()
        result_df['prefix_nll'] = float('inf')
        sep = ' ' if self.config.padding_side == 'right' else ''

        for i in tqdm(range(0, len(df), self.config.batch_size)):
            batch = df.iloc[i:i + self.config.batch_size]
            goals = batch['goal'].tolist()
            prefixes = batch['prefix'].tolist()

            # Prepare inputs and get context lengths in one pass
            context_lens = []
            formatted_inputs = []
            for goal, prefix in zip(goals, prefixes):
                goal += self.config.surrogate_attack_prompt.format(prefix=prefix.lstrip())

                chat = [{"role": "user", "content": goal}]
                context = self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True
                )
                context_id = self.tokenizer.encode(context, add_special_tokens=False)
                context_lens.append(len(context_id))
                formatted_inputs.append(context + sep + prefix.lstrip())

            # Tokenize batch
            inputs = self.tokenizer(
                formatted_inputs,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
                add_special_tokens=False
            )

            # Calculate losses for the batch
            with torch.no_grad():
                # Do not use cache cause otherwise gemma-2 messes up the devices
                logits = self.model(**inputs, use_cache=False).logits

            # Calculate loss for each sequence in the batch
            losses = []
            for j in range(len(logits)):
                if self.config.padding_side == 'left':
                    pad_lens = (inputs['attention_mask'] == 0).sum(1)
                    target_logits = logits[j, pad_lens[j] + context_lens[j] - 1: -1, :]
                    target_ids = inputs.input_ids[j,
                                 pad_lens[j] + context_lens[j]:]  # Target IDs for the first i tokens
                else:
                    input_lens = (inputs['attention_mask'] == 1).sum(1)
                    target_logits = logits[j, context_lens[j] - 1: input_lens[j] - 1, :]
                    target_ids = inputs.input_ids[j, context_lens[j]: input_lens[j]]
                loss = F.cross_entropy(target_logits, target_ids, reduction='sum')
                losses.append(loss.item())

            # Update DataFrame with calculated losses
            result_df.iloc[i:i + len(batch), result_df.columns.get_loc('prefix_nll')] = losses

        return result_df

    def __del__(self):
        """Cleanup resources when the calculator is destroyed."""
        self._clear_gpu_memory()
