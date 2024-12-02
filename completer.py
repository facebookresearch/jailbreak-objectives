# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Module for getting complete responses from prefixes using target LLM.
"""

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class CompletionConfig:
    """Configuration for getting completions"""
    model_id: str
    batch_size: int = 32
    max_new_tokens: int = 256
    temperature: float = 1.0
    n_samples: int = 25
    surrogate_attack_prompt: str = ""

    def __post_init__(self):
        if self.model_id in ['google/gemma-2-9b-it', 'meta-llama/Llama-2-7b-chat-hf']:
            self.sep = ' ' # padding_side = 'right'
        else:
            self.sep = '' # padding_side = 'left'


class PrefixCompleter:
    """Class for getting completions from prefixes using target LLM"""

    def __init__(self, config: CompletionConfig):
        """Initialize the completer with config"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.llm = LLM(model=config.model_id)
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=1.,
            max_tokens=config.max_new_tokens,
        )

    def expand_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand dataframe to include multiple samples per prefix"""
        expanded_rows = []
        for _, row in df.iterrows():
            for sample_id in range(self.config.n_samples):
                expanded_row = row.to_dict()
                expanded_row['sample_id'] = sample_id
                expanded_row['completion'] = ""  # Placeholder
                expanded_rows.append(expanded_row)
        return pd.DataFrame(expanded_rows)

    def get_completions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get completions for all prefixes in dataframe"""
        # First expand the dataframe
        expanded_df = self.expand_dataframe(df)

        # Unify column names if needed
        expanded_df.rename(columns={'target': 'prefix'}, inplace=True)
        expanded_df.rename(columns={'target_ce_loss': 'prefix_nll'}, inplace=True)

        # Get completions in batches
        for i in tqdm(range(0, len(expanded_df), self.config.batch_size),
                     desc="Getting completion"):
            _end = min(i + self.config.batch_size, len(expanded_df))
            batch = expanded_df.iloc[i:_end]
            goals = batch['goal'].tolist()
            prefixes = batch['prefix'].tolist()

            # Prepare inputs
            formatted_inputs = []
            for goal, prefix in zip(goals, prefixes):
                # Add surrogate attack prompt if specified
                if self.config.surrogate_attack_prompt != "":
                    goal += self.config.surrogate_attack_prompt.format(
                        prefix=prefix.lstrip())

                chat = [{"role": "user", "content": goal}]
                context = self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True
                )
                context = context[len(self.tokenizer.bos_token):]

                # Format final input
                formatted_inputs.append(context + self.config.sep + prefix.lstrip())

            # Generate completions
            outputs = self.llm.generate(formatted_inputs, self.sampling_params)
            expanded_df.loc[i:_end - 1, 'completion'] = [o.outputs[0].text for o in outputs]

        # concatenate completion with prefix to get full responses
        expanded_df['completion'] = expanded_df.apply(lambda row: row['prefix'] + row['completion'],
                                                      axis=1)
        return expanded_df

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'llm'):
            del self.llm
        torch.cuda.empty_cache()
        torch.cuda.synchronize()