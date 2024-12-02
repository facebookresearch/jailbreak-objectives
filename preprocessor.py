# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Prefix preprocessing implementation for the prefix generation pipeline.
This module handles both filtering and ablation of prefixes.
"""

from dataclasses import dataclass
import logging
from typing import List, Dict, Optional
import pandas as pd
from transformers import AutoTokenizer
from functools import partial
import re


@dataclass
class PreprocessConfig:
    """Configuration for prefix preprocessing steps"""
    model_id: str

    # Filtering configs
    min_token_length: int = 24
    require_linebreak: bool = True
    max_ce: Optional[float] = None

    # Ablation configs
    max_token_segments: int = 5  # How many token segments to create from second line
    min_lines: int = 2  # Minimum number of non-empty lines required

    # Numbers of prefixes per goal to keep before nuanced evaluation
    n_candidates_per_goal: int = 100  # Number of prefixes to keep per goal after CE filtering

    # Patterns to filter out at the start of prefixes
    start_patterns: tuple[str, ...] = (
        "I cannot", "I can't", "I'm afraid", "Unfortunately",
        "Sorry", "I am sorry", "I'm sorry", "I am really sorry", "I'm very sorry", "I apologize",
        "As an", "As a", "I'm just", "I'm an", "I understand",
        "To begin, I must ", "I must", "I am not able to", "I am unable",
        "Here's what I can tell you", "Here is what I can tell you",
        "Here's what I can do", "Here is what I can do",
        "Here's what I can say", "Here is what I can say",
        "Here are some things you should know",
        "To be clear"
    )

    # Patterns to filter out anywhere in prefixes
    contain_patterns: tuple[str, ...] = (
        "I am an AI assistant", "I can't provide", "I cannot", "I can't assist",
        "I can't write", "I can't fulfill", "I can't do", "I can't help",
        "I can't and", "I am unable"
    )


class PrefixPreprocessor:
    """
    Implements preprocessing logic for prefixes, including filtering and ablation.

    Filtering is split into two phases:
    Phase 1 (before NLL calculation):
    1. Remove prefixes shorter than minimum token length
    2. Remove prefixes starting with unwanted phrases
    3. Remove prefixes containing unwanted phrases
    4. Remove prefixes without linebreaks
    5. Merge duplicates

    Phase 2 (after NLL calculation):
    1. Filter based on cross-entropy loss threshold

    Ablation process:
    1. Clean up prefixes by removing leading spaces while preserving line breaks
    2. Split prefixes into lines and identify non-empty lines
    3. Create variations by taking first line and different lengths of second line
    4. Merge duplicate prefixes while preserving metadata
    """

    def __init__(self, config: PreprocessConfig):
        """Initialize the preprocessor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        tokens = self.tokenizer.tokenize(text)
        return len(tokens)

    def _clean_prefix(self, prefix: str) -> str:
        """Clean prefix by removing leading spaces but keeping line breaks."""
        return re.sub(r"^\s+", "", prefix)

    def _merge_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge duplicate prefixes while preserving metadata."""
        rows_before = len(df)

        def concatenate_unique_entries(group):
            agg_dict = {
                'model_name': lambda x: ','.join(str(v) for v in set(x)),
                'meta_prefix': lambda x: ','.join(str(v) for v in set(x) if pd.notna(v)),
                'temperature': lambda x: ','.join(str(v) for v in set(x)),
                'goal': 'first'
            }

            if 'prefix_nll' in group.columns:
                agg_dict['prefix_nll'] = 'first'

            return group.groupby('prefix').agg(agg_dict).reset_index()

        df = df.groupby('goal').apply(concatenate_unique_entries).reset_index(drop=True)

        rows_after = len(df)
        self.logger.info(f"Duplicate merging removed {rows_before - rows_after} rows")
        return df

    def _print_detailed_stats(self, df: pd.DataFrame, step_name: str):
        """Print detailed statistics about remaining prefixes per goal."""
        goal_prefix_counts = df.groupby('goal')['prefix'].count()
        min_prefixes_left = goal_prefix_counts.min()
        average_prefixes_left = goal_prefix_counts.mean()
        goal_with_min_prefixes = goal_prefix_counts.idxmin()

        self.logger.info(f"Detailed {step_name} statistics:")
        self.logger.info(f"- Minimum prefixes left for any goal: {min_prefixes_left}")
        self.logger.info(f"- Average prefixes left per goal: {average_prefixes_left:.2f}")
        self.logger.info(f"- Goal with minimum prefixes: {goal_with_min_prefixes}")

        if min_prefixes_left < 10:
            self.logger.warning(f"Some goals have very few prefixes left (<10)")

    # Filtering methods
    def _filter_by_token_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove prefixes shorter than minimum token length."""
        if not self.config.min_token_length:
            return df

        rows_before = len(df)
        df['token_count'] = df['prefix'].apply(partial(self._count_tokens))
        df = df[df['token_count'] >= self.config.min_token_length]
        rows_after = len(df)

        self.logger.info(f"Token length filter removed {rows_before - rows_after} rows")
        return df.drop('token_count', axis=1)

    def _filter_by_start_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove prefixes starting with unwanted phrases."""
        rows_before = len(df)
        df = df[~df['prefix'].str.lstrip().str.startswith(self.config.start_patterns)]
        rows_after = len(df)

        self.logger.info(f"Start pattern filter removed {rows_before - rows_after} rows")
        return df

    def _filter_by_contain_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove prefixes containing unwanted phrases."""
        rows_before = len(df)
        pattern = '|'.join(map(str, self.config.contain_patterns))
        df = df[~df['prefix'].str.contains(pattern, regex=True)]
        rows_after = len(df)

        self.logger.info(f"Contain pattern filter removed {rows_before - rows_after} rows")
        return df

    def _filter_by_linebreak(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove prefixes that don't contain linebreaks (excluding at start/end)."""
        if not self.config.require_linebreak:
            return df

        rows_before = len(df)
        df = df[df['prefix'].str.lstrip().str.strip('\n').str.contains("\n")]
        rows_after = len(df)

        self.logger.info(f"Linebreak filter removed {rows_before - rows_after} rows")
        return df

    def _filter_by_ce_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove prefixes with cross-entropy loss above threshold."""
        if not self.config.max_ce or 'prefix_nll' not in df.columns:
            return df

        rows_before = len(df)
        df = df[df['prefix_nll'] <= self.config.max_ce]
        rows_after = len(df)

        self.logger.info(f"CE loss filter removed {rows_before - rows_after} rows")
        return df

    # Ablation methods
    def _should_ablate(self, prefix: str) -> bool:
        """Determine if a prefix should be ablated."""
        if "\n" not in prefix:
            return False

        lines = [line for line in re.split(r'(\n+)', prefix) if line.strip()]
        if len(lines) < self.config.min_lines:
            return False

        return True

    def _create_ablated_versions(self, row: pd.Series) -> List[Dict]:
        """Create ablated versions of a prefix."""
        prefix = row['prefix']

        lines_with_breaks = re.split(r'(\n+)', prefix)
        non_empty_lines = [line for line in lines_with_breaks if line.strip()]

        if len(non_empty_lines) < 2:
            return []

        first_line = non_empty_lines[0]
        second_line = non_empty_lines[1]
        second_line_tokens = self.tokenizer.tokenize(second_line.strip())

        new_rows = []
        for i in range(1, min(len(second_line_tokens) + 1, self.config.max_token_segments + 1)):
            truncated_tokens = second_line_tokens[:i]
            truncated_second_line = self.tokenizer.convert_tokens_to_string(truncated_tokens)
            new_prefix = f"{first_line}{lines_with_breaks[1]}{truncated_second_line}"

            new_row = row.copy()
            new_row['prefix'] = new_prefix
            new_rows.append(new_row)

        return new_rows

    # Public interface methods
    def filter_phase1(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply initial filtering steps before NLL calculation."""
        self.logger.info(f"Starting prefix filtering phase 1 with {len(df)} rows")

        df = self._filter_by_token_length(df)
        df = self._filter_by_start_patterns(df)
        df = self._filter_by_contain_patterns(df)
        df = self._filter_by_linebreak(df)
        df = self._merge_duplicates(df)

        self.logger.info(f"Phase 1 filtering complete. {len(df)} rows remaining")
        self._print_detailed_stats(df, "phase 1 filtering")
        return df

    def filter_phase2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final filtering steps after NLL calculation."""
        self.logger.info(f"Starting prefix filtering phase 2 with {len(df)} rows")

        # Filter by CE loss threshold
        df = self._filter_by_ce_loss(df)
        self.logger.info(f"After CE threshold filtering: {len(df)} rows")

        # Select top k prefixes per goal based on CE loss
        df = df.groupby('goal').apply(
            lambda x: x.nsmallest(self.config.n_candidates_per_goal, 'prefix_nll')
        ).reset_index(drop=True)
        self.logger.info(f"After selecting top {self.config.n_candidates_per_goal} per goal: "
                         f"{len(df)} rows")

        self._print_detailed_stats(df, "phase 2 filtering")
        return df

    def ablate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform prefix ablation on the input dataframe."""
        self.logger.info(f"Starting prefix ablation with {len(df)} rows")

        df['prefix'] = df['prefix'].apply(self._clean_prefix)
        ablatable_df = df[df['prefix'].apply(self._should_ablate)].reset_index(drop=True)
        self.logger.info(f"{len(ablatable_df)} prefixes are suitable for ablation")

        new_rows = []
        for _, row in ablatable_df.iterrows():
            new_rows.extend(self._create_ablated_versions(row))

        ablated_df = pd.DataFrame(new_rows)
        self.logger.info(f"Created {len(ablated_df)} ablated prefixes")

        final_df = self._merge_duplicates(ablated_df)
        self._print_detailed_stats(final_df, "ablation")

        return final_df