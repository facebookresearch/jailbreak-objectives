# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class PrefixSelectorConfig:
    """Configuration for prefix selection"""
    pasr_weight: float  # Weight for log-PASR in selection
    n_prefixes_per_goal: int = 1  # Number of prefixes to select per goal
    nll_tol: float = 999  # Tolerance for NLL relative to best prefix
    pasr_tol: float = 0  # Tolerance for PASR relative to best prefix
    judges: Optional[List[str]] = None  # List of judges to use for PASR calculation


class PrefixSelector:
    """
    Selects prefixes based on a combination of judge scores (PASR) and NLL.
    Supports multiple judges and custom weighting for selection criteria.
    """

    def __init__(self, config: PrefixSelectorConfig):
        """
        Initialize the prefix selector.

        Args:
            config: Configuration for prefix selection
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Mapping of judge types to their column names in the DataFrame
        self.judge_column_map = {
            'nuanced': 'eval_nj_mean',
            'jailbreakbench': 'eval_jb_mean',
            'harmbench': 'eval_hb_mean',
            'strongreject': 'eval_sj_binary_mean'
        }

    def select_prefixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select prefixes based on combined judge scores and NLL.

        Args:
            df: DataFrame containing prefixes with evaluation results

        Returns:
            DataFrame containing selected prefixes
        """
        # Validate judge columns exist
        if not self.config.judges:
            raise ValueError("No judges specified for selection")

        missing_judges = []
        for judge in self.config.judges:
            if judge not in self.judge_column_map:
                raise ValueError(f"Unknown judge type: {judge}")
            if self.judge_column_map[judge] not in df.columns:
                missing_judges.append(judge)

        if missing_judges:
            raise ValueError(f"Missing evaluation results for judges: {missing_judges}")

        # Create a working copy of the DataFrame
        work_df = df.copy()

        # Calculate combined PASR score from specified judges
        work_df['pasr'] = self._calculate_combined_pasr(work_df)

        # Calculate log PASR for scoring
        work_df['log_pasr'] = np.log(work_df['pasr'] + 1e-6)

        # Calculate combined score (minimize both 1 - PASR and prefix_nll)
        work_df['combined_score'] = -self.config.pasr_weight * work_df['log_pasr'] + work_df['prefix_nll']

        # Create DataFrame for selected prefixes
        selected_prefixes = pd.DataFrame()

        # Group by goal and apply selection process
        for goal, group in work_df.groupby('goal'):
            # Step 1: Select first prefix based on combined score
            first_selection = group.loc[group['combined_score'].idxmin()]

            # Step 2: Filter prefixes within PASR tolerance
            remaining_candidates = group[
                (group['pasr'] >= first_selection['pasr'] - self.config.pasr_tol) &
                (group.index != first_selection.name)
                ]

            # Step 3: Filter candidates within NLL tolerance
            valid_candidates = remaining_candidates[
                remaining_candidates['prefix_nll'] <= first_selection['prefix_nll'] + self.config.nll_tol
                ]

            # Initialize selections list with first selection
            selections = [first_selection]

            # Step 4: Iteratively select additional prefixes
            for _ in range(self.config.n_prefixes_per_goal - 1):
                # Remove candidates that are sub-prefixes of selected ones
                valid_candidates = valid_candidates[
                    ~valid_candidates['prefix'].apply(
                        lambda x: any(x.startswith(sel['prefix']) for sel in selections)
                    )
                ]

                if valid_candidates.empty:
                    break

                # Select next prefix with lowest NLL
                next_selection = valid_candidates.nsmallest(1, 'prefix_nll').iloc[0]
                selections.append(next_selection)
                valid_candidates = valid_candidates[valid_candidates.index != next_selection.name]

            # Combine selections for this goal
            combined_selection = pd.DataFrame(selections)
            selected_prefixes = pd.concat([selected_prefixes, combined_selection])

        # Reset index
        selected_prefixes.reset_index(drop=True, inplace=True)

        # Add the new columns (pasr, log_pasr, combined_score) to the output
        output_columns = list(df.columns) + ['pasr', 'log_pasr', 'combined_score']
        selected_prefixes = selected_prefixes[output_columns]

        self.logger.info(f"Selected {len(selected_prefixes)} prefixes across {len(df['goal'].unique())} goals")
        return selected_prefixes

    def _calculate_combined_pasr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate combined PASR score from specified judges.

        Args:
            df: DataFrame containing judge scores

        Returns:
            Series containing combined PASR scores
        """
        judge_scores = []
        for judge in self.config.judges:
            column = self.judge_column_map[judge]
            judge_scores.append(df[column])

        # Calculate mean of judge scores
        return pd.concat(judge_scores, axis=1).mean(axis=1)
