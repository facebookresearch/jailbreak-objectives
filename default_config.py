# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple
from ml_collections import ConfigDict


@dataclass
class PipelineConfig:
    # ===== Task config =====
    victim_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    input_csv: str = "./input/demo_requests.csv"  # malicious requests
    output_dir: str = "./output"
    # Step to resume from. start_step = 1 runs from scratch.
    start_step: int = 1
    # ID of the run to resume from. Leave blank if running from scratch and a random new run_id will be created.
    run_id: str = ""
    # Maximum number of prefixes to keep for each request
    n_prefixes_per_goal: int = 1

    # ===== Generating candidate prefixes and others =====
    # Batch size used in generating candidate prefixes, evaluating nll, and completing full responses
    batch_size: int = 256
    # Maximum number of tokens of generated candidate prefixes
    max_new_tokens: int = 32
    # Temperature used in generating candidate prefixes and completing full responses
    temperature: float = 1.0
    # Uncensored models for generating candidate prefixes.
    # For the given victim LLM, we iterate through all uncensored LLMs listed here
    uncensored_models: Tuple[str, ...] = (
        "TheDrummer/Tiger-Gemma-9B-v1",
        "Orenguteng/Llama-3-8B-Lexi-Uncensored",
        "Orenguteng/Llama-3.1-8B-Lexi-Uncensored",
        "georgesung/llama2_7b_chat_uncensored",
        "Tap-M/Luna-AI-Llama2-Uncensored"
    )
    # In generating prefixes candidates using uncensored models, prefill the model's response with meta prefixes
    meta_prefixes: Tuple[str, ...] = ("Here", "To", "Sure", "")
    # Number of randomly generated prefixes for each meta prefix above
    meta_prefix_samples: Tuple[int, ...] = (50, 50, 50, 150)
    # Only used for guided decoding. Select k tokens using uncensored LLM, and then rank them based on victim LLM
    guided_topk: int = 3

    # ===== Preprocessing=====
    # Filter out candidate prefixes shorter than this token length, as they are mostly refusals
    min_token_length: int = 16
    # Filter out candidate prefixes with negative log likelihood (cross-entropy loss) higher than this value.
    max_ce: float = 1e6  # no limit
    # Configs for heuristic truncation of substrings to save eval computes.
    # Filter out prefixes with number of non-empty lines fewer than min_lines
    min_lines: int = 2
    # Progressively truncate each candidate prefixes in the second line to get multiple shorter prefixes
    # max_token_segments specifies the maximum number of tokens to keep in the second line
    # E.g., first line \n abcdefg -> [first line \n a, first line \n ab, ..., first line \n abcde]
    max_token_segments: int = 5
    # Maximum number of prompts per request to keep before evaluating PASR to save compute
    n_candidates_per_goal: int = 100

    # ===== Evaluating NLL =====
    # the surrogate attack prompt used in estimating NLL and PASR
    surrogate_attack_prompt: int = 0

    # ===== Evaluating PASR =====
    # Number of randomly completed responses for each prefix to evaluate PASR
    n_samples: int = 25
    # Maximum number of tokens for each completed response. Allow long response to enable nuanced evaluation
    max_new_tokens_completion: int = 512
    # Evaluation judges to use in determining jailbreak success.
    # Choose one or more from ["nuanced", "jailbreakbench", "harmbench", "strongreject"]
    judges: Tuple[str] = ("nuanced",)
    # Consider responses shorted than this as incomplete responses
    filter_len: int = 300
    # Maximum number of generated tokens for nuanced judge and jailbreakbench.
    # Nuanced judge is required to generate reasoning steps before making decisions.
    max_new_tokens_eval: int = 512
    # Batch size used in judge's evaluation (in vLLM)
    batch_size_judge: int = 32

    # ===== Prefix selection =====
    # Which judge's score to consider as PASR in selection
    selection_judges: Tuple[str] = ("nuanced",)
    # Weight for PASR when considering both two criteria: combined score = pasr_weight * log(PASR) + NLL
    pasr_weight: float = 10.0


def get_config() -> ConfigDict:
    """
    Define the default configuration using ml_collections.ConfigDict.

    Returns:
        config: ConfigDict object with default pipeline configurations.
    """
    config = ConfigDict()
    default_config = PipelineConfig()
    for key, value in asdict(default_config).items():
        config[key] = value
    return config