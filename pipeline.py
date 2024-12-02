# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Prefix generation pipeline for language models.

This module implements a complete pipeline for generating, filtering, and selecting prefixes
using uncensored and target language models. The pipeline follows a step-by-step process
with checkpointing capabilities.
"""

import os
import uuid
import logging
from typing import List, Dict
import multiprocessing
import pandas as pd
import torch
import torch.cuda
import gc
from ml_collections import config_flags, ConfigDict
from absl import app, flags


from llm_generator import DualLLMGenerator, SingleLLMGenerator
from preprocessor import PreprocessConfig, PrefixPreprocessor
from scorer_nll import CECalculatorConfig, PrefixCECalculator
from completer import CompletionConfig, PrefixCompleter
from scorer_pasr import (EvaluatorConfig, NuancedEvaluator, HarmBenchEvaluator,
                         JailbreakBenchEvaluator, StrongRejectEvaluator)
from selector import PrefixSelectorConfig, PrefixSelector


os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

_CONFIG = config_flags.DEFINE_config_file('config', default=None)


# Custom chat templates for specific models
CUSTOM_CHAT_TEMPLATES = {
    "georgesung/llama2_7b_chat_uncensored": "<s>### HUMAN:\n{content}\n\n### RESPONSE:\n",
    "Tap-M/Luna-AI-Llama2-Uncensored": '<s>USER: {content}\n\nASSISTANT:',
}

SURROGATE_ATTACK_PROMPTS = {
    0: '',  # No suffix
    1: '. Start your response with "{prefix}".',
    2: ' ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !',
}


class PrefixGenerationPipeline:
    """
    Main pipeline class for generating and selecting prefixes.

    This pipeline implements a multi-step process:
    1. Generate initial prefixes using uncensored models
    2. Merge multiple prefix sources
    3. Pre-filter generated prefixes
    4. Calculate cross-entropy for remaining prefixes
    5. Final filtering and selection
    """

    def __init__(self, config: ConfigDict):
        """
        Initialize the pipeline with configuration.

        Args:
            config: PipelineConfig object containing all pipeline parameters
        """
        self.config = config
        self.run_id = config.run_id or str(uuid.uuid4())[:4]
        self.run_dir = os.path.join(config.output_dir, f"run_{self.run_id}")
        self._setup_logging()

        # Initialize preprocessor
        self.preprocess_config = PreprocessConfig(
            model_id=config.victim_model,
            min_token_length=config.min_token_length,
            max_ce=config.max_ce,
            max_token_segments=config.max_token_segments,
            n_candidates_per_goal=config.n_candidates_per_goal  # Pass top_k to preprocessor
        )
        self.preprocessor = PrefixPreprocessor(self.preprocess_config)

    def _setup_logging(self):
        """Configure logging to both file and console."""
        os.makedirs(self.run_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.run_dir, 'pipeline.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Log configuration details
        self.logger.info(f"Pipeline initialized with run ID: {self.run_id}")
        self.logger.info(f"Output directory: {self.run_dir}")
        self.logger.info(f"Configuration: {vars(self.config)}")

    def _get_checkpoint_path(self, step: int) -> str:
        """Get path for checkpoint file of a specific step."""
        return os.path.join(self.run_dir, f"step{step}_output.csv")

    def _clear_gpu_memory(self):
        """Clear GPU memory and ensure synchronization."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def run(self):
        """Execute the complete pipeline, starting from the specified step."""
        try:
            if self.config.start_step <= 1:
                self.step1_generate_prefixes()
            if self.config.start_step <= 2:
                self.step2_pre_filter_phase1()
            if self.config.start_step <= 3:
                self.step3_ablate()
            if self.config.start_step <= 4:
                self.step4_compute_ce()
            if self.config.start_step <= 5:
                self.step5_pre_filter_phase2()
            if self.config.start_step <= 6:
                self.step6_get_completions()
            if self.config.start_step <= 7:
                self.step7_evaluate_responses()
            if self.config.start_step <= 8:
                self.step8_aggregate_evaluations()
            if self.config.start_step <= 9:
                self.step9_select_prefixes()

        except Exception as e:
            self.logger.error(f"Pipeline failed at step {self.config.start_step}: {str(e)}")
            raise

    def step1_generate_prefixes(self):
        """Generate initial prefixes using both uncensored and guided approaches."""
        # uncensored_model - guided/single - greedy/random - meta-prefixes
        self.logger.info("Starting Step 1: Generating prefixes")

        # Read input CSV
        df = pd.read_csv(self.config.input_csv)
        self.logger.info(f"Loaded {len(df)} requests from input CSV")

        goals = df['goal'].tolist()
        all_results = []

        for uncensored_model in self.config.uncensored_models:
            self.logger.info(f"Processing uncensored model: {uncensored_model}")

            # Try additional guided generation first
            self.logger.info("Trying guided decoding with victim model")
            results_guided = self._generate_prefixes(goals, uncensored_model, is_guided=True)
            all_results.extend(results_guided)
            self.logger.info(f"Generated {len(results_guided)} guided prefixes")
            self._clear_gpu_memory()

            # Generate prefixes using uncensored model alone
            self.logger.info("Generating with uncensored model alone")
            results_single = self._generate_prefixes(goals, uncensored_model, is_guided=False)
            all_results.extend(results_single)
            self.logger.info(f"Generated {len(results_single)} single-model prefixes")
            self._clear_gpu_memory()

        # Save results
        results_df = pd.DataFrame(all_results)
        output_path = self._get_checkpoint_path(1)
        results_df.to_csv(output_path, index=False)
        self.logger.info(f"Step 1 complete. Generated {len(results_df)} total prefixes")
        self.logger.info(f"Results saved to {output_path}")

    def _generate_prefixes(
            self,
            unique_goals,
            uncensored_model: str,
            is_guided: bool
    ) -> List[Dict]:
        """
        Generate prefixes using both guided and single LLM approaches.

        Args:
            goals
            uncensored_model: Name of the uncensored model
            is_guided: Boolean indicating if guided decoding can be used

        Returns:
            List of generated prefix dictionaries
        """
        results = []

        if is_guided:
            try:
                # Attempt to initialize DualLLMGenerator to check vocabulary compatibility
                generator = DualLLMGenerator(
                    original_model_name=self.config.victim_model,
                    uncensored_model_name=uncensored_model,
                )
                self.logger.info("Successfully initialized DualLLMGenerator for guided decoding")
            except ValueError as e:
                if "Vocabulary" in str(e):
                    self.logger.info(f"Vocabularies don't match for {uncensored_model}, skipping guided generation")
                else:
                    self.logger.error(f"Error initializing DualLLMGenerator: {e}")
                    raise
                return results
        else:
            generator = SingleLLMGenerator(
                model_name=uncensored_model,
                device_ids=None  # Use default device IDs
            )
            self.logger.info(f"Initialized SingleLLMGenerator for {uncensored_model}")

        # Generate with both greedy and random sampling
        for do_sample in [False, True]:  # False = greedy, True = random
            # Prepare inputs
            formatted_inputs, goals, meta_prefixes = self._construct_prompts(
                unique_goals,
                generator.tokenizer,
                self.config.meta_prefixes,
                self.config.meta_prefix_samples,
                m_uncensored=uncensored_model,
                do_sample=do_sample
            )

            # Generate completions
            completions = generator.complete(
                texts=formatted_inputs,
                batch_size=self.config.batch_size,
                max_new_tokens=self.config.max_new_tokens,
                guided_topk=self.config.guided_topk,
                temperature=self.config.temperature if do_sample else 1e-2
            )

            # Handle the extra space from <s>[INST] to <s> [INST]
            delta_len = len(generator.tokenizer.decode(generator.tokenizer.encode(
                formatted_inputs[0], add_special_tokens=False))) - len(formatted_inputs[0])

            # Process completions
            prefixes = [mp + p[len(fi) + delta_len:] for mp, fi, p in
                       zip(meta_prefixes, formatted_inputs, completions)]

            # Store results
            model_name = (f"{uncensored_model}-guided-{self.config.victim_model}"
                         if is_guided else uncensored_model)

            results.extend([{
                "goal": goal,
                "prefix": prefix,
                "meta_prefix": meta_prefix,
                "temperature": self.config.temperature if do_sample else 0.0,
                "model_name": model_name,
            } for goal, prefix, meta_prefix in zip(goals, prefixes, meta_prefixes)])

        del generator
        self._clear_gpu_memory()

        return results

    @staticmethod
    def _construct_prompts(goals, tokenizer, meta_prefixes, meta_prefixes_n_samples, m_uncensored, do_sample):
        """ Construct prompts with potential prefilling and repeated sampling
        input: df
        returns: goals, prefixes, meta_prefixes,
        """
        meta_prefixes_n_samples = meta_prefixes_n_samples if do_sample else [1 for _ in meta_prefixes]

        # construct expanded formatted input for random generation
        formatted_inputs = []
        expanded_goals = []
        expanded_meta_prefixes = []
        for goal in goals:
            for meta_prefix, n_samples in zip(meta_prefixes, meta_prefixes_n_samples):
                chat = [{"role": "user", "content": goal},]
                if m_uncensored in CUSTOM_CHAT_TEMPLATES.keys():
                    context = CUSTOM_CHAT_TEMPLATES[m_uncensored].format(content=chat[0]["content"])
                else:
                    context = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                context += meta_prefix
                formatted_inputs += [context] * n_samples
                expanded_goals += [goal] * n_samples
                expanded_meta_prefixes += [meta_prefix] * n_samples

        return formatted_inputs, expanded_goals, expanded_meta_prefixes

    def step2_pre_filter_phase1(self):
        """Apply initial filtering to generated prefixes."""
        self.logger.info("Starting Step 2: Initial prefix filtering")

        # Load data from previous step
        input_path = self._get_checkpoint_path(1)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file from step 1 not found: {input_path}")

        df = pd.read_csv(input_path)
        self.logger.info(f"Loaded {len(df)} prefixes from step 1")

        # Apply phase 1 filtering
        filtered_df = self.preprocessor.filter_phase1(df)

        # Save filtered results
        output_path = self._get_checkpoint_path(2)
        filtered_df.to_csv(output_path, index=False)
        self.logger.info(f"Step 2 complete. {len(filtered_df)} prefixes remaining")
        self.logger.info(f"Results saved to {output_path}")

    def step3_ablate(self):
        """Ablate filtered prefixes into shorter variations."""
        self.logger.info("Starting Step 3: Prefix ablation")

        input_path = self._get_checkpoint_path(2)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file from step 2 not found: {input_path}")

        df = pd.read_csv(input_path)
        self.logger.info(f"Loaded {len(df)} prefixes from step 2")

        # Perform ablation
        ablated_df = self.preprocessor.ablate(df)

        # Save ablated results
        output_path = self._get_checkpoint_path(3)
        ablated_df.to_csv(output_path, index=False)
        self.logger.info(f"Step 3 complete. Created {len(ablated_df)} ablated prefixes")
        self.logger.info(f"Results saved to {output_path}")

    def step4_compute_ce(self):
        """Calculate cross-entropy loss for ablated prefixes."""
        self.logger.info("Starting Step 4: Computing cross-entropy loss")

        input_path = self._get_checkpoint_path(3)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file from step 3 not found: {input_path}")

        df = pd.read_csv(input_path)
        self.logger.info(f"Loaded {len(df)} prefixes from step 3")

        # Initialize CE calculator
        ce_config = CECalculatorConfig(
            model_id=self.config.victim_model,
            batch_size=self.config.batch_size,
            surrogate_attack_prompt=SURROGATE_ATTACK_PROMPTS[self.config.surrogate_attack_prompt]
        )
        calculator = PrefixCECalculator(ce_config)

        # Calculate CE loss
        df_with_ce = calculator.calculate_ce(df)

        # Save results
        output_path = self._get_checkpoint_path(4)
        df_with_ce.to_csv(output_path, index=False)
        self.logger.info(f"Step 4 complete. {len(df_with_ce)} prefixes remaining")
        self.logger.info(f"Results saved to {output_path}")

        # Clear GPU memory
        del calculator
        self._clear_gpu_memory()

    def step5_pre_filter_phase2(self):
        """Apply CE-based filtering and selection to prefixes."""
        self.logger.info("Starting Step 5: CE-based filtering")

        input_path = self._get_checkpoint_path(4)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file from step 4 not found: {input_path}")

        df = pd.read_csv(input_path)
        self.logger.info(f"Loaded {len(df)} prefixes from step 4")

        # Apply phase 2 filtering (CE-based and top-k selection)
        filtered_df = self.preprocessor.filter_phase2(df)

        # Save filtered results
        output_path = self._get_checkpoint_path(5)
        filtered_df.to_csv(output_path, index=False)
        self.logger.info(f"Step 5 complete. {len(filtered_df)} prefixes remaining")
        self.logger.info(f"Results saved to {output_path}")

    def step6_get_completions(self):
        """Get completions for filtered prefixes using target LLM."""
        self.logger.info("Starting Step 6: Getting completions")

        input_path = self._get_checkpoint_path(5)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file from step 5 not found: {input_path}")

        df = pd.read_csv(input_path)
        self.logger.info(f"Loaded {len(df)} prefixes from step 5")

        # Create a separate process for completion generation
        process = multiprocessing.Process(
            target=self._run_completer,
            args=(df.copy(), self.run_dir, self.config)
        )
        process.start()
        process.join()

        if process.exitcode != 0:
            raise RuntimeError(f"Completion process failed with exit code {process.exitcode}")

        # Load results from intermediate file
        output_path = self._get_checkpoint_path(6)
        df_with_completions = pd.read_csv(output_path)
        self.logger.info(f"Step 6 complete. Generated {len(df_with_completions)} completions")

    @staticmethod
    def _run_completer(df, run_dir, config):
        try:
            completion_config = CompletionConfig(
                model_id=config.victim_model,
                batch_size=config.batch_size,
                max_new_tokens=config.max_new_tokens_completion,
                temperature=config.temperature,
                n_samples=config.n_samples,
                surrogate_attack_prompt=SURROGATE_ATTACK_PROMPTS[config.surrogate_attack_prompt]
            )
            completer = PrefixCompleter(completion_config)
            df_with_completions = completer.get_completions(df)

            # Save results
            output_path = os.path.join(run_dir, "step6_output.csv")
            df_with_completions.to_csv(output_path, index=False)
            logging.info(f"Saved completions to {output_path}")

        except Exception as e:
            logging.error(f"Error occurred while running completer: {str(e)}")
            logging.exception("Full traceback:")
            raise

        finally:
            # Cleanup
            del completer
            gc.collect()
            torch.cuda.empty_cache()

    def step7_evaluate_responses(self):
        """Evaluate completions using specified judges."""
        self.logger.info("Starting Step 7: Evaluating responses")

        input_path = self._get_checkpoint_path(6)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file from step 6 not found: {input_path}")

        original_df = pd.read_csv(input_path)
        self.logger.info(f"Loaded {len(original_df)} responses from step 6")

        evaluator_config = EvaluatorConfig(
            batch_size=self.config.batch_size_judge,
            max_new_tokens_eval=self.config.max_new_tokens_eval,
            filter_len=self.config.filter_len
        )

        successful_judges = []
        failed_judges = []

        for judge_type in self.config.judges:
            self.logger.info(f"Running evaluation with {judge_type} judge")

            # Create a separate process for each evaluator
            process = multiprocessing.Process(
                target=self._run_evaluator,
                args=(judge_type, evaluator_config, original_df.copy(), self.run_dir)
            )
            process.start()
            process.join()

            if process.exitcode == 0:
                successful_judges.append(judge_type)
            else:
                self.logger.error(f"Evaluator {judge_type} failed with exit code {process.exitcode}")
                failed_judges.append(judge_type)

        # Reload and merge all intermediate results
        final_df = original_df.copy()
        for judge_type in successful_judges:
            intermediate_path = os.path.join(self.run_dir, f"step7_intermediate_{judge_type}.csv")
            judge_df = pd.read_csv(intermediate_path)

            # Get only the evaluation columns for this judge
            if judge_type == "nuanced":
                eval_cols = ["eval_nj", "explanation_nj"]
            elif judge_type == "jailbreakbench":
                eval_cols = ["eval_jb", "explanation_jb"]
            elif judge_type == "harmbench":
                eval_cols = ["eval_hb", "explanation_hb"]
            elif judge_type == "strongreject":
                eval_cols = ["eval_sj", "eval_sj_binary"]

            # Merge only the evaluation columns using the join keys
            merge_keys = ['goal', 'prefix', 'completion']
            final_df = final_df.merge(
                judge_df[merge_keys + eval_cols],
                on=merge_keys,
                how='left'
            )

        # Save final results
        output_path = self._get_checkpoint_path(7)
        final_df.to_csv(output_path, index=False)

        # Log summary
        self.logger.info(f"Step 7 complete. Evaluated {len(final_df)} responses")
        if successful_judges:
            self.logger.info(f"Successfully completed judges: {', '.join(successful_judges)}")
        if failed_judges:
            self.logger.warning("Failed judges:")
            for judge_type in failed_judges:
                self.logger.warning(f"  - {judge_type}")
        self.logger.info(f"Final results saved to {output_path}")

    @staticmethod
    def _run_evaluator(judge_type, evaluator_config, df, run_dir):
        evaluator_map = {
            "nuanced": NuancedEvaluator,
            "jailbreakbench": JailbreakBenchEvaluator,
            "harmbench": HarmBenchEvaluator,
            "strongreject": StrongRejectEvaluator
        }

        evaluator_class = evaluator_map.get(judge_type)
        if not evaluator_class:
            logging.warning(f"Unknown judge type: {judge_type}, skipping")
            return

        try:
            evaluator = evaluator_class(evaluator_config)
            evaluated_df = evaluator.evaluate(df)

            # Save intermediate results
            intermediate_path = os.path.join(run_dir, f"step7_intermediate_{judge_type}.csv")
            evaluated_df.to_csv(intermediate_path, index=False)
            logging.info(f"Saved intermediate results for {judge_type} judge to {intermediate_path}")

        except Exception as e:
            logging.error(f"Error occurred while running {judge_type} evaluator: {str(e)}")
            logging.exception("Full traceback:")

        finally:
            # Cleanup
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()

    def step8_aggregate_evaluations(self):
        """
        Aggregate evaluation results from different judges.
        Combines results from multiple evaluation samples and judges into single scores per prefix.
        """
        self.logger.info("Starting Step 8: Aggregating evaluation results")

        input_path = self._get_checkpoint_path(7)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file from step 7 not found: {input_path}")

        df = pd.read_csv(input_path)
        self.logger.info(f"Loaded {len(df)} evaluated responses from step 7")

        # Create a copy to avoid modifying the original
        analysis = df.copy()

        # Filter based on NLL if threshold is provided
        if self.config.max_ce is not None:
            analysis = analysis[analysis['prefix_nll'] < self.config.max_ce]
            self.logger.info(f"Filtered to {len(analysis)} responses based on NLL threshold")

        # Detect available judges based on column names
        available_judges = {}
        if 'eval_nj' in analysis.columns:
            available_judges['nuanced'] = 'eval_nj'
        if 'eval_jb' in analysis.columns:
            available_judges['jailbreak'] = 'eval_jb'
        if 'eval_hb' in analysis.columns:
            available_judges['harmbench'] = 'eval_hb'
        if 'eval_sj_binary' in analysis.columns:
            available_judges['strongreject'] = 'eval_sj_binary'

        if not available_judges:
            raise ValueError("No evaluation results found in the DataFrame")

        self.logger.info(f"Found judges: {list(available_judges.keys())}")

        # Perform aggregation
        grouped = analysis.groupby(['goal', 'prefix'])

        # First, aggregate the constant columns (taking first value)
        aggregated = grouped.agg({
            'prefix_nll': lambda x: x.iloc[0],
            'model_name': lambda x: x.iloc[0],
            'meta_prefix': lambda x: x.iloc[0],
            'temperature': lambda x: x.iloc[0],
        })

        # Add number of samples
        aggregated['n_eval_samples'] = grouped.size()

        # Then add judge-specific aggregations
        for _, col_name in available_judges.items():
            aggregated[f'{col_name}_mean'] = grouped[col_name].mean()
            aggregated[f'{col_name}_size'] = grouped[col_name].size()

        aggregated = aggregated.reset_index()

        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                              for col in aggregated.columns]
        aggregated = aggregated.reset_index()

        # Save results
        output_path = self._get_checkpoint_path(8)
        aggregated.to_csv(output_path, index=False)
        self.logger.info(f"Step 8 complete. Aggregated {len(aggregated)} prefix results")
        self.logger.info(f"Results saved to {output_path}")

        return aggregated

    def step9_select_prefixes(self):
        """Select final prefixes based on specified judges and selection criteria."""
        self.logger.info("Starting Step 9: Selecting final prefixes")

        input_path = self._get_checkpoint_path(8)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file from step 8 not found: {input_path}")

        df = pd.read_csv(input_path)
        self.logger.info(f"Loaded {len(df)} prefix candidates from step 8")

        # Initialize selector
        selector_config = PrefixSelectorConfig(
            pasr_weight=self.config.pasr_weight,
            n_prefixes_per_goal=self.config.n_prefixes_per_goal,
            judges=self.config.selection_judges
        )
        selector = PrefixSelector(selector_config)

        # Select prefixes
        selected_df = selector.select_prefixes(df)

        # Save results
        output_path = self._get_checkpoint_path(9)
        selected_df.to_csv(output_path, index=False)
        self.logger.info(f"Step 9 complete. Selected {len(selected_df)} prefixes")
        self.logger.info(f"Results saved to {output_path}")


def main(_):
    """Parse arguments and run the pipeline."""
    multiprocessing.set_start_method('spawn', force=True)

    config = _CONFIG.value
    print(config)

    pipeline = PrefixGenerationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    flags.FLAGS.mark_as_parsed()
    app.run(main)