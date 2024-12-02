# An Objective for Nuanced LLM Jailbreaks

This codebase implements the target prefix generation pipeline from our [nuanced LLM jailbreaks]() paper.
For a given user requests and victim LLM, the pipeline automatically generates and selects target prefixes.
Replacing the original "Sure, here is ..." with these prefixes enables more nuanced jailbreak attacks.

[**[arXiv]()**]

![Alt Text](assets/intro_fig.jpg)

## Pre-Generated Target Prefixes  (Ready to Use)
We pre-generated target prefixes for commonly used jailbreak requests in [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv) for users to directly use and evaluate.
These prefixes are stored in the `data` folder.
Note: The CSV files do not include the original jailbreak requests (i.e. missing the 'goal' column) to avoid directly hosting the original AdvBench dataset. 
Please run ```python  recover_requests.py`` to recover the original jailbreak requests.

We processed 100 jailbreak requests in advance, including 50 curated by us and 50 from [PAIR](https://github.com/patrickrchao/JailbreakingLLMs/blob/main/data/harmful_behaviors_custom.csv). For these requests, we considered four victim LLMs: llama-2, 3, 3.1, and Gemma-2.


## Quick Start

1. **Create conda environment (optional):**

    ```bash
    conda create -n nuancedjb python=3.11
    conda activate nuancedjb
    pip install -r requirements.txt
    ```


2. **Generating prefixes for Llama-3-8B:**
   ```bash
   python pipeline.py \
      --config="./default_config.py" \
      --config.victim_model="meta-llama/Meta-Llama-3-8B-Instruct" \
      --config.input_csv="./input/demo_requests.csv" \
      --config.output_dir="./output" \
      --config.start_step=1
   ```

3. **(Optional) Loading from an existing checkpoint and redo selection with a different weight for PASR:**
   ```bash
   python pipeline.py \
      --config="./default_config.py" \
      --config.victim_model="meta-llama/Meta-Llama-3-8B-Instruct" \
      --config.input_csv="./input/demo_requests.csv" \
      --config.output_dir="./output" \
      --config.start_step=7 \
      --config.run_id={existing run id like ABCD}
   ```

## Optional Arguments
Please refer to `default_config.py` for more optional arguments and their descriptions.


## License
This repository is made available under a CC-by-NC license, however you may have other legal obligations that govern your use of other content, such as the terms of service for third party models.


## Citation
If you find our work helpful, please cite it with
```bibtex

```