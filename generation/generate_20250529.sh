#!/bin/bash
DEFAULT_PROMPT=None
DEFAULT_DATASET_PATH="/p3/home/apasupat/fresh_clone_copy/"

export OPENAI_API_KEY=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

USER_PROMPT="${1:-$DEFAULT_PROMPT}"
DATASET_PATH="${2:-$DEFAULT_DATASET_PATH}"

python3.10 generation/main.py \
  --model_backend api \
  --precision bf16 \
  --task "pyprep" \
  --model "gpt-3.5-turbo" \
  --dataset_path "${DATASET_PATH}datasets/combined/" \
  --limit 1 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "${DATASET_PATH}output3/all_gpt4_generations.json" \
  --user_prompt "$USER_PROMPT" \
  --print_generation  \
  --metric_output_path "${DATASET_PATH}output3/all_test_gpt4_evaluation_results.json"