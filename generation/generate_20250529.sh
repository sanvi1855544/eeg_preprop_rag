#!/bin/bash
DEFAULT_PROMPT=None
DEFAULT_DATASET_PATH="/p3/home/apasupat/fresh_clone_copy/"

USER_PROMPT="${1:-$DEFAULT_PROMPT}"
DATASET_PATH="${2:-$DEFAULT_DATASET_PATH}"

python3.10 main.py \
  --model_backend api \
  --precision bf16 \
  --task "pyprep" \
  --model "gpt-3.5-turbo" \
  --dataset_path "${DATASET_PATH}datasets/combined/" \
  --limit 1 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "${DATASET_PATH}output3/all_gpt4_generations.json" \
  --save_references_path "${DATASET_PATH}output3/all_test_gpt4_references.json" \
  --generation_only \
  --user_prompt "$USER_PROMPT" \
  --print_generation 