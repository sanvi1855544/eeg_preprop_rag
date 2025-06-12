#!/bin/bash
export OPENAI_API_KEY=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
DEFAULT_PROMPT=None
DEFAULT_DATASET_PATH="/p3/home/abaxter/eeg_preprop_rag/"
#DEFAULT_DATASET_PATH="/content/eeg_preprop_rag/"

USER_PROMPT="${1:-$DEFAULT_PROMPT}"
DATASET_PATH="${2:-$DEFAULT_DATASET_PATH}"

python3 generation/main.py \
  --model_backend api \
  --precision bf16 \
  --task "pyprep" \
  --model "gpt-3.5-turbo" \
  --dataset_path "${DATASET_PATH}datasets/combined/" \
  --limit 1 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "${DATASET_PATH}output3/gpt4_generations.json" \
  --save_references_path "${DATASET_PATH}output3/test_gpt4_references.json" \
  --generation_only \
  --user_prompt "$USER_PROMPT" \
  --print_generation 
