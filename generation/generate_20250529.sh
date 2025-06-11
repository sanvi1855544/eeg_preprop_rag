#!/bin/bash

export OPENAI_API_KEY=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

DEFAULT_PROMPT=None
USER_PROMPT="${1:-$DEFAULT_PROMPT}"


# No need for CUDA_VISIBLE_DEVICES since OpenAI runs remotely
python3 generation/main.py \
  --model_backend api \
  --precision bf16 \
  --task "pyprep" \
  --model "gpt-3.5-turbo" \
  --dataset_path "/p3/home/abaxter/eeg_preprop_rag/datasets/combined/" \
  --limit 1 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path /p3/home/abaxter/eeg_preprop_rag/output3/gpt4_generations.json \
  --save_references_path /p3/home/abaxter/eeg_preprop_rag/output3/test_gpt4_references.json \
  --generation_only \
  --print_generation \
  --user_prompt "$USER_PROMPT"



