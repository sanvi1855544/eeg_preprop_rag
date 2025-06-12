#!/bin/bash

export OPENAI_API_KEY=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

mkdir -p generation_output

# No need for CUDA_VISIBLE_DEVICES since OpenAI runs remotely
python3 main.py \
  --model_backend api \
  --precision bf16 \
  --task "pyprep" \
  --model "gpt-3.5-turbo" \
  --dataset_path "/p3/home/apasupat/fresh_clone/retrieval/cbramod_beir/" \
  --limit 10 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path /p3/home/apasupat/fresh_clone/generation/generation_output/gpt4_generations.json \
  --save_references_path /p3/home/apasupat/fresh_clone/generation/generation_output/test_gpt4_references.json