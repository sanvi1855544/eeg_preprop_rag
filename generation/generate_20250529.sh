#!/bin/bash
#export OPENAI_API_KEY=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
export CUDA_VISIBLE_DEVICES=1 ##used to be just 1
mkdir -p generation_output

# No need for CUDA_VISIBLE_DEVICES since OpenAI runs remotely
python3 main.py \
  --model_backend vllm \
  --precision bf16 \
  --task "pyprep" \
  --load_in_4bit \
  --model "meta-llama/Llama-2-7b-chat-hf" \
  --dataset_path "/p3/home/abaxter/eeg_preprop_rag/datasets/pyprep" \
  --limit 10 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path /p3/home/apasupat/fresh_clone/generation/generation_output/vllm_generations.json
