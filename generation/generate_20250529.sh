#!/bin/bash

export CUDA_VISIBLE_DEVICES=1 ##used to be just 1
#unset CUDA_VISIBLE_DEVICES
python main.py  --model_backend vllm  --precision bf16  --task "pyprep"  --load_in_4bit --model "meta-llama/Llama-2-7b-chat-hf"   --dataset_path "/p3/home/abaxter/eeg_preprop_rag/datasets/pyprep"  --allow_code_execution   --save_generations   --save_generations_path /p3/home/abaxter/eeg_preprop_rag/output2/starcoder2-7b_generations.json 

##15 instead of 7 bigcode/starcoder2-15b