#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python main.py  --model_backend vllm --precision bf16  --task "mne-codegen"   --model "bigcode/starcoder2-7b"   --dataset_path "json"   --data_files_test "/p3/home/spal2/eeg_preprop_rag/results_sentence-transformers/mne_retrieval_flat.json"   --allow_code_execution   --save_generations   --save_generations_path mne_ir_outputs.json   --generation_only
