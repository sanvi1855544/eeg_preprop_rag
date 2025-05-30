#!/bin/bash

python eval_beir_pyserini_repo.py \
  --dataset mne \
  --output_metadir work_20250529 \
  --index_dir INDEX_DIR \
  --top_k TOP_K \
  --k1 K1 \
  --b B \
  --output_file PATH_TO_YOUR_SCORE_FILE \
  --results_file PATH_TO_YOUR_RETRIEVAL_RESULTS_FILE