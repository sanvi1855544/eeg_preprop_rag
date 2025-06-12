#!/bin/bash

mkdir -p results
mkdir -p combined_results_repoeval_all

PYTHONPATH=. python -m create.repoeval_repo

#python generate_embeddings.py \
#    --model sentence-transformers/all-MiniLM-L6-v2 \
#    --output_dir cbramod_embeddings \
#    --passages ./beir_output/corpus.jsonl \
#    --shard_id 0 \
#    --num_shards 1

python generate_embeddings_custom.py \
  --model_name_or_path sentence-transformers/all-MiniLM-L6-v2 \
  --hf_datasets output/repoeval/datasets/function_level_completion_2k_context_codex.test.clean.jsonl \
  --passages ../datasets/combined/corpus.jsonl \
  --local_corpus_tsv ./local_corpus/combined_corpus_repoeval_all.tsv \
  --output_dir combined_embeddings_repoeval_all \
  --prefix combined_passages \
  --num_shards 1 \

python eval_beir_sbert_open_custom.py \
  --dataset_path ../datasets/combined/ \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch_size 64 \
  --embedding_path ./combined_embeddings_repoeval_all/combined_passages_00.pkl \
  --local_corpus ./local_corpus/combined_corpus_repoeval_all.tsv \
  --results_file combined_results_repoeval_all/results.json

#python eval_beir_sbert_open_custom.py \
#  --dataset humaneval \
#  --model sentence-transformers/all-MiniLM-L6-v2 \
#  --batch_size 64 \
#  --embdding_path "outputs/embeddings/*.pkl" \
#  --local_corpus data/corpus.tsv \
#  --results_file outputs/retrieval_results.jsonl \
#  --output_file outputs/topk_output.json