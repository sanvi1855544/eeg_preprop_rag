import json


with open('results_sentence-transformers/mne_retrieval.json', 'r') as f:
    retrieval = [json.loads(line) for line in f.readlines()]

with open('retrieval/datasets/mne_ir_dataset/corpus.jsonl', 'r') as f:
    corpus = [json.loads(line) for line in f.readlines()]

with open('retrieval/datasets/mne_ir_dataset/queries.jsonl', 'r') as f:
    queries = [json.loads(line) for line in f.readlines()]



