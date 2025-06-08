import os
import json

# Paths
REPO_DIR = "./CBraMod"  # your cloned repo folder with code files
OUTPUT_DIR = "beir_output"

# 1. Collect code files as corpus
corpus = []
for root, _, files in os.walk(REPO_DIR):
    for file in files:
        if file.endswith(".py"):  # only Python files, or change as needed
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            doc_id = os.path.relpath(filepath, REPO_DIR)  # relative path as id
            corpus.append({"doc_id": doc_id, "text": text})

# 2. Define queries manually (example)
queries = [
    {"query_id": "q1", "text": "How to load EEG data?"},
    {"query_id": "q2", "text": "Preprocessing pipeline for EEG signals"},
]

# 3. Define qrels manually (example)
# You say which queries relate to which docs with relevance 1 or 0
qrels = [
    ("q1", "scripts/load_data.py", 1),
    ("q1", "scripts/preprocessing.py", 0),
    ("q2", "scripts/preprocessing.py", 1),
    ("q2", "scripts/utils.py", 0),
]

# Make output folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "qrels"), exist_ok=True)

# Save corpus.jsonl
with open(os.path.join(OUTPUT_DIR, "corpus.jsonl"), "w", encoding="utf-8") as f:
    for doc in corpus:
        f.write(json.dumps(doc) + "\n")

# Save queries.jsonl
with open(os.path.join(OUTPUT_DIR, "queries.jsonl"), "w", encoding="utf-8") as f:
    for query in queries:
        f.write(json.dumps(query) + "\n")

# Save qrels/test.tsv
with open(os.path.join(OUTPUT_DIR, "qrels", "test.tsv"), "w", encoding="utf-8") as f:
    for qid, docid, rel in qrels:
        f.write(f"{qid}\t{docid}\t{rel}\n")

print(f"BEIR files saved in {OUTPUT_DIR}")
