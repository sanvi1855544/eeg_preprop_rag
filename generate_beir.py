import os
import json
import glob

REPO_DIR = "./CBraMod"          # Your local repo path
OUTPUT_DIR = "beir_output"
QUERIES_FILE = "./beir_output/queries.jsonl"

# === Step 1: Load queries ===
queries = []
with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        queries.append(json.loads(line))

# === Step 2: Build corpus from .py files ===
corpus = []
corpus_ids = set()

for root, _, files in os.walk(REPO_DIR):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            rel_path = os.path.relpath(filepath, REPO_DIR)
            corpus_ids.add(rel_path)
            corpus.append({"doc_id": rel_path, "text": text})

# === Step 3: Qrels — match queries where _id.split(".py")[0] + ".py" == doc_id ===
qrels = []
for query in queries:
    qid = query["_id"]
    if ".py" in qid:
        base_doc_id = qid.split(".py")[0] + ".py"
        if base_doc_id in corpus_ids:
            qrels.append((qid, base_doc_id, 1))

# === Step 4: Write corpus + qrels ===
os.makedirs(os.path.join(OUTPUT_DIR, "qrels"), exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "corpus.jsonl"), "w", encoding="utf-8") as f:
    for doc in corpus:
        f.write(json.dumps(doc) + "\n")

with open(os.path.join(OUTPUT_DIR, "qrels", "test.tsv"), "w", encoding="utf-8") as f:
    for qid, docid, rel in qrels:
        f.write(f"{qid}\t{docid}\t{rel}\n")

print(f"✅ Done: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")