import os
import json
import csv

DATASETS_DIR = "datasets"
COMBINED_DIR = os.path.join(DATASETS_DIR, "combined")

os.makedirs(COMBINED_DIR, exist_ok=True)

combined_corpus = []
combined_queries = []
combined_test_rows = []

# Get dataset folders (exclude "combined")
dataset_folders = [
    d for d in os.listdir(DATASETS_DIR)
    if os.path.isdir(os.path.join(DATASETS_DIR, d)) and d != "combined"
]

for dataset_name in dataset_folders:
    dataset_path = os.path.join(DATASETS_DIR, dataset_name)
    print(f"Processing {dataset_name}...")

    # Read and prefix corpus.jsonl
    corpus_path = os.path.join(dataset_path, "corpus.jsonl")
    with open(corpus_path, "r") as f:
        for line in f:
            doc = json.loads(line)
            doc["_id"] = f"{dataset_name}_{doc['_id']}"
            combined_corpus.append(doc)

    # Read and prefix queries.jsonl
    queries_path = os.path.join(dataset_path, "queries.jsonl")
    with open(queries_path, "r") as f:
        for line in f:
            query = json.loads(line)
            query["_id"] = f"{dataset_name}_{query['_id']}"
            combined_queries.append(query)

    # Read and prefix test.tsv
    test_path = os.path.join(dataset_path, "qrels/test.tsv")
    with open(test_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue  # skip malformed lines
            qid, docid, rel = row
            combined_test_rows.append([
                f"{dataset_name}_{qid}",
                f"{dataset_name}_{docid}",
                rel
            ])

# Save combined corpus.jsonl
corpus_out = os.path.join(COMBINED_DIR, "corpus.jsonl")
with open(corpus_out, "w") as f:
    for item in combined_corpus:
        f.write(json.dumps(item) + "\n")

# Save combined queries.jsonl
queries_out = os.path.join(COMBINED_DIR, "queries.jsonl")
with open(queries_out, "w") as f:
    for item in combined_queries:
        f.write(json.dumps(item) + "\n")

# Save combined test.tsv
test_out = os.path.join(COMBINED_DIR, "qrels/test.tsv")
with open(test_out, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(combined_test_rows)

print("✅ Combination complete!")
