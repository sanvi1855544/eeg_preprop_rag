import json

INPUT_PATH = "/p3/home/spal2/eeg_preprop_rag/results_sentence-transformers/mne_retrieval.json"
OUTPUT_PATH = "/p3/home/spal2/eeg_preprop_rag/results_sentence-transformers/mne_retrieval_flat.json"

flat_data = []

with open(INPUT_PATH, "r") as f:
    for line in f:
        entry = json.loads(line)
        for query_id, doc_scores in entry.items():
            for doc_id, score in doc_scores.items():
                flat_data.append({
                    "query_id": query_id,
                    "code_id": doc_id,
                    "score": score
                })

with open(OUTPUT_PATH, "w") as f:
    json.dump(flat_data, f, indent=2)

print(f"Flattened {len(flat_data)} records to {OUTPUT_PATH}")
