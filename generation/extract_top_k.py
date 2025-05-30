import json
from collections import defaultdict


input_file = "/home/spal2/eeg_preprop_rag/results_sentence-transformers/mne_retrieval_flat.json"
output_file = "/home/spal2/eeg_preprop_rag/results_sentence-transformers/topk_retrieval.json"
top_k = 3

with open(input_file, "r") as f:
    retrieval_data = json.load(f)


query_to_hits = defaultdict(list)
for item in retrieval_data:
    query_to_hits[item["query_id"]].append(item)


topk_results = {}
for query_id, hits in query_to_hits.items():
    sorted_hits = sorted(hits, key=lambda x: x["score"], reverse=True)
    topk_results[query_id] = sorted_hits[:top_k]


with open(output_file, "w") as f:
    json.dump(topk_results, f, indent=2)

print(f"Top-{top_k} results saved to {output_file}")
