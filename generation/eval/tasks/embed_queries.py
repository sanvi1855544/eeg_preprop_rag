import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load queries from JSONL
queries = {}
with open("datasets/pyprep/queries.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        queries[item["_id"]] = item["text"]

# Initialize model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode queries
query_ids = list(queries.keys())
query_texts = list(queries.values())
query_embeddings = model.encode(query_texts, show_progress_bar=True, convert_to_numpy=True)

# Save to disk
np.savez("query_embeddings.npz", ids=np.array(query_ids), embeddings=query_embeddings)
