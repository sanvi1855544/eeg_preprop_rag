from sentence_transformers import SentenceTransformer
import json
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("datasets/combined/corpus.jsonl", "r") as f:
    docs = [json.loads(line) for line in f]

texts = [doc["text"] for doc in docs]
doc_ids = [doc["_id"] for doc in docs]

embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Save as numpy array + index metadata
np.save("output3/corpus_embeddings.npy", embeddings)

with open("output3/corpus_index.json", "w") as f:
    json.dump([{"_id": d["_id"], "text": d["text"]} for d in docs], f)