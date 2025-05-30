import json
from pathlib import Path
import os

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(l) for l in f]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    queries_path = "/home/spal2/eeg_preprop_rag/retrieval/datasets/mne_ir_dataset/queries.jsonl"
    corpus_path = "/home/spal2/eeg_preprop_rag/retrieval/datasets/mne_ir_dataset/corpus.jsonl"
    topk_path = "/home/spal2/eeg_preprop_rag/results_sentence-transformers/topk_retrieval.json"
    references_path = "datasets/mne_ir_dataset/code.jsonl"  
    output_path = "/home/spal2/eeg_preprop_rag/generation/test.jsonl"

    queries = {q["_id"]: q["text"] for q in load_jsonl(queries_path)}
    corpus = {c["_id"]: c["text"] for c in load_jsonl(corpus_path)}
    topk = load_json(topk_path)
    references = {r["_id"]: r["text"] for r in load_jsonl(references_path)} if Path(references_path).exists() else {}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f_out:
        for qid, results in topk.items():
            prompt = queries[qid]
            docs = [{"text": corpus[e["code_id"]]} for e in results if e["code_id"] in corpus]
            entry = {"prompt": prompt, "docs": docs}
            if qid in references:
                entry["code"] = references[qid]
            json.dump(entry, f_out)
            f_out.write("\n")
    print(f"Wrote RAG dataset to: {output_path}")

if __name__ == "__main__":
    main()
