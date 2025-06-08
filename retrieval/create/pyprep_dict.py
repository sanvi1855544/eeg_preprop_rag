import json

def build_doc_dict(query_id, queries, corpus, retrieval_results, topk=5):
    prompt_text = queries[query_id]["text"]
    docs_scores = retrieval_results.get(query_id, {})
    retrieved_docs = sorted(docs_scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    context_docs = [{"text": corpus[doc_id]["text"]} for doc_id, _ in retrieved_docs if doc_id in corpus]
    return {
        "query_id": query_id,
        "prompt": prompt_text,
        "data": {"docs": context_docs}
    }


def load_jsonl_to_dict(path, id_key):
    data_dict = {}
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            key = obj[id_key]
            data_dict[key] = obj
    return data_dict

def load_retrieval_results_jsonl(path):
    retrieval_results = {}
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            # Assuming obj is { query_id: [(doc_id, score), ...] }
            for query_id, retrieved in obj.items():
                retrieval_results[query_id] = retrieved
    return retrieval_results

def main():
    retrieval_results = load_retrieval_results_jsonl("output2/retrieval_results.json")

    queries = load_jsonl_to_dict("datasets/pyprep/queries.jsonl", id_key="_id")
    corpus = load_jsonl_to_dict("datasets/pyprep/corpus.jsonl", id_key="_id")

    prompt_docs = []
    for query_id in queries.keys():
        doc = build_doc_dict(query_id, queries, corpus, retrieval_results, topk=5)
        prompt_docs.append(doc)

    with open("output2/prompt_docs.json", "w") as f:
        json.dump(prompt_docs, f, indent=2)

if __name__ == "__main__":
    main()
