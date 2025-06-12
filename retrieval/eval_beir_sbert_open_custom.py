import os
import argparse
import pickle
import glob
import json
import random
import logging
import csv
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from beir import LoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def get_top_docs(results, corpus, task_id: str, topk: int = 10) -> list[str]:
    if task_id not in results:
        return []
    doc_scores = results[task_id]
    doc_scores_sorted = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    doc_scores_sorted = doc_scores_sorted[:topk]
    doc_code_snippets = [corpus[code_id] for code_id, score in doc_scores_sorted]
    return doc_code_snippets

def main():
    if os.path.exists(args.results_file):
        os.remove(args.results_file)

    model = SentenceTransformer(args.model)

    retriever = None  # (You can instantiate EvaluateRetrieval here if needed)

    documents, doc_ids = [], []

    # Load corpus either from HF dataset or local TSV corpus file
    if args.hf_dataset is not None:
        logging.info(f"Loading corpus from Hugging Face dataset: {args.hf_dataset}")
        corpus_data = list(load_dataset(args.hf_dataset)["train"])
        for idx, passage in enumerate(corpus_data):
            passage["id"] = f"{args.hf_dataset.split('/')[-1]}_{idx}"
            if "text" not in passage:
                passage["text"] = passage.get("doc_content", "")
        corpus = {doc["id"]: doc for doc in corpus_data}

    elif args.local_corpus is not None:
        logging.info(f"Loading corpus from local TSV: {args.local_corpus}")
        corpus_data = []
        with open(args.local_corpus, 'r', encoding='utf-8') as f:
            tsvreader = csv.reader(f, delimiter='\t')
            for idx, row in enumerate(tsvreader):
                doc = {}
                if len(row) == 3:
                    doc["id"], doc["title"], doc["text"] = row
                elif len(row) == 2:
                    doc["id"], doc["text"] = row
                    doc["title"] = ""
                elif len(row) == 1:
                    doc["id"] = str(idx)
                    doc["title"] = ""
                    doc["text"] = row[0]
                corpus_data.append(doc)
        corpus = {doc["id"]: doc for doc in corpus_data}

    else:
        logging.error("Either --hf_dataset or --local_corpus must be provided.")
        return

    doc_ids = list(corpus.keys())
    documents = {doc_id: corpus[doc_id]["text"] for doc_id in doc_ids}

    # Load precomputed document embeddings from given path (supports glob pattern)
    all_embeddings = {}
    for fn in glob.glob(args.embedding_path):
        ids, embeddings = pickle.load(open(fn, "rb"))
        for id_, embedding in zip(ids, embeddings):
            all_embeddings[id_] = embedding

    documents_embeddings = [all_embeddings[doc_id] for doc_id in doc_ids]

    # Load BEIR dataset from the independent directory path
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_path}")

    logging.info(f"Loading BEIR dataset from directory: {args.dataset_path}")
    corpus_beir, queries, qrels = GenericDataLoader(data_folder=args.dataset_path).load(split="test")

    corpus_ids, query_ids = list(corpus_beir), list(queries)

    results = {}
    query_embeddings = []

    # Encode queries in batches
    for i in tqdm(range(0, len(queries), args.batch_size), desc="Encoding queries"):
        end = min(len(queries), i + args.batch_size)
        batch_ids = query_ids[i:end]
        batch_texts = [queries[q_id] for q_id in batch_ids]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)
        query_embeddings.extend(batch_embeddings)

    assert len(query_embeddings) == len(queries)

    # Compute dot product similarity between queries and document embeddings
    for idx in tqdm(range(len(queries)), desc="Computing similarities"):
        query_id = query_ids[idx]
        query_embedding = query_embeddings[idx]
        similarities = np.dot(documents_embeddings, query_embedding.cpu())
        results[query_id] = {str(doc_id): float(score) for doc_id, score in zip(doc_ids, similarities)}

    # Save results to file
    with open(args.results_file, 'w', encoding='utf-8') as fw:
        for query_id in results:
            fw.write(json.dumps({query_id: results[query_id]}) + "\n")

    # Display a random query and its top 3 documents
    top_k = 3
    query_id, ranking_scores = random.choice(list(results.items()))
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    logging.info(f"Query : {queries[query_id]}")

    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        title = corpus.get(doc_id, {}).get("title", "")
        text = corpus.get(doc_id, {}).get("text", "")
        logging.info(f"Rank {rank+1}: {doc_id} [{title}] - {text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset directory containing corpus.jsonl, queries.jsonl, qrels.tsv")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformer model name or path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for query encoding")
    parser.add_argument("--embedding_path", type=str, required=True, help="Glob pattern path to encoded embeddings pickle files")
    parser.add_argument("--results_file", type=str, required=True, help="File path to save retrieval results JSONL")
    parser.add_argument("--hf_dataset", type=str, default=None, help="Optional HF dataset name to load corpus")
    parser.add_argument("--local_corpus", type=str, default=None, help="Optional local corpus TSV file path")
    args = parser.parse_args()

    main()
