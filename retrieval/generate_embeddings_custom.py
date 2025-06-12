import os
import argparse
import pickle
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import normalize_text
import slurm
import csv
import numpy as np
import json

def embed_passages(args, passages, model):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, p in enumerate(passages):
            batch_ids.append(p.get("id", p.get("_id", str(k))))
            text = p.get("text", "")
            if args.no_title or "title" not in p:
                # use text only
                pass
            else:
                text = p.get("title", "") + " " + text
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = normalize_text.normalize(text)
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:
                embeddings = model.encode(batch_text, convert_to_tensor=True)
                embeddings = embeddings.cpu()

                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                if k % 100000 == 0 and k > 0:
                    print(f"Encoded passages {total}")

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings

def save_corpus_as_tsv(passages, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for p in passages:
            pid = p.get("id", p.get("_id", ""))
            title = p.get("title", "")
            text = p.get("text", "")
            writer.writerow([pid, title, text])
    print(f"Saved combined corpus TSV to {output_path}")

def main(args):
    model = SentenceTransformer(args.model_name_or_path)

    hf_passages = []
    if args.hf_datasets:
        print(f"Loading Hugging Face dataset: {args.hf_datasets}")
        if os.path.exists(args.hf_datasets):
            hf_passages_raw = load_dataset("json", data_files=args.hf_datasets, split="train")
        else:
            hf_passages_raw = load_dataset(args.hf_datasets)["train"]

        hf_passages = [
            {
                "id": str(i),
                "text": p.get("prompt", "")
            }
            for i, p in enumerate(hf_passages_raw)
        ]
        print(f"Example HF passage: {hf_passages[0]}")

    custom_passages = []
    if args.passages:
        print(f"Loading custom dataset from: {args.passages}")
        custom_passages_raw = load_dataset("json", data_files=args.passages, split="train")

        custom_passages = [
            {
                "id": p.get("_id", f"custom_{i}"),
                "text": p.get("text", "")
            }
            for i, p in enumerate(custom_passages_raw)
        ]

    passages = hf_passages + custom_passages
    print(f"Total passages combined: {len(passages)}")

    if args.local_corpus_tsv:
        save_corpus_as_tsv(passages, args.local_corpus_tsv)

    shard_size = len(passages) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:
        end_idx = len(passages)

    passages = passages[start_idx:end_idx]
    print(f"Embedding generation for {len(passages)} passages (shard {args.shard_id}) from idx {start_idx} to {end_idx}.")

    allids, allembeddings = embed_passages(args, passages, model)

    # Save files in embedding_folder
    embedding_folder = args.output_dir
    os.makedirs(embedding_folder, exist_ok=True)

    # Save embeddings as numpy array
    npy_path = os.path.join(embedding_folder, "corpus_embeddings.npy")
    np.save(npy_path, allembeddings)
    print(f"Saved embeddings numpy array to {npy_path}")

    # Save corpus index json with id and text only
    index_path = os.path.join(embedding_folder, "corpus_index.json")
    corpus_index = [{"id": pid, "text": passages[i]["text"]} for i, pid in enumerate(allids)]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(corpus_index, f, ensure_ascii=False, indent=2)
    print(f"Saved corpus index json to {index_path}")

    # Save shard pickle file
    embedding_id = args.shard_id + args.embedding_start_idx
    save_file = os.path.join(embedding_folder, f"{args.prefix}_{embedding_id:02d}.pkl")
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)
    print(f"Saved shard pickle to {save_file}")

    print(f"Total passages processed {len(allids)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--passages", type=str, default=None, help="Path to custom passages (jsonl)")
    parser.add_argument("--hf_datasets", type=str, default=None, help="Hugging Face dataset name or path")
    parser.add_argument("--output_dir", type=str, default="combined_embeddings", help="Directory to save embeddings and index")
    parser.add_argument("--prefix", type=str, default="passages", help="Prefix for output files")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard index to process")
    parser.add_argument("--embedding_start_idx", type=int, default=0, help="Embedding file start index")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards total")
    parser.add_argument("--per_gpu_batch_size", type=int, default=512, help="Batch size for encoding")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="SentenceTransformer model name or path")
    parser.add_argument("--no_fp16", action="store_true", help="Disable fp16 inference")
    parser.add_argument("--no_title", action="store_true", help="Do not prepend title to passage text")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="Normalize text before encoding")
    parser.add_argument("--local_corpus_tsv", type=str, default=None,
                        help="Path to save combined corpus TSV (optional)")

    args = parser.parse_args()

    slurm.init_distributed_mode(args)

    main(args)
