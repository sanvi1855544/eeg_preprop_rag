import json
import os
from pathlib import Path
import functools
from sentence_transformers import SentenceTransformer
from eval.base import Task  # Assuming you're using the DS-1000 task base
from eval.utils import extract_code_pieces
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

TASK_NAME = "pyprep"

def format_prompt(example):
    return f'"""{example["prompt"]}"""\n\ndef'

class PyprepTask(Task):
    def __init__(self, dataset_path="datasets/combined", split="test", data_files=None, cache_dir=None, **kwargs):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=None,
            data_files=data_files,
            cache_dir=cache_dir,
            stop_words=["</code>", "# SOLUTION END"],
            requires_execution=False  # set to True if you evaluate by executing output
        )
        self.dataset_path = dataset_path
        self.split = split
        self.dataset = self.get_dataset()

        ##Below here newly added
       # with open("outputs2/results_pyprep.json") as f: ##This is wrong
        #    self.similarity_results = json.load(f)

        '''with open("/p3/home/abaxter/eeg_preprop_rag/datasets/combined/corpus.jsonl") as f:
            self.corpus = {
                entry["_id"]: entry["text"]
                for entry in map(json.loads, f)
            }'''
        corpus_file = os.path.join(dataset_path, "corpus.jsonl")
        with open(corpus_file) as f:
            self.corpus = {
                entry["_id"]: entry["text"]
                for entry in map(json.loads, f)
            }

        self.topk_docs = 5  # Or any k you want

    '''def get_prompt(self, doc, return_dict=False):
        prompt = format_prompt(doc)
        return {"prompt": prompt} if return_dict else prompt
        ##HERE: I think I should be changing this'''

    '''def get_top_k_context(query_id, similarity_data, k=5):
        # Get dict of related snippets and scores
        similarity_scores = similarity_data.get(query_id, {})
        # Sort by similarity score (descending) and take top-k
        top_k = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [item[0] for item in top_k]'''

    def get_prompt(self, doc, return_dict: bool = False, topk: int = 2, user_prompt=None):
        query_id = doc["_id"]
        prompt = doc["prompt"]
        #sol = doc["canonical_solution"]

        if user_prompt is not None and user_prompt != "None":
            prompt = user_prompt
        #prompt = "How do I run inference using a trained CBraMod model to generate predictions?"

        # Load corpus embeddings and metadata
        corpus_embeddings = np.load("retrieval/combined_embeddings_repoeval_all/corpus_embeddings.npy")

        with open("retrieval/combined_embeddings_repoeval_all/corpus_index.json", "r") as f:
            corpus_index = json.load(f)

        # Embed the new prompt
        model = SentenceTransformer("all-MiniLM-L6-v2")

        query_embedding = model.encode([prompt], convert_to_numpy=True)

        # Compute similarity
        scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
        topk_indices = scores.argsort()[-topk:][::-1]

        top_docs = [corpus_index[i] for i in topk_indices]
        top_docs_text = "\n\n".join(doc["text"] for doc in top_docs)
        full_prompt = f"""{prompt}
        \n---**Documentation**\n{top_docs_text}"""
        print(full_prompt)
        return full_prompt


    def get_reference(self, doc):
        ref = doc["canonical_solution"]
        print("Reference:\n")
        print(ref)
        return ref

    def postprocess_generation(self, generation, idx=None, new_tokens_only=False):
        for stop in self.stop_words:
            generation = generation.split(stop)[0]
        if "```python\n" in generation:
            generation = extract_code_pieces(generation, prefix="```python")
        elif "```\n" in generation:
            generation = extract_code_pieces(generation, prefix="```")
        return generation.strip()

    def process_results(self, generations, references):
        num_correct = 0
        for i, ref in enumerate(references):
            for gen in generations[i]:
                if gen.strip() == ref.strip():
                    num_correct += 1
        accuracy = num_correct / len(references) / len(generations[0])
        return {f"mean pass@1 accuracy ({len(generations[0])} samples)": accuracy}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def get_dataset(self):
        with open(Path(self.dataset_path) / "queries.jsonl", "r") as f:
            queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in f}
        with open(Path(self.dataset_path) / "corpus.jsonl", "r") as f:
            docs = {json.loads(l)["_id"]: json.loads(l)["text"] for l in f}
        with open(Path(self.dataset_path) / "qrels" / f"{self.split}.tsv", "r") as f:
            lines = f.readlines()[1:]
            qrels = [line.strip().split("\t") for line in lines]

        dataset = []
        for query_id, corpus_id, score in qrels:
            if int(score) == 1:
                dataset.append({
                    "_id": query_id,  
                    "prompt": queries[query_id],
                    "canonical_solution": docs[corpus_id]
                })
        return dataset


def create_all_tasks():
    return {
        TASK_NAME: PyprepTask
    }