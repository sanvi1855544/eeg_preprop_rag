import json
from pathlib import Path
import functools
from eval.base import Task  # Assuming you're using the DS-1000 task base
from eval.utils import extract_code_pieces
import random
#import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity



TASK_NAME = "pyprep"


def format_prompt(example):
    return f'"""{example["prompt"]}"""\n\ndef'

class PyprepTask(Task):
    def __init__(self, dataset_path="datasets/pyprep_dataset", split="test", data_files=None, cache_dir=None, **kwargs):
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
        '''with open("retrieval/results_pyprep.json") as f:
            self.similarity_results = json.load(f)

        with open("datasets/pyprep/corpus.jsonl") as f:
            self.corpus = {
                entry["doc_id"]: entry["text"]
                for entry in map(json.loads, f)
            }'''

        self.topk_docs = 5  # Or any k you want


    '''def get_prompt(self, doc, return_dict=False):
        prompt = format_prompt(doc)
        return {"prompt": prompt} if return_dict else prompt
        ##HERE: I think I should be changing this'''

    def get_top_k_context(new_prompt, model, query_ids, query_embeddings, corpus, k=5):
        # Embed the new (random) prompt
        new_embedding = model.encode([new_prompt], convert_to_numpy=True)

        # Compute cosine similarity with stored query embeddings
        sims = cosine_similarity(new_embedding, query_embeddings)[0]
        top_indices = sims.argsort()[-k:][::-1]  # Top k indices

        # Get top-k (query_id, similarity) pairs
        top_matches = [(query_ids[i], sims[i]) for i in top_indices]

        # Transform query_id to code_id and retrieve associated code
        docs = []
        for qid, _ in top_matches:
            code_id = qid.replace("_doc", "_code")
            if code_id in corpus:
                docs.append(corpus[code_id])
        return docs

    def get_prompt(self, return_dict: bool = False):
        """Generates a RAG prompt using top-k retrieved similar queries."""

        # --- STEP 1: Pick a random query ---
        #random_query_id = random.choice(self.query_ids)
        #random_query_text = self.queries[random_query_id]  # e.g., "How do I compute the G matrix?"
        query = "How do I interpolate noisy EEG channels using PyPREP?"

        # --- STEP 2: Get top-k similar query documents ---
        top_docs = get_top_k_context(
            new_prompt=random_query_text,
            model=self.model,
            query_ids=self.query_ids,
            query_embeddings=self.query_embeddings,
            corpus=self.corpus,
            k=self.topk_docs
        )

        # --- STEP 3: Retrieve the associated function stub (prompt) ---
        doc = self.documents[random_query_id]
        prompt = doc["prompt"]
        intent = doc.get("intent", "")

        # --- STEP 4: Build base function prompt ---
        try:
            function_head, function_prefix = prompt.split("\n", 1)
        except ValueError:
            function_head, function_prefix = prompt, ""

        docstring = f'    """{intent}\n    """' if intent else ""
        code_body = function_prefix.replace("\t", " " * 4)
        base_prompt = "\n".join([function_head, docstring, code_body])

        # --- STEP 5: Add top-k context ---
        context = ""
        if top_docs:
            instruction = "Please refer to the following documentation to generate the code:\n"
            context = instruction + "\n\n".join(top_docs)

        full_prompt = context + "\n" + base_prompt

        print(full_prompt)
        quit()  ##Gonna wanna get rid of this

        return {
            "prompt": base_prompt,
            "context": context,
            "full_prompt": full_prompt,
            "query_id": random_query_id
        } if return_dict else full_prompt

    def get_reference(self, doc):
        return doc["canonical_solution"]

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
                    "prompt": queries[query_id],
                    "canonical_solution": docs[corpus_id]
                })
        return dataset


def create_all_tasks():
    return {
        TASK_NAME: PyprepTask
    }
