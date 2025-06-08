import json
from pathlib import Path
import functools
from eval.base import Task  # Assuming you're using the DS-1000 task base
from eval.utils import extract_code_pieces

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
       # with open("outputs2/results_pyprep.json") as f: ##This is wrong
        #    self.similarity_results = json.load(f)

        with open("/p3/home/abaxter/eeg_preprop_rag/datasets/pyprep/corpus.jsonl") as f:
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


    def get_prompt(self, doc, return_dict: bool = False, topk: int = 1):
        query_id = doc["_id"]
        prompt = doc["prompt"]
        #sol = doc["canonical_solution"]
        json_path = "/p3/home/abaxter/eeg_preprop_rag/output2/prompt_docs.json"

        # Open and load the JSON content into a Python dict
        with open(json_path, "r", encoding="utf-8") as f:
            prompt_docs_list = json.load(f)
        retrieval_results = {item["query_id"]: item for item in prompt_docs_list}

        if query_id not in retrieval_results:
            print("BADDDD") ##eventually do gotta figure this out
            return ""

        docs_data = retrieval_results[query_id]["data"].get("docs", [])
        # Take topk docs, extract their text, join them
        top_docs_text = "\n\n".join(doc["text"] for doc in docs_data[:topk])

        # Optionally format a prompt, e.g. add the original question
        prompt = f"Use the following documentation to answer the question below.\n\nDocumentation:\n{top_docs_text}\n\nQuestion:\n{prompt}\n\nAnswer:"

        print(prompt) #Could get rid of this
        return prompt


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
                    "_id": query_id,  
                    "prompt": queries[query_id],
                    "canonical_solution": docs[corpus_id]
                })
        return dataset


def create_all_tasks():
    return {
        TASK_NAME: PyprepTask
    }
