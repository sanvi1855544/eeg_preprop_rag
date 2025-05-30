"""
Evaluating Large Language Models on MNE Preprocessing Tasks
https://mne.tools/stable/index.html

This benchmark evaluates LLMs on their ability to generate MNE preprocessing pipelines
from natural language prompts based on realistic EEG/MEG use cases.

Prompt examples and code references are curated from the official MNE documentation.
"""

import os
from transformers import AutoTokenizer
from eval.base import Task
from eval.utils import extract_generation_code, extract_code_pieces
from eval.tasks.custom_metrics.code_eval import compute_code_eval

_CITATION = """
@misc{gramfort2013meg,
  title={MEG and EEG data analysis with MNE-Python},
  author={Gramfort, Alexandre and Luessi, Martin and Larson, Eric and Engemann, Denis and Strohmeier, Daniel and Brodbeck, Christian and Goj, Roman and Jas, Mainak and Brooks, Thomas and Parkkonen, Lauri and Hämäläinen, Matti},
  journal={Frontiers in Neuroscience},
  volume={7},
  pages={267},
  year={2013},
  publisher={Frontiers Media SA},
  doi={10.3389/fnins.2013.00267}
}
"""


def create_all_tasks():
    return {
        "mne-codegen": create_task(strip_prompt=True),
        "mne-codegen-unstripped": create_task(strip_prompt=False)
    }


def create_task(strip_prompt):
    class MNECodegenTask(GeneralMNECodegen):
        def __init__(self, **kwargs):
            super().__init__(strip_prompt, **kwargs)

    return MNECodegenTask


class GeneralMNECodegen(Task):
    """Task for prompting an LLM to generate MNE preprocessing code."""

    def __init__(
        self, strip_prompt, k=[1, 5], num_workers=8, timeout=5.0, topk_docs: int = 3,
        dataset_path: str = None, dataset_name: str = None, data_files: dict = None,
        cache_dir: str = None, tokenizer: str = None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            data_files=data_files,
            cache_dir=cache_dir,
            stop_words=["\nif __name__", "\nprint(", "\ndef", "<file_sep>"],
            requires_execution=True,
        )
        self.strip_prompt = strip_prompt
        self.k = k
        self.num_workers = num_workers
        self.timeout = timeout
        self.topk_docs = topk_docs
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if tokenizer else None

    def get_dataset(self):
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds prompt for LLM, either with or without stripped whitespace."""
        prompt = 'Write an EEG preprocessing script'
        #prompt = doc["prompt"].strip() if self.strip_prompt else doc["prompt"]
        prompt = '"""' + prompt + '"""\n# Write the corresponding MNE code below:\n'

        context = doc
        if context:
            if isinstance(context, list):
                context = "\n".join([ctx["text"] if isinstance(ctx, dict) else ctx for ctx in context[:self.topk_docs]])
            prompt = context + "\n\n" + prompt

        if self.tokenizer and self.tokenizer.name_or_path.startswith("deepseek-ai"):
            prompt = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], 
                tokenize=False, add_generation_prompt=True
            )
        return prompt

    def get_reference(self, doc):
        if "code" not in doc:
            return None  
        return doc["code"]


    def postprocess_generation(self, generation, idx, new_tokens_only=False):
        return generation
        # if not new_tokens_only:
        #     full_prompt = self.get_prompt(self.dataset["test"][idx])
        #     generation = generation[len(full_prompt):]
        #     generation = self._stop_at_stop_token(generation, self.stop_words)
        #     generation = self.dataset["test"][idx]["prompt"] + generation
        #     generation = extract_generation_code(generation, self.dataset["test"][idx]["prompt"])
        # else:
        #     generation = extract_code_pieces(generation)
        # return generation

    def process_results(self, generations, references):
        """Compare generated MNE code to reference implementations using execution and output correctness."""
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
            k=self.k,
            num_workers=self.num_workers,
            timeout=self.timeout,
        )
        return results
