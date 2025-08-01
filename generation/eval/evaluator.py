import inspect
import json
import os
import warnings

from typing import List


from eval import tasks
from eval.generation import (
    parallel_generations, openai_generations, 
    gemini_generations, litellm_generations
)

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""

class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name, intermediate_generations=None):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        # if args.limit is used, make sure args.limit_start + args.limit <= len(dataset)
        n_tasks = min(self.args.limit, len(dataset) - self.args.limit_start) if self.args.limit else len(dataset)
        # when args.limit is None
        # adjust n_tasks by args.limit_start to prevent out of bounds issues 
        if not self.args.limit:
            n_tasks -= self.args.limit_start
        references = [task.get_reference(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references

        curr_generations = []  # list[list[str | None] | None]
        if intermediate_generations:
            curr_generations = [gen for gen in intermediate_generations if gen]
            n_tasks -= len(curr_generations)
        intermediate_save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}_intermediate.json"
        curr_sample_idx = len(curr_generations)

        generations = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
            curr_sample_idx=curr_sample_idx,  # curr_sample_idx will added to limit_start to fix indexing
            save_every_k_tasks=self.args.save_every_k_tasks,
            intermediate_generations=curr_generations,
            intermediate_save_generations_path=intermediate_save_generations_path,
        )

        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references

    def evaluate(self, task_name, intermediate_generations=None, user_prompt=None):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations, references = self.generate_text(task_name, intermediate_generations=intermediate_generations, user_prompt=user_prompt)

        if not self.accelerator or self.accelerator.is_main_process:
            if not self.args.load_generations_path:
                save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}.json"
                self.save_json_files(generations, references, save_generations_path, f"references_{task_name}.json")

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")
            results = task.process_results(generations, references)
            return results

    def save_json_files(
        self,
        generations: List[str],
        references: List[str],
        save_generations_path: str,
        save_references_path: str,
    ) -> None:
        if self.args.save_generations:
            with open(save_generations_path, "w") as fp:
                json.dump(generations, fp)
                print(f"generations were saved at {save_generations_path}")
        if self.args.save_references:
            with open(save_references_path, "w") as fp:
                json.dump(references, fp)
                print(f"references were saved at {save_references_path}")


class vllmEvaluator(Evaluator):
    def __init__(self, accelerator, model, tokenizer, args):
        super().__init__(accelerator, model, tokenizer, args)

    def generate_text(self, task_name, intermediate_generations=None, user_prompt=None):
        # TBD: handle args.limit
        from vllm import LLM, SamplingParams
        
        task = tasks.get_task(task_name, self.args)
        
        dataset = task.get_dataset()
        references = [task.get_reference(x) for x in dataset]
        
        if self.args.load_generations_path:
            # load generated code
            with open(self.args.load_generations_path) as fp:
                generations = json.load(fp)
                print(
                    f"generations loaded, from {len(generations)} examples with {len(generations[0])} candidates"
                )
            return generations, references
        
        print("Preprocessing input data..")
        pre_kwargs = {
            'tokenizer': self.tokenizer,
            'remove_linebreak': self.args.remove_linebreak or "starcoder" in self.args.model,
            'add_linebreak': 'Llama-3' in self.args.model,
            'max_length_input': self.args.max_length_input,
        }
        task.preprocess_all_data(**pre_kwargs)
        dataset = task.get_dataset()
        prompts = [task.get_prompt(ex, user_prompt=user_prompt) for ex in dataset]
        
        print("Generating..")
        sampling_params = SamplingParams(
            temperature=self.args.temperature, 
            top_p=self.args.top_p, 
            max_tokens=self.args.max_length_generation - self.args.max_length_input,
            ignore_eos=self.args.ignore_eos,
            n=self.args.n_samples,
        )
        outputs = self.model.generate(prompts, sampling_params)
        
        generations = []
        for idx,output in enumerate(outputs):
            generated_text_list = []
            for k in range(sampling_params.n):
                gen_text = output.outputs[k].text
                if not self.args.new_tokens_only:
                    gen_text = prompts[idx] + gen_text
                if self.args.postprocess:
                    gen_text = task.postprocess_generation(gen_text, idx, new_tokens_only=self.args.new_tokens_only)
                generated_text_list.append(gen_text)
            generations.append(generated_text_list)
        
        return generations, references # both are in the format of [[sample1, sample2, ..., samplek], [], []]


class ApiEvaluator(Evaluator):
    def __init__(self, model: str, args):
        super().__init__(None, model, None, args)
    
    def generate_text(self, task_name, intermediate_generations=None, user_prompt=None):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        # if args.limit is used, make sure args.limit_start + args.limit <= len(dataset)
        n_tasks = min(self.args.limit, len(dataset) - self.args.limit_start) if self.args.limit else len(dataset)
        # when args.limit is None
        # adjust n_tasks by args.limit_start to prevent out of bounds issues 
        if not self.args.limit:
            n_tasks -= self.args.limit_start
        references = [task.get_reference(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references
        
        curr_generations = []  # list[list[str | None] | None]
        if intermediate_generations:
            curr_generations = [gen for gen in intermediate_generations if gen]
            n_tasks -= len(curr_generations)
        intermediate_save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}_intermediate.json"
        curr_sample_idx = len(curr_generations)
        
        print("Preprocessing input data..")
        from transformers import GPT2TokenizerFast
        
        pre_kwargs = {
            'tokenizer': GPT2TokenizerFast.from_pretrained('RaymondLi/gpt-4-tokenizer'),
            'remove_linebreak': self.args.remove_linebreak,
            'add_linebreak': False,
            'max_length_input': self.args.max_length_input,
        }
        task.preprocess_all_data(**pre_kwargs)
        dataset = task.get_dataset()
        
        print("Generating..")
        if "gemini" in self.model:
            generations = gemini_generations(
                task,
                dataset,
                self.model,
                n_tasks=n_tasks,
                args=self.args,
                curr_sample_idx=curr_sample_idx,  # curr_sample_idx will added to limit_start to fix indexing
                save_every_k_tasks=self.args.save_every_k_tasks,
                intermediate_generations=curr_generations,
                intermediate_save_generations_path=intermediate_save_generations_path,
            )
        # elif "gpt" in self.model:
        #     generations = openai_generations(
        #         task,
        #         dataset,
        #         self.model,
        #         n_tasks=n_tasks,
        #         args=self.args,
        #         curr_sample_idx=curr_sample_idx,  # curr_sample_idx will added to limit_start to fix indexing
        #         save_every_k_tasks=self.args.save_every_k_tasks,
        #         intermediate_generations=curr_generations,
        #         intermediate_save_generations_path=intermediate_save_generations_path,
        #     )
        else:
            generations = litellm_generations(
                task,
                dataset,
                self.model,
                n_tasks=n_tasks,
                args=self.args,
                curr_sample_idx=curr_sample_idx,  # curr_sample_idx will added to limit_start to fix indexing
                save_every_k_tasks=self.args.save_every_k_tasks,
                intermediate_generations=curr_generations,
                intermediate_save_generations_path=intermediate_save_generations_path,
                user_prompt=user_prompt
            )

        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references