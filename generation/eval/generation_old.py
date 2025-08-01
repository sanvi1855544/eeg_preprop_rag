import json
import time
import openai
import tiktoken
from tqdm import tqdm
from math import ceil
from typing import List, Optional

from accelerate.utils import set_seed
from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteria, StoppingCriteriaList

#from eval.utils import TokenizedDataset, complete_code
from .utils import TokenizedDataset, complete_code

from openai import OpenAI
client = OpenAI()

class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""
    def __init__(self, start_length, eof_strings, tokenizer, check_fn=None):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                [stop_string in decoded_generation for stop_string in self.eof_strings]
            )
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        return all([self.check_fn(decoded_generation) for decoded_generation in decoded_generations])

class TooLongFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if the generated function is too long by a certain multiplier based on input length."""

    def __init__(self, input_length, multiplier):
        self.input_length = input_length
        self.multiplier = multiplier

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if generated sequence is too long."""
        return input_ids.shape[1] > int(self.input_length * self.multiplier)
        

def parallel_generations(
        task,
        dataset,
        accelerator,
        model,
        tokenizer,
        n_tasks,
        args,
        curr_sample_idx: int = 0,
        save_every_k_tasks: int = -1,
        intermediate_generations: Optional[List[Optional[List[Optional[str]]]]] = None,
        intermediate_save_generations_path: Optional[str] = None,
):
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            if accelerator.is_main_process:
                print(
                    f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
                )
        return generations[:n_tasks]

    set_seed(args.seed, device_specific=True)

    # Setup generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_length": args.max_length_generation,
    }
    stopping_criteria = []
    # The input_length / start_length set to 0 for now will be adjusted later
    # Check if the task has a custom check_fn method for the stopping criteria
    if task.stop_words and tokenizer.eos_token:
        task.stop_words.append(tokenizer.eos_token)    
    if hasattr(task, "check_fn"):
        stopping_criteria.append(
            EndOfFunctionCriteria(0, task.stop_words, tokenizer, task.check_fn)
        )
    elif task.stop_words:
        stopping_criteria.append(
            EndOfFunctionCriteria(0, task.stop_words, tokenizer)
        )
    if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
        stopping_criteria.append(
            TooLongFunctionCriteria(0, task.max_length_multiplier)
        )
    
    if stopping_criteria:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(stopping_criteria)

    if args.instruction_tokens:
        instruction_tokens = args.instruction_tokens.split(",")
        if len(instruction_tokens) != 3:
            raise ValueError(
                "Instruction tokens should contain exactly 3 tokens separated by a comma. If a token is empty, represent it as ''"
            )
        for token in instruction_tokens:
            if token.strip() != "":
                task.stop_words.append(token)
    else:
        instruction_tokens = None
    if accelerator.is_main_process:
        print(f"number of problems for this task is {n_tasks}")
    n_copies = ceil(args.n_samples / args.batch_size)

    ds_tokenized = TokenizedDataset(
        task,
        dataset,
        tokenizer,
        num_devices=accelerator.state.num_processes,
        max_length=args.max_length_input,
        limit_start=args.limit_start + curr_sample_idx,
        n_tasks=n_tasks,
        n_copies=n_copies,
        prefix=args.prefix,
        has_encoder=args.modeltype == "seq2seq",
        instruction_tokens=instruction_tokens,
    )

    # do not confuse args.batch_size, which is actually the num_return_sequences
    ds_loader = DataLoader(ds_tokenized, batch_size=1)

    is_loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    is_loaded_in_4bit = getattr(model, "is_loaded_in_4bit", False)
    if args.max_memory_per_gpu is not None:
        # The model is already sharded across multiple GPUs
        ds_loader = accelerator.prepare(ds_loader)
    elif not is_loaded_in_8bit and not is_loaded_in_4bit:
        # we only wrap data loader to avoid extra memory occupation
        model = model.to(accelerator.device)
        ds_loader = accelerator.prepare(ds_loader)
    else:
        # model.to() is not supported for 8bit and 4bit models
        model, ds_loader = accelerator.prepare(model, ds_loader)

    generations = complete_code(
        task,
        accelerator,
        model,
        tokenizer,
        ds_loader,
        n_tasks=n_tasks,
        limit_start=args.limit_start + curr_sample_idx,
        batch_size=args.batch_size,
        prefix=args.prefix,
        instruction_tokens=instruction_tokens,
        postprocess=args.postprocess,
        is_wrapped=is_loaded_in_8bit or is_loaded_in_4bit,
        save_every_k_tasks=save_every_k_tasks,
        intermediate_generations=intermediate_generations,
        intermediate_save_generations_path=intermediate_save_generations_path,
        **gen_kwargs,
    )
    return generations



def parse_code_snippets(text: str) -> list[str]:
    """Extract code pieces from a text string.
    Args:
        text: str, model prediciton text.
    Rets:
        code_pieces: list[str], code pieces in the text.
    """
    code_pieces = []
    while "```python" in text:
        st_idx = text.index("```python") + 10
        # end_idx = text.index("```", st_idx)
        if "```" in text[st_idx:]:
            end_idx = text.index("```", st_idx)
        else: 
            end_idx = len(text)
        code_pieces.append(text[st_idx:end_idx].strip())
        text = text[end_idx+3:].strip()
    return '\n\n'.join(code_pieces)



# %% OpenAI Generations
from openai import OpenAI, AzureOpenAI
# fill in specification here
gpt_tokenizer = tiktoken.get_encoding("cl100k_base")

def openai_generations(
    task,
    dataset,
    model,
    n_tasks,
    args,
    curr_sample_idx: int = 0,
    save_every_k_tasks: int = -1,
    intermediate_generations: Optional[List[Optional[List[Optional[str]]]]] = None,
    intermediate_save_generations_path: Optional[str] = None,
):
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            print(
                f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
            )
            # if accelerator.is_main_process:
        return generations[:n_tasks]
    
    def get_response(prompt: str, n_iters: int = 2, sleep: int = 10, repoeval_prompt=False, **kwargs) -> list[str]:
        prompt_tokens = gpt_tokenizer.encode(prompt)
        prompt = gpt_tokenizer.decode(prompt_tokens[: args.max_length_input])
        
        # response = client.chat.completions.create(
        #     model=model, 
        #     messages=[{"role": "user", "content": prompt}],
        #     **kwargs
        # )
        # response = completion(
        #     model=model, 
        #     messages=[{"role": "user", "content": prompt}],
        #     **kwargs
        # )
        # return [c.message.content for c in response.choices]
        i_iters = 0
        response = ""
        while i_iters < n_iters:
            i_iters += 1
            try:
                if repoeval_prompt:
                    messages = [
                        {"role": "system", "content": "Instruction: Continue writing the code."},
                        {"role": "system", "name": "example_user", "content": "Continue writing the following code:\n\n```\ndef return_none():\n```"},
                        {"role": "system", "name": "example_assistant", "content": "```\n    return None\n```"},
                        {"role": "user", "content": "Continue writing the following code:\n\n```\n" + prompt + '\n```'},
                    ]
                else:
                    messages=[{"role": "user", "content": prompt}]
                    
                response = client.chat.completions.create(
                    model=model, 
                    messages = messages,
                    **kwargs
                )
                return [c.message.content for c in response.choices]
            except:
                time.sleep(i_iters * sleep)
        return [response]

    # Setup generation settings
    gen_kwargs = {
        "max_tokens": args.max_length_generation,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    
    save_every_k_tasks = args.save_every_k_tasks if args.save_every_k_tasks > 0 else n_tasks+1
    intermediate_generation_file = args.save_generations_path + '.partial'
    generations = []
    for i in tqdm(range(args.limit_start + curr_sample_idx, n_tasks)):
        i_prompt = task.get_prompt(doc=dataset[i])
        i_resp = get_response(prompt=i_prompt, repoeval_prompt=task.__class__.__name__=='RepoEval', **gen_kwargs) # list[str]
        generations.append(i_resp)
        if len(generations) % save_every_k_tasks == 0:
            with open(intermediate_generation_file, 'w') as fp:
                json.dump(generations, fp)
    
    processed_generations = []
    for i, gs in enumerate(generations):
        processed_gs = [
            task.postprocess_generation(generation=g,idx=i,new_tokens_only=True) 
            for g in gs
        ]
        processed_generations.append(processed_gs)

    return intermediate_generations + processed_generations




# %% LiteLLM Generations
# import litellm
# litellm.set_verbose=True
from litellm import completion

def litellm_generations(
    task,
    dataset,
    model,
    n_tasks,
    args,
    curr_sample_idx: int = 0,
    save_every_k_tasks: int = -1,
    intermediate_generations: Optional[List[Optional[List[Optional[str]]]]] = None,
    intermediate_save_generations_path: Optional[str] = None,
):
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            print(
                f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
            )
            # if accelerator.is_main_process:
        return generations[:n_tasks]
    
    def get_response(prompt: str, n_iters: int = 2, sleep: int = 30, **kwargs) -> list[str]:
        prompt_tokens = gpt_tokenizer.encode(prompt)
        prompt = gpt_tokenizer.decode(prompt_tokens[: args.max_length_input])

        # for debugging
        # response = completion(
        #     model=model, 
        #     messages=[{"role": "user", "content": prompt}],
        #     **kwargs
        # )
        # return [c.message.content for c in response.choices]
        i_iters = 0
        response = ""
        while i_iters < n_iters:
            i_iters += 1
            try:
                response = completion(
                    model=model, 
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                return [c.message.content for c in response.choices]
            except:
                time.sleep(i_iters * sleep)
        return [response]

    # Setup generation settings
    gen_kwargs = {
        "max_tokens": args.max_length_generation,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    generations = []
    for i in tqdm(range(args.limit_start + curr_sample_idx, n_tasks)):
        i_prompt = task.get_prompt(doc=dataset[i])
        i_resp = get_response(prompt=i_prompt, **gen_kwargs) # list[str]
        generations.append(i_resp)
    
    processed_generations = []
    for i, gs in enumerate(generations):
        processed_gs = [
            task.postprocess_generation(generation=g,idx=i,new_tokens_only=True) 
            for g in gs
        ]
        processed_generations.append(processed_gs)

    return intermediate_generations + processed_generations


# %% Gemini Generations
import os, json
import google.generativeai as genai
genai.configure(api_key=os.environ["API_KEY"])

def gemini_generations(
    task,
    dataset,
    model,
    n_tasks,
    args,
    curr_sample_idx: int = 0,
    save_every_k_tasks: int = -1,
    intermediate_generations: Optional[List[Optional[List[Optional[str]]]]] = None,
    intermediate_save_generations_path: Optional[str] = None,
):
    if args.load_generations_path:
        # load generated code
        with open(args.load_generations_path) as fp:
            generations = json.load(fp)
            print(
                f"generations loaded, {n_tasks} selected from {len(generations)} with {len(generations[0])} candidates"
            )
            # if accelerator.is_main_process:
        return generations[:n_tasks]
    
    model = genai.GenerativeModel(model) # "gemini-1.5-flash"
    gen_kwargs = {
        "max_output_tokens": args.max_length_generation,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    def get_response(prompt: str, n_iters: int = 5, sleep: int = 30) -> list[str]:
        prompt_tokens = gpt_tokenizer.encode(prompt)
        prompt = gpt_tokenizer.decode(prompt_tokens[: args.max_length_input])
        # response = model.generate_content(prompt, generation_config=gen_kwargs)
        # return [c.content.parts[0].text for c in response.candidates]
        i_iters = 0
        response = ""
        while i_iters < n_iters:
            i_iters += 1
            try:
                response = model.generate_content(prompt, generation_config=gen_kwargs)
                return [c.content.parts[0].text for c in response.candidates]
            except:
                time.sleep(i_iters * sleep)
        return [response]

    generations = []
    for i in tqdm(range(args.limit_start + curr_sample_idx, n_tasks)):
        i_prompt = task.get_prompt(doc=dataset[i])
        i_resp = get_response(prompt=i_prompt) # list[str]
        if not (isinstance(i_resp, list) and isinstance(i_resp[0], str)):
            i_resp = [""]
        assert json.dumps(i_resp)
        generations.append(i_resp)
    
    processed_generations = []
    for i, gs in enumerate(generations):
        processed_gs = [
            task.postprocess_generation(generation=g,idx=i,new_tokens_only=True) 
            for g in gs
        ]
        processed_generations.append(processed_gs)

    return intermediate_generations + processed_generations