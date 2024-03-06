
from vllm import LLM, SamplingParams
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import warnings

from accelerate.utils import InitProcessGroupKwargs

import warnings
warnings.filterwarnings("ignore")


INSTRUCTION_PROMPT = {
    "alpaca": "### Instruction: {prompt}\n\n### Response: ",
    "summarization": "{prompt}\n\nTL;DR: ",
    "hhh": "### Instruction: {prompt}\n\n### Response: "
}
def get_data_from_task(task_name: str, subset: str=None):
    if task_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca_eval", split="eval")
        dataset = dataset.rename_column("instruction", "prompt")
        return dataset
    elif task_name == "summarization":
        dataset = load_dataset("openai/summarize_from_feedback", "comparisons", split="validation")
        def reformat_summarize(example):
            example['prompt'] = f"SUBREDDIT: {example['info']['subreddit']}\n\nTITLE: {example['info']['title']}\n\nPOST: {example['info']['post']}"
            example['output'] = example['summaries'][example['choice']]['text'].strip()
            return example

        dataset = dataset.map(reformat_summarize, batched=False, num_proc=4)
        return dataset
    elif task_name == "hhh":
        dataset = load_dataset("HuggingFaceH4/hhh_alignment", subset, split="test")
        dataset = dataset.rename_column("input", "prompt")
        def get_output(example):
            labels = example['targets']['labels']
            if labels[0] == 1:
                example['output'] = example['targets']['choices'][0]
            else:
                example['output'] = example['targets']['choices'][1]
            return example

        dataset = dataset.map(get_output, batched=False)
        return dataset
    else:
        raise ValueError


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="alpaca")
    parser.add_argument('--model', type=str, default='UCLA-AGI/zephyr-7b-sft-full-SPIN-iter3')
    parser.add_argument("--ref_model", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='eval/results')
    parser.add_argument('--candidate_key', "-c", type=str, default='output')
    parser.add_argument('--baseline_key', "-b", type=str, default='reference')
    parser.add_argument('--world_size', "-w", type=int, default=8) # controls the number of gpus vLLM is allowed to use
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_path = args.model
    world_size = args.world_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_path,
        tensor_parallel_size=world_size,
        max_model_len=2048
    )

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

    # load data
    subset = args.subset
    data = get_data_from_task(args.task, subset)
    prompts_all = [INSTRUCTION_PROMPT[args.task].format(prompt=data[idx]['prompt']) for idx in range(len(data))]
    prompts_old = [data[idx]['prompt'] for idx in range(len(data))]
    corrects_all = [data[idx]['output'] for idx in range(len(data))]
    
    start=time.time()

    #run vllm
    results_gathered = list(map(lambda x: x.outputs[0].text, 
                                llm.generate(prompts_all, sampling_params)))

    results = [r.replace("</s>","").lstrip() for r in results_gathered]
    print(results[0])
    timediff=time.time()-start
    print(f"time elapsed: {timediff}")

    formatted_examples = []
    if args.ref_model is not None:
        destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        ref_llm = LLM(
            model=args.ref_model,
            tensor_parallel_size=world_size,
            max_model_len=2048
        )

        results_gathered = list(map(lambda x: x.outputs[0].text, 
                                ref_llm.generate(prompts_all, sampling_params)))
        results_ref = [r.replace("</s>","").lstrip() for r in results_gathered]
        for idx in range(len(corrects_all)):
            formatted_examples.append({
                'instruction': prompts_old[idx],
                args.candidate_key: results[idx].strip(),
                args.baseline_key: results_ref[idx].strip()
            })

    # collecting data
    else:
        for idx in range(len(corrects_all)):
            formatted_examples.append({
                'instruction': prompts_old[idx],
                args.candidate_key: results[idx].strip(),
                args.baseline_key: corrects_all[idx].strip()
            })
    
    fn = os.path.join(str(output_dir), f'{args.task}_{args.candidate_key}.json')
    if subset is not None:
        fn = os.path.join(str(output_dir), f'{args.task}-{args.subset}_{args.candidate_key}.json')

    json.dump(formatted_examples, open(fn, 'w'), indent=2)
    
if __name__ == "__main__":
    main()