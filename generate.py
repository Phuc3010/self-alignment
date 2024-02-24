from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta

import warnings

from accelerate.utils import InitProcessGroupKwargs

import warnings
warnings.filterwarnings("ignore")
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument('--world_size', type=int, default=8) # controls the number of gpus vLLM is allowed to use
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument('--input_dir', type=str, default='UCLA-AGI/SPIN_iter0')
    parser.add_argument('--split', type=str, default='train')
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_path = args.model
    data_frac = args.data_frac
    world_size = args.world_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_path,
        tensor_parallel_size=world_size,
    )
    print(f"Number of generated responses {args.n}")

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256, n=args.n)

    # load data
    with open(args.input_dir+f"/{args.split}.jsonl", "r") as json_file:
        json_lst = list(json_file)
    import json
    data = [json.loads(ele) for ele in json_lst]

    random.seed(42)
    random.shuffle(data)
    if args.frac_len > 0:
        sub_len = args.frac_len 
        if sub_len*(data_frac+1) > len(data):
            data = data[sub_len*data_frac:]
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]

    prompts_all = ["### Instruction: " + data[idx]['prompt'] + "\n\n### Response: " for idx in range(len(data))]
    prompts_old = [data[idx]['prompt'] for idx in range(len(data))]
    corrects_all = [data[idx]['generations'][0] for idx in range(len(data))]

    start=time.time()

    #run vllm
    results_gathered = list(map(lambda x: [x.outputs[idx].text for idx in range(len(x.outputs))], 
                                llm.generate(prompts_all, sampling_params)))

    print(results_gathered[0])
    print(len(results_gathered[0]))
    results = [[ele.replace("</s>","").lstrip() for ele in r]for r in results_gathered]
    

    timediff=time.time()-start
    print(f"time elapsed: {timediff}")

    # collecting data
    for idx in range(len(corrects_all)):
        sample = {"prompt": prompts_old[idx], "generations": [corrects_all[idx]]+results[idx], "desirable": [True]+[False]*len(results[idx])}
        if args.split == 'test':
            filename = f"{args.output_dir}/test.jsonl"
        else:
            filename = f"{args.output_dir}/train.jsonl"
        with open(filename, 'a') as f:
            json.dump(sample, f)
            f.write('\n')


if __name__ == "__main__":
    main()