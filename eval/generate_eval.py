
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
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,  required=True)
    parser.add_argument('--model', type=str, default='DatPySci/pythia-1b-kto-iter0')
    parser.add_argument("--ref_model", type=str, default='DatPySci/pythia-1b-spin-iter0')
    parser.add_argument('--output_dir', type=str, default='eval/results')
    parser.add_argument('--world_size', type=int, default=1) # controls the number of gpus vLLM is allowed to use
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
    )

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256)

    # load data
    data = load_dataset("tatsu-lab/alpaca_eval", split="eval")
    prompts_all = ["### Instruction: " + data[idx]['instruction'] + "\n\n### Response: " for idx in range(len(data))]
    prompts_old = [data[idx]['instruction'] for idx in range(len(data))]
    corrects_all = [data[idx]['output'] for idx in range(len(data))]

    start=time.time()

    #run vllm
    results_gathered = list(map(lambda x: x.outputs[0].text, 
                                llm.generate(prompts_all, sampling_params)))

    results = [r.replace("</s>","").lstrip() for r in results_gathered]
    print(results[0])
    timediff=time.time()-start
    print(f"time elapsed: {timediff}")

    alpaca_formatted_examples = []
    if args.ref_model is not None:
        destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        ref_llm = LLM(
            model=args.ref_model,
            tensor_parallel_size=world_size,
        )

        results_gathered = list(map(lambda x: x.outputs[0].text, 
                                ref_llm.generate(prompts_all, sampling_params)))
        results_ref = [r.replace("</s>","").lstrip() for r in results_gathered]
        for idx in range(len(corrects_all)):
            alpaca_formatted_examples.append({
                'instruction': prompts_old[idx],
                "kto-nnpu": results[idx].strip(),
                "spin": results_ref[idx].strip()
            })

    # collecting data
    else:
        for idx in range(len(corrects_all)):
            alpaca_formatted_examples.append({
                'instruction': prompts_old[idx],
                "kto-nnpu": results[idx].strip(),
                "spin": corrects_all[idx].strip()
            })
    fn = os.path.join(str(output_dir), f'alpaca_{args.exp_name}.json')
    json.dump(alpaca_formatted_examples, open(fn, 'w'), indent=2)
    
if __name__ == "__main__":
    main()