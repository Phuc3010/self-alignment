import json
import numpy as np
import torch
from transformers import AutoTokenizer
from collections import defaultdict
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from datetime import datetime
import os
import argparse
from tqdm import tqdm
import glob
from typing import List

def tokenize_pair(sources:List[str], candidate1s:List[str], candidate2s:List[str], source_max_length=1124, candidate_max_length=512):
    ids = []
    assert len(sources) == len(candidate1s) == len(candidate2s)
    max_length = source_max_length + 2 * candidate_max_length
    for i in range(len(sources)):
        source_ids = tokenizer.encode(source_prefix + sources[i], max_length=source_max_length, truncation=True)
        candidate_max_length = (max_length - len(source_ids)) // 2
        candidate1_ids = tokenizer.encode(cand1_prefix + candidate1s[i], max_length=candidate_max_length, truncation=True)
        candidate2_ids = tokenizer.encode(cand2_prefix + candidate2s[i], max_length=candidate_max_length, truncation=True)
        ids.append(source_ids + candidate1_ids + candidate2_ids)
    encodings = tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding="max_length", max_length=max_length)
    return encodings

def load_model_answers(answer_dir: str, task: str, model_list: List=None):
    filenames = glob.glob(os.path.join(answer_dir, "*.json"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        task_file, model_name = os.path.basename(filename).split(".json")[0].split("_")
        answer = {}
        if task_file == task:
            if model_list is not None and model_name not in model_list: 
                continue
            data = json.load(open(filename))
            answer["inputs"] = [data[idx]['instruction'] for idx in range(len(data))]
            answer["candidate_texts"] = [[data[idx][model_name], data[idx]["reference"]] for idx in range(len(data))]
            model_answers[model_name] = answer
    
    return model_answers


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", type=str, default="llm-blender/PairRM-hf")
    parser.add_argument("--task", type=str, default="alpaca")
    parser.add_argument("--result_dir", type=str, default="eval/results")
    parser.add_argument(
        "--model_list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    args = parser.parse_args()
    source_prefix = "<|source|>"
    cand1_prefix = "<|candidate1|>"
    cand2_prefix = "<|candidate2|>"
    model_answers = load_model_answers(args.result_dir, args.task, args.model_list)
    pairrm = DebertaV2PairRM.from_pretrained(args.judge_model, device_map="cuda:0", torch_dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)

    wins = defaultdict(lambda: 0)
    score_diff = defaultdict(list)
    with torch.no_grad():
        for model in list(model_answers.keys()):
            inputs = model_answers[model]['inputs']
            candidate_texts = model_answers[model]['candidate_texts']
            batch_size = 32
            for i in tqdm(range(0, len(inputs), batch_size), desc=f"Evaluate {model}"):
                batch_inputs = inputs[i:i+batch_size]
                batch_candidate_texts = candidate_texts[i:i+batch_size]
                candidates_A = [ele[0] for ele in batch_candidate_texts]
                candidates_B = [ele[1] for ele in batch_candidate_texts]
                encodings = tokenize_pair(batch_inputs, candidates_A, candidates_B)
                encodings = {k:v.to(pairrm.device) for k,v in encodings.items()}
                outputs = pairrm(**encodings)
                score = outputs.logits.tolist()
                score_diff[model].extend(score)
                comparison_results = (outputs.logits > 0).tolist()
                for ele in comparison_results:
                    if ele == True:
                        wins[model] += 1
                    else:
                        wins["reference"] += 1

            results = {
                'date': str(datetime.now()),
                'total': len(inputs),
                'task': args.task,
                'judge' : args.judge_model,
                'candidate': {
                    'name': model,
                    'wins': wins[model],
                    'score_diff': np.mean(score_diff[model])
                },
                'baseline': {
                    'name': "reference",
                    'wins': wins["reference"],
                    "score_diff": -np.mean(score_diff[model])
                },
            }
            wins = defaultdict(lambda: 0)
            score_diff = defaultdict(list)

            with open(os.path.join(args.result_dir, "results.jsonl"), 'a+') as f:
                json.dump(results, f)
                f.write('\n')