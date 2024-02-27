from datasets import load_dataset, Dataset
import torch
import os
import json
from pathlib import Path
from tqdm import tqdm 
import pyarrow.parquet as pq
import logging
import argparse
import random
random.seed(42)
from reformat import save_to_parquet

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='generated/iter0')
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    split = args.split
    data_path = os.path.join(args.input_dir, split+".jsonl")
    with open(data_path, "r") as json_file:
        json_lst = list(json_file)

    json_lst = [json.loads(ele) for ele in json_lst]
    prompt = [ele['real'][0]['content'] for ele in json_lst]
    print(len(set(prompt)))
    data = Dataset.from_list(json_lst)
    print(data)
    save_to_parquet(data, Path(args.input_dir)/ f'{split}_prefs-00000-of-00001.parquet')
    os.remove(data_path)
