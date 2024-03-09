# Hyper-parameters tuning
## Evaluate Language Models
### Install
```shell
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```
### Evaluate LLMs on Open LLM leaderboard
After Install LM evaluation harness, you can evaluate LLMs across diverse tasks that used in Open LLM Leaderboard using this command:
```shell
accelerate launch -m lm_eval --model=hf --model_args pretrained={model_name_or_path},dtype=bfloat16 --tasks gsm8k --batch_size 1 --num_fewshot 5 --output_path ${results_path}-gsm8k --wandb_args project=huggingface, name=gsm8k --log_samples
accelerate launch -m lm_eval --model=hf --model_args pretrained={model_name_or_path},dtype=bfloat16 --tasks hellaswag --batch_size 1 --num_fewshot 10 --output_path ${results_path}-hellaswag --wandb_args project=huggingface, name=hellaswag --log_samples
accelerate launch -m lm_eval --model=hf --model_args pretrained={model_name_or_path},dtype=bfloat16 --tasks truthfulqa_mc2 --batch_size 1 --num_fewshot 0 --output_path ${results_path}-truthful --wandb_args project=huggingface, name=truthfulqa --log_samples
accelerate launch -m lm_eval --model=hf --model_args pretrained={model_name_or_path},dtype=bfloat16 --tasks mmlu --batch_size 1 --num_fewshot 5 --output_path ${results_path}-mmlu --wandb_args project=huggingface, name=mmlu --log_samples
accelerate launch -m lm_eval --model=hf --model_args pretrained={model_name_or_path},dtype=bfloat16 --tasks winogrande --batch_size 1 --num_fewshot 5 --output_path ${results_path}-winograde --wandb_args project=huggingface, name=winograde --log_samples
accelerate launch -m lm_eval --model=hf --model_args pretrained={model_name_or_path},dtype=bfloat16 --tasks arc_challenge --batch_size 1 --num_fewshot 25  --output_path ${results_path}-arc --wandb_args project=huggingface, name=arc --log_samples
```
- Where `{model_name_or_path}` is the model from huggingface or local, `{results_path}` the path to save evaluation results
## Runing Hyper-parameter scanning
### Step 1: Generating the data
Generate Synthetic dataset for each iteration using vLLM fast text generation inference.
```shell
python3 src/generate_vllm.py --model {model_path} --input_dir generated/synthetic -- output_dir generated/iter{num_iter} --world_size 8 
python3 src/generate_vllm.py --model {model_path} --input_dir generated/synthetic --frac_len 2000 --data_frac 0 --split test --output_dir generated/iter{num_iter} --world_size 8 
python3 src/convert_data.py --input_dir generated/iter0 --split test
python3 src/convert_data.py --input_dir generated/iter0 --split train
```
Where `{model_path}` is the `model_name_or_path` from local hugginface, `{num_iter}` is the iteration. 
Options for 
- '--model': load model checkpoint for generation from local or Huggingface
    - default: `alignment-handbook/zephyr-7b-sft-full`
- `--input_dir`: Directory to data files with prompts for generation.
    - default: `generated/synthetic`
- `--output_dir`: Directoty to save the output data.
    - default: `generated/iter0`
- `--data_frac`: break full data into fractions for generations (Generate a fraction of full data with `--frac_len` samples per fraction). This helps to generate data by small batches to avoid unexpected crashes as data generation can be very time-consuming.
- `--frac_len`: Length of the data fraction. If it is 0 which uses the entier dataset for generation. If not specified, it will generate 50_000 respones. According to the paper, this should be set at `50_000` at iter0 and `100_000` for subsequent iterations.
- `--split`: choose the split for data generation.
    - default: `train`
    
`src/convert_data.py` will convert the generated data from json format into `.parquet` format for fine-tuning.
### Step 2: Runing Fine-tuning scripts
The experiments can be launched with the following command: `bash scripts/pref_scan/finetune_scan.sh`
```shell
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

#!/bin/bash
priors=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
epochs=("1" "2")
SCRIPT_COMMAND="ACCELERATE_LOG_LEVEL=info accelerate launch --config_file config/deepspeed_zero3.yaml --num_processes=8 --main_process_port 2950 src/run_kto.py config/config.yaml --prior="

# Loop through different prior values
for epoch in "${epochs[@]}"; do
  for prior in "${priors[@]}"; do
    # Run the script with the current prior value
    OUTPUT_DIR="data/model/zephyr-7b-kto-iter0-${prior}-${epoch}"
    EVAL_PATH="data/output/zephyr-7b-kto-iter0-${prior}-${epoch}"
    
    echo "$SCRIPT_COMMAND$prior --num_train_epochs=${epoch} --output_dir=${OUTPUT_DIR} --hub_model_id=zephyr-7b-kto-iter0-${prior}-epoch${epoch}"

    eval "accelerate launch -m lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks gsm8k --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/gsm8k  --log_samples"
    eval "accelerate launch -m lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks hellaswag --batch_size 1 --num_fewshot 10 --output_path ${EVAL_PATH}/hellaswag --log_samples"
    eval "accelerate launch -m lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks truthfulqa_mc2 --batch_size 1 --num_fewshot 0 --output_path ${EVAL_PATH}/truthful --log_samples"
    eval "accelerate launch -m lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks mmlu --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/mmlu --log_samples"
    eval "accelerate launch -m lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks winogrande --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/winograde --log_samples"
    eval "accelerate launch -m lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks arc_challenge --batch_size 1 --num_fewshot 25  --output_path ${EVAL_PATH}/arc --log_samples"
  done
done
```