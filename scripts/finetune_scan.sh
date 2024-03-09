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
    
    eval  "$SCRIPT_COMMAND$prior --num_train_epochs=${epoch} --output_dir=${OUTPUT_DIR} --hub_model_id=zephyr-7b-kto-iter0-${prior}-epoch${epoch}"

    # eval "lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks gsm8k --device cuda --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}-gsm8k --wandb_args project=huggingface, name=gsm8k --log_samples"
    # eval "lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks hellaswag --device cuda --batch_size 1 --num_fewshot 10 --output_path ${EVAL_PATH}-hellaswag --wandb_args project=huggingface, name=hellaswag --log_samples"
    # eval "lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks truthfulqa_mc2 --device cuda --batch_size 1 --num_fewshot 0 --output_path ${EVAL_PATH}-truthful --wandb_args project=huggingface, name=truthfulqa --log_samples"
    # eval "lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks mmlu --device cuda --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}-mmlu --wandb_args project=huggingface, name=mmlu --log_samples"
    # eval "lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks winogrande --device cuda --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}-winograde --wandb_args project=huggingface, name=winograde --log_samples"
    # eval "lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks arc_challenge --device cuda --batch_size 1 --num_fewshot 25  --output_path ${EVAL_PATH}-arc --wandb_args project=huggingface, name=arc --log_samples"
  done
done