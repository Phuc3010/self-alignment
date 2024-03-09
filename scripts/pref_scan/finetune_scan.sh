export CUDA_VISIBLE_DEVICES="4,5"

#!/bin/bash
priors=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7")
epochs=("1" "2")
SCRIPT_COMMAND="ACCELERATE_LOG_LEVEL=info accelerate launch --config_file config/deepspeed_zero2.yaml --num_processes=2 --main_process_port 2950 src/run_kto.py config/config.yaml --prior="

# Loop through different prior values
for epoch in "${epochs[@]}"; do
  for prior in "${priors[@]}"; do
    # Run the script with the current prior value
    OUTPUT_DIR="data/model/tiny-llama-kto-iter0-${prior}-${epoch}"
    EVAL_PATH="data/output/tiny-llama-kto-iter0-${prior}-${epoch}"
    
    eval "$SCRIPT_COMMAND$prior --num_train_epochs=${epoch} --output_dir=${OUTPUT_DIR} --hub_model_id=tiny-llama-kto-iter0-${prior}-epoch${epoch}"

    eval "accelerate launch -m lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks gsm8k --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/gsm8k  --log_samples"
    eval "accelerate launch -m lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks hellaswag --batch_size 1 --num_fewshot 10 --output_path ${EVAL_PATH}/hellaswag --log_samples"
    eval "accelerate launch -m lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks truthfulqa_mc2 --batch_size 1 --num_fewshot 0 --output_path ${EVAL_PATH}/truthful --log_samples"
    eval "accelerate launch -m lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks mmlu --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/mmlu --log_samples"
    eval "accelerate launch -m lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks winogrande --batch_size 1 --num_fewshot 5 --output_path ${EVAL_PATH}/winograde --log_samples"
    eval "accelerate launch -m lm_eval --model=hf --model_args pretrained=${OUTPUT_DIR},dtype=bfloat16 --tasks arc_challenge --batch_size 1 --num_fewshot 25  --output_path ${EVAL_PATH}/arc --log_samples"
  done
done
