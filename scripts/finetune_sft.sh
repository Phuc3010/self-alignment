export CUDA_VISIBLE_DEVICES="0,1"
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file config/fsdp.yaml --num_processes=2 --main_process_port 2950 src/run_sft.py config/config_sft.yaml