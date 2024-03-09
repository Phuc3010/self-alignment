export CUDA_VISIBLE_DEVICES="0, 4"
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file config/deepspeed_zero2.yaml --num_processes=2 --main_process_port 2950 src/run_kto.py config/config.yaml