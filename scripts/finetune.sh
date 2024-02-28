ACCELERATE_LOG_LEVEL=info

# Launches a distributed training job with the `accelerate` CLI tool. Key parameters include:
# --config_file: Path to the DeepSpeed configuration file. This file defines distributed training options and optimizations.
# --num_processes: Sets the number of processes to launch, typically equal to the number of GPUs available for parallel training.
# Additional override options (specified at command line) that can alter settings defined in config.yaml:
# --num_train_epochs=6: Specifies the total number of training epochs.
# --learning_rate=1e-7: Sets the learning rate for the training process.
# --beta=0.1: Custom beta parameter value.
# --warmup_ratio=0.1: Defines the warmup ratio for learning rate scheduling.
# --output_dir="${path_to_save_checkpoint}": Directory where training checkpoints will be saved.
# Execution command: Runs 'spin/run_spin.py' with 'configs/config.yaml' as its configuration.

accelerate launch --config_file config/deepspeed_zero3.yaml --num_processes=8 --main_process_port 2950 src/run_spin.py config/config_full.yaml