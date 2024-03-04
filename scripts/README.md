# Scripts to Train and Evaluate Chat Models
## Step 1: Reformat data
This step reformat the UltraChat 200k dataset from multi-turn conversation into single-turn as the prompt and ground truh responses. The output formatted will be save in `--output_dir`. This step only need to one time at iter 0.
```shell
python3 src/reformat.py [options]
```
Options
- `--data`: directory to the SFT dataset (local or huggingface)
    - default: `HuggingFaceH4/ultrachat_200k`
- `--output_dir`: local directory to the reformated data files 
    - default: `generated/synthetic`
## Step 2: Generation
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
- `--frac_len`: Length of the data fraction. If it is 0 which uses the entier dataset for generation.
- `--split`: choose the split for data generation.
    - default: `train`
`src/convert_data.py` will convert the generated data from json format into `.parquet` format for fine-tuning.
## Fine-tuning
```shell
# Full training with Zero-3 on 8 GPUs
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file config/deepspeed_zero3.yaml --num_processes=8 --main_process_port 2950 src/run_kto.py config/config.yaml
# Full training with Multi GPU on 8 GPUs
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file config/multi_gpu.yaml --num_processes=8 --main_process_port 2950 src/run_kto.py config/config.yaml
```
You mighe need to change the configuration during training in each iteration in `config/config.yaml`. Here are some key configs
- `model_name_or_path`: load model checkpoint for finetuning.
- `dataset_mixer`: Choose data to mix for fine-tuning. It should be changed for each iteration.
    - default: `generated/iter0`
- `output_dir`: the output directory of finetuned model and checkpoints.
    - default: `zephyr-7b-kto-iter0`
- `prior`: The prior for nnPU loss in KTO trainer.
