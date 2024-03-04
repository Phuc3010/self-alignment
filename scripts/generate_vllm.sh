python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --output_dir generated/iter0 --world_size 8

# Generate for the test split as well
python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --frac_len 2000 --data_frac 1 --split test --output_dir generated/iter0 --world_size 8 

python3 src/convert_data.py --input_dir generated/iter0 --split test
python3 src/convert_data.py --input_dir generated/iter0 --split train
