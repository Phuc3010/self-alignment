python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 0 --output_dir generated/iter0 --
python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 1 --output_dir generated/iter0
python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 2 --output_dir generated/iter0
python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 3 --output_dir generated/iter0
python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 4 --output_dir generated/iter0
python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 5 --output_dir generated/iter0
python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 6 --output_dir generated/iter0
python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 7 --output_dir generated/iter0
python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 8 --output_dir generated/iter0
python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 9 --output_dir generated/iter0


# Generate for the test split as well
python3 src/generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir generated/synthetic --frac_len 2000 --data_frac 1 --split test --output_dir generated/iter0

python3 src/convert_data.py --input_dir generated/iter0 --split test
python3 src/convert_data.py --input_dir generated/iter0 --split train
