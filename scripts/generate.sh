python3 generate.py --model DatPySci/pythia-1b-sft-full --input_dir generated/synthetic --frac_len 300 --data_frac 0 --world_size 1 --output_dir generated/iter0 --n 2
# python3 generate.py --model DatPySci/pythia-1b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 1 --world_size 1 --output_dir generated/iter0
# python3 generate.py --model DatPySci/pythia-1b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 2 --world_size 1 --output_dir generated/iter0
# python3 generate.py --model DatPySci/pythia-1b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 3 --world_size 1 --output_dir generated/iter0
# python3 generate.py --model DatPySci/pythia-1b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 4 --world_size 1 --output_dir generated/iter0
# python3 generate.py --model DatPySci/pythia-1b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 5 --world_size 1 --output_dir generated/iter0
# python3 generate.py --model DatPySci/pythia-1b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 6 --world_size 1 --output_dir generated/iter0
# python3 generate.py --model DatPySci/pythia-1b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 7 --world_size 1 --output_dir generated/iter0
# python3 generate.py --model DatPySci/pythia-1b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 1 --world_size 1 --output_dir generated/iter0
# python3 generate.py --model DatPySci/pythia-1b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 9 --world_size 1 --output_dir generated/iter0
# python3 generate.py --model DatPySci/pythia-1b-sft-full --input_dir generated/synthetic --frac_len 5000 --data_frac 10 --world_size 1 --output_dir generated/iter0

# Generate for the test split as well
python3 generate.py --model DatPySci/pythia-1b-sft-full --input_dir generated/synthetic --frac_len 300 --data_frac 0 --world_size 1 --output_dir generated/iter0 --split test