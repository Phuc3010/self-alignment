lm_eval --model=hf --model_args pretrained=alignment-handbook/zephyr-7b-sft-full,dtype=bfloat16 --tasks gsm8k --device cuda:0 --batch_size 8 --num_fewshot 5
lm_eval --model=hf --model_args pretrained=alignment-handbook/zephyr-7b-sft-full,dtype=bfloat16 --tasks hellaswag --device cuda:0 --batch_size 8 --num_fewshot 10
lm_eval --model=hf --model_args pretrained=alignment-handbook/zephyr-7b-sft-full,dtype=bfloat16 --tasks truthfulqa_mc2 --device cuda:0 --batch_size 8 --num_fewshot 0


lm_eval --model=hf --model_args pretrained=alignment-handbook/zephyr-7b-sft-full,dtype=bfloat16 --tasks mmlu --device cuda:0 --batch_size 8 --num_fewshot 5
lm_eval --model=hf --model_args pretrained=alignment-handbook/zephyr-7b-sft-full,dtype=bfloat16 --tasks winogrande --device cuda:0 --batch_size 8 --num_fewshot 5
lm_eval --model=hf --model_args pretrained=alignment-handbook/zephyr-7b-sft-full,dtype=bfloat16 --tasks arc_challenge --device cuda:0 --batch_size 8 --num_fewshot 25 








