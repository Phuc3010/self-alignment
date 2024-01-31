import logging
import random
import sys
from trl import create_reference_model
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
# import sys
# sys.path.append("/home/phuc/Documents/RLHF")

from src.alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    # apply_single_turn_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from typing import Literal
from peft import PeftConfig, PeftModel
from src.alignment.trainer import SPINConfig, SPINTrainer


logger = logging.getLogger(__name__)

INSTRUCTION_TEMPLATE = "### Instruction: {prompt}\n\n### Response: "
def apply_single_turn_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"]
):
    if task in ["sft", "generation"]:
        messages = example['messages']
        for msg in messages: 
            if msg['role'] == "assistant":
                response = msg['content']
                break
        example['text'] = INSTRUCTION_TEMPLATE.format_map({"prompt": example['prompt']}) + response
    elif task in ["dpo", "rm"]:
        try:
            chosen_messages = example['chosen']
            rejected_messages = example['rejected']
        except:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
        for msg in chosen_messages:
            if msg['role'] == "assistant":
                chosen_response = msg['content']
                break
        for msg in rejected_messages:
            if msg['role'] == "assistant":
                rejected_response = msg['content']
                break
        prompt = example['prompt']
        example["text_prompt"] = INSTRUCTION_TEMPLATE.format_map({"prompt": prompt})
        example['text_chosen'] = chosen_response
        example['text_rejected'] = rejected_response
    return example


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SPINConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    template_func = apply_chat_template if data_args.turn_type == "multi" else apply_single_turn_template
    raw_datasets = raw_datasets.map(
        template_func,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
                {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
            )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    if training_args.bf16:
        training_args.bf16 = False
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        training_args.fp16 = False
        torch_dtype = torch.float16

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    num_generated_data = min(len(raw_datasets['train']), training_args.num_synthetic)
    training_args.max_steps = num_generated_data*training_args.num_train_epochs*training_args.num_iters//training_args.per_device_train_batch_size
    model = model_args.model_name_or_path
    #########################
    # Instantiate DPO trainer
    #########################
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model.generate()
    trainer = SPINTrainer(
        model,
        args=training_args,
        beta=training_args.beta,
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        precompute_ref_log_probs=True,
        max_prompt_length=training_args.max_prompt_length,
    )

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpointk
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_length": training_args.max_length,
    "max_new_tokens": 256
    }

    prompt = [ele for ele in raw_datasets['train']['prompt']]
    chosen = [ele for ele in raw_datasets['train']['chosen']]
    query_tensor = [tokenizer(ele, return_tensors="pt").input_ids.squeeze() for ele in prompt]
    num_synthetic = training_args.num_synthetic
    num_generated = min(len(prompt), num_synthetic)
    print(f"Number of generated samples per iteration {num_generated}")
    for iter in range(training_args.num_iters):
        idxs = random.sample(range(len(prompt)), num_generated)
        sampled_prompt = [prompt[idx] for idx in idxs]
        sampled_chosen = [chosen[idx] for idx in idxs]
        sampled_query_tensor = [query_tensor[idx] for idx in idxs]
        generated_response = trainer.generate(sampled_query_tensor, **generation_kwargs, batch_size=16)
        trainer.step(sampled_prompt, sampled_chosen, generated_response)
        logger.info("*** Save model ***")
        trainer.save_model(training_args.output_dir+f"/iter{iter+1}")

    logger.info("*** Training complete ***")
    logger.info(f"Model saved to {training_args.output_dir}")

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
