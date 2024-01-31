import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys
sys.path.append("/home/phuc/Documents/RLHF")
from src.alignment import ModelArguments, DataArguments, H4ArgumentParser, get_kbit_device_map, get_peft_config, get_quantization_config, get_tokenizer
from trl import RewardConfig, RewardTrainer

tqdm.pandas()


if __name__ == "__main__":
    parser = H4ArgumentParser((ModelArguments, DataArguments, RewardConfig))
    model_args, data_args, training_args = parser.parse()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    quantization_config = get_quantization_config(model_args)
    if training_args.bf16:
        torch_dtype = torch.bfloat16
        training_args.bf16 = False
    elif training_args.fp16:
        torch_dtype = torch.float16
        training_args.fp16 = False

    model_kwargs = dict(
        revision=model_args.model_revision,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = get_tokenizer(model_args, data_args)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=1, **model_kwargs
    )
    model.config.pad_token_id = model.config.eos_token_id

    ################
    # Dataset
    ################
    raw_datasets = load_dataset("Anthropic/hh-rlhf")
    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids_chosen"]) <= training_args.max_length
        and len(x["input_ids_rejected"]) <= training_args.max_length
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(
            model_args
        ),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)