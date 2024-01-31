import pandas as pd
from accelerate.utils import tqdm
from contextlib import nullcontext
from trl import create_reference_model
import wandb 
import warnings
from collections import defaultdict
import sys
sys.path.append("/home/phuc/Documents/RLHF")
import numpy as np
import torch.nn.functional as F
from typing import Literal, Union, Any 
import random 
from trl.models import PreTrainedModelWrapper
from typing import Optional, Union, Callable, List, Tuple, Dict, Literal
from torch import nn
from transformers import DataCollator
import torch
from trl import DPOTrainer
from transformers.trainer_callback import TrainerCallback
from trl.trainer.utils import pad_to_length
from datasets import Dataset
from trl import IterativeSFTTrainer
from trl.trainer.utils  import DPODataCollatorWithPadding
from transformers import PreTrainedTokenizerBase, PreTrainedModel, DataCollator
from .configs import SPINConfig, DPOConfig
from transformers import logging
from transformers.trainer_utils import EvalLoopOutput
from trl.core import PPODecorators

logger = logging.get_logger(__name__)



class SPINTrainer(IterativeSFTTrainer):
    def __init__(
        self,
        model: PreTrainedModel = None,
        ref_model: Optional[PreTrainedModelWrapper]=None,
        beta: float=0.1,
        loss_type: Literal["sigmoid", "linear"]="sigmoid",
        label_smoothing: float=0.0,
        truncation_mode="keep_end",
        label_pad_token_id: int = -100,
        padding_value: int = None,
        args: SPINConfig= None,
        tokenizer: PreTrainedTokenizerBase = None,
        precompute_ref_log_probs: bool = False,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        data_collator: Optional[DataCollator] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: int=None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        optimize_device_cache: Optional[bool] = False,
    ):
        if data_collator is None:
            warnings.warn("No data collator is provided. Using 'DataCollatorForLanguageModeling'")
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id)
            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )
        else:
            data_collator = data_collator
        if max_prompt_length is None:
            max_prompt_length = 256

        
        super().__init__(model, args, tokenizer, optimizers, data_collator, eval_dataset, max_length, truncation_mode, preprocess_logits_for_metrics, compute_metrics, optimize_device_cache)
        print(self.data_collator)
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.model = model
        self.tokenizer = tokenizer
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.label_pad_token_id = label_pad_token_id
        if ref_model:
            self.ref_model = ref_model
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        if precompute_ref_log_probs:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)
        self.precompute_ref_log_probs = precompute_ref_log_probs
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self.max_prompt_length = max_prompt_length
    
    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = nullcontext

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    def build_tokenized_answer(self, prompt, answer):
        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        response_token_ids_start_idx = len(prompt_input_ids)

        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )
    
    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]
        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)}")
        chosen_tokens = self.build_tokenized_answer(prompt, chosen)

        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = self.build_tokenized_answer(prompt, rejected)

        # Last prompt token might get merged by tokenizer and
        # it should not be included for generation if that happens
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]

        # Make sure prompts only have one different token at most an
        # and length only differs by 1 at most
        num_diff_tokens = sum(
            [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
        )
        num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the "
                "last token due to tokenizer merge ops."
            )

        # add BOS token to head of prompt
        prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
        chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
        rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

        # add EOS token to end of answer
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(rejected_tokens["prompt_input_ids"])

        for k, toks in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens
        return batch

    @torch.no_grad()
    def generate(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = False,
        generate_ref_response: bool = False,
        **generation_kwargs,
    ):
        if generate_ref_response:
            ref_model = self.ref_model
        if isinstance(query_tensor, List):
            response = self._generate_batched(
                self.model,
                query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )
            if generate_ref_response:
                with self.optional_peft_ctx():
                    ref_response = self._generate_batched(
                        ref_model,
                        query_tensor,
                        length_sampler=length_sampler,
                        batch_size=batch_size,
                        return_prompt=return_prompt,
                        **generation_kwargs,
                    )

        else:
            if len(query_tensor.shape) == 2:
                raise ValueError(
                    "query_tensor must be a tensor of shape (`seq_len`) or a list of tensors of shape (`seq_len`)"
                )

            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()
            response = self.accelerator.unwrap_model(self.model).generate(
                input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
            )
            if generate_ref_response:
                with self.optional_peft_ctx():
                    ref_response = ref_model.generate(input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs)

            if not return_prompt:
                response = response[:, query_tensor.shape[0] :]
                if generate_ref_response:
                    ref_response = ref_response[:, query_tensor.shape[0] :]

        response = [self.tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response]
        if generate_ref_response:
            ref_response = [self.tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in ref_response]
            return response, ref_response
        return response

    def _generate_batched(
        self,
        model: PreTrainedModelWrapper,
        query_tensors: List[torch.Tensor],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        remove_padding: bool = True,
        **generation_kwargs,
    ):
        outputs = []

        padding_side_default = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.accelerator.device)

            generations = self.accelerator.unwrap_model(model).generate(**padded_inputs, **generation_kwargs)

            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                output = generation[(1 - mask).sum() :]  # remove padding

                if not return_prompt:
                    output = output[(mask).sum() :]  # remove prompt

                if remove_padding and self.tokenizer.eos_token_id in output:
                    pad_mask = output == self.tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                    output = output[: pad_start + 1]  # keep the eos token at the end

                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs

    def spin_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = 0
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = -self.beta*logits
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards
    
    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)
    
    # def update_dataset(self, generated_responses):
    #     prompts = [ele['prompt'] for ele in self.train_dataset]
    #     chosen = [ele['chosen'] for ele in self.train_dataset]
    #     self.train_dataset = Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": generated_responses})
    #     self.train_dataset = self.train_dataset.map(self.tokenize_row)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.spin_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        # mask = batch['chosen_attention_mask']
        # entropy = (-policy_chosen_logps*mask).sum(axis=1).mean()

        # metrics[f"{prefix}logps/entropy"] = entropy.detach()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics
    
    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)
        return concatenated_batch

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        compute_loss_context_manager = nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
    
    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
        
    @torch.no_grad()
    def update_reference_model(self):
        self.ref_model = create_reference_model(self.model)
    
    @PPODecorators.empty_device_cache()
    def step(
        self,
        query: List[str]=None,
        chosen: List[str]=None,
        texts: Optional[List[str]] = None,
    ):
        """
        Run an optimisation step given a list of input_ids, attention_mask, and labels or a list of text and text_labels.
        Args:
            input_ids (List[`torch.LongTensor`]):
                List of tensors containing the input_ids (if not provided, text will be used)
            attention_mask (List[`torch.LongTensor`], , *optional*):
                List of tensors containing the attention_mask
            labels (List[`torch.FloatTensor`], *optional*):
                List of tensors containing the labels (if set to None, will default to input_ids)
            texts (List[`str`], *optional*):
                List of strings containing the text input (if not provided, input_ids will directly be used)
            texts_labels (List[`str`], *optional*):
                List of strings containing the text labels (if set to None, will default to text)
        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        self.model.train()

        if self.state.global_step == 0:
            self.tr_loss = torch.tensor(0.0).to(self.args.device)
            self._globalstep_last_logged = self.state.global_step

        if texts is None or query is None or chosen is None:
            raise ValueError("Step should include `query`, `chosen`, `texts` as keyword arguments.")
        batch_data = Dataset.from_dict({"prompt": query, "chosen": chosen, "rejected": texts})
        batch_data = batch_data.map(self.tokenize_row)
        if self.precompute_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(torch.utils.data.DataLoader(batch_data, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            batch_data = batch_data.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            batch_data = batch_data.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )
        
        step_dataloader = torch.utils.data.DataLoader(
            batch_data,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )
        for epoch in range(self.args.num_train_epochs):
            for _, batch in enumerate(step_dataloader):
                with self.accelerator.accumulate(self.model):
                    loss, metrics = self.compute_loss(self.model, batch, return_outputs=True)

                    if self.args.n_gpu > 1:
                        loss = loss.mean()

                    tr_loss_step = loss.detach()

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients and self.args.max_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    self.state.global_step += 1

                    # update stats etc
                    self.tr_loss += tr_loss_step

                    self._maybe_log_save_evaluate()
                    self.log(metrics)
        self.update_reference_model()

    def _maybe_log_save_evaluate(self):
        # check if eval is required
        if self.args.eval_steps is not None:
            if self.state.global_step % self.args.eval_steps == 0 and self.state.global_step != 0:
                self.evaluate(self.eval_dataset)

        # check if logging is required
        if self.args.logging_steps is not None:
            if self.state.global_step % self.args.logging_steps == 0 and self.state.global_step != 0:
                logs: Dict[str, float] = {}

                tr_loss_scalar = self._nested_gather(self.tr_loss).mean().item()

                # reset tr_loss to zero
                self.tr_loss -= self.tr_loss

                logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
                logs["learning_rate"] = self._get_learning_rate()

                self._globalstep_last_logged = self.state.global_step

                self.log(logs)
            
    def evaluation_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output
    
    # def log(self, logs: Dict[str, float]) -> None:
    #     """
    #     Log `logs` on the various objects watching training.

    #     Subclass and override this method to inject custom behavior.

    #     Args:
    #         logs (`Dict[str, float]`):
    #             The values to log.
    #     """
    #     if self.state.epoch is not None:
    #         logs["epoch"] = round(self.state.epoch, 2)
    #     if self.args.include_num_input_tokens_seen:
    #         logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

    #     output = {**logs, **{"step": self.state.global_step}}
    #     self.state.log_history.append(output)
    #     self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
    

class WeightedDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid",
        args: DPOConfig = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: str = None,
        ref_adapter_name: str = None,
    ):
        super().__init__(self, model, ref_model, beta, label_smoothing, loss_type, args, data_collator, label_pad_token_id, padding_value, truncation_mode, train_dataset, eval_dataset,\
                           tokenizer, model_init, callbacks, optimizers, preprocess_logits_for_metrics, max_length, max_prompt_length, max_target_length, peft_config, is_encoder_decoder,\
                            disable_dropout, generate_during_eval, compute_metrics, precompute_ref_log_probs, model_init_kwargs, ref_model_init_kwargs, model_adapter_name, ref_adapter_name)
        self.k = 0 
        self.step = 0
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )
        weights = nn.Sigmoid()(-chosen_rewards).detach()
        pi_logratios = weights*policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = 0
        else:
            ref_logratios = weights*reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        return losses, chosen_rewards, rejected_rewards, weights
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards, weights = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/weights"] = weights.mean().cpu()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics
    
# if __name__=="__main__":
#     model_id = "EleutherAI/pythia-410m-deduped-v0"
#     dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:16]") 
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     tokenizer.pad_token = tokenizer.eos_token 
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     column_names = list(dataset.features)
#     raw_dataset =  dataset.map(
#         apply_single_turn_template,
#         fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
#         num_proc=4,
#         remove_columns=column_names,
#         desc="Applying chat template",
#     )

#     args = SPINConfig(
#         bf16=True,
#         do_eval=False ,
#         evaluation_strategy="no",
#         eval_steps=None,
#         gradient_accumulation_steps=4,
#         gradient_checkpointing=True,
#         gradient_checkpointing_kwargs={"use_reentrant": False}, 
#         hub_model_id="pythia-160m-spin-full",
#         learning_rate=5e-7,
#         log_level="info",
#         logging_steps=1,
#         lr_scheduler_type="cosine",
#         num_iters=4,
#         num_synthetic=16_000,
#         max_length=512,
#         max_prompt_length=384,
#         num_train_epochs=2,
#         output_dir="data/pythia-160m-spin-full",
#         per_device_eval_batch_size=4,
#         per_device_train_batch_size=4,
#         push_to_hub=True,
#         save_strategy="no",
#         save_steps=100,
#         seed=42,
#         warmup_ratio=0.1,
#         optim="rmsprop",
#         save_total_limit=1,
#     ) 
#     args.max_steps = len(raw_dataset)*args.num_train_epochs*args.num_iters//args.per_device_train_batch_size
#     raw_dataset = raw_dataset.rename_columns(
#                 {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
#             )
         
#     model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", use_cache=False)
#     trainer = SPINTrainer(model, max_length=512, args=args,  tokenizer=tokenizer)
#     generation_kwargs = {
#         "max_new_tokens": 128,
#         "do_sample": True,
#         "pad_token_id": tokenizer.pad_token_id,
#         "temperature": 1,
#         "top_p": 1.0
#     }
#     from datasets import Dataset
#     prompt = [ele for ele in raw_dataset['prompt']]
#     chosen = [ele for ele in raw_dataset['chosen']]
#     query_tensor = [tokenizer(ele, return_tensors="pt").input_ids.squeeze() for ele in prompt]

#     # Despite returning the usual output, the streamer will also print the generated text to stdout.
#     import pandas as pd
#     for iter in range(args.num_iteTrainingArgumentsrs):
#         generate_response = trainer.generate(query_tensor, **generation_kwargs, batch_size=16)

#         # for index in random.sample(range(len(prompt)), 3):
#         #     query = prompt[index]
#         #     golden = chosen[index]
#         #     rejected = generate_response[index]
#         #     print(f"Prompt: {query}\n")
#         #     print(f"{golden}\n")
#         #     print(f"Generated: {rejected}\n")

#         trainer.step(prompt, chosen, generate_response)