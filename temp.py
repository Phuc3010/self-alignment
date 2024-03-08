from accelerate import FullyShardedDataParallelPlugin, Accelerator
import torch
import torch.distributed._shard.checkpoint as dist_cp
from transformers import AutoModelForCausalLM
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

# fsdp_plugin = FullyShardedDataParallelPlugin(
#     state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
#     optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
# )

# accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
# import torch
from datasets import load_dataset
data = load_dataset("generated/synthetic")
print(data)
# model = AutoModelForCausalLM.from_pretrained("DatPySci/tiny-llama-sft-full", device_map="cpu", torch_dtype=torch.bfloat16)
# model_path = "data/model/tiny-llama-sft-full/checkpoint-1000/pytorch_model_fsdp_0"
# state_dict = {
#         "model": model.state_dict()
#     }

# print(state_dict['model']['lm_head.weight'])
# dist_cp.load_state_dict(
#             state_dict=state_dict,
#             storage_reader=dist_cp.FileSystemReader(model_path),
#             no_dist=True,
#     )
    
# result = model.load_state_dict(state_dict["model"])
# print(model.state_dict()['lm_head.weight'])
