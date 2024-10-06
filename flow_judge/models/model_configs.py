from typing import Any, Union, TYPE_CHECKING
from enum import Enum
from .model_types import ModelType
import torch
import os

class Engine(Enum):
    VLLM = "vllm"
    VLLM_ASYNC = "vllm_async"
    HF = "hf"  # HF stands for Hugging Face (Transformers)
    LLAMAFILE = "llamafile"  # Add this line

class ModelConfig:
    """Configuration for a model."""

    def __init__(
        self,
        model_id: str,
        model_type: ModelType,
        generation_params: dict[str, Any],
        **kwargs: dict[str, Any],
    ):
        """Initialize the model config."""
        self.model_id = model_id
        self.model_type = model_type
        self.generation_params = generation_params
        if model_type == ModelType.TRANSFORMERS:
            self.hf_kwargs = kwargs
        elif model_type == ModelType.VLLM or model_type == ModelType.VLLM_ASYNC:
            self.vllm_kwargs = kwargs
        elif model_type == ModelType.LLAMAFILE:
            self.llamafile_kwargs = kwargs
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

# Define configurations
Vllm = ModelConfig(
    model_id="flowaicom/Flow-Judge-v0.1",
    model_type=ModelType.VLLM,
    generation_params={"temperature": 0.1, "top_p": 0.95, "max_tokens": 1000},
    max_model_len=8192,
    trust_remote_code=True,
    enforce_eager=True,
    dtype="bfloat16",
    disable_sliding_window=True,
    gpu_memory_utilization=0.90,
    max_num_seqs=256,
)

VllmAwq = ModelConfig(
    model_id="flowaicom/Flow-Judge-v0.1-AWQ",
    model_type=ModelType.VLLM,
    generation_params={"temperature": 0.1, "top_p": 0.95, "max_tokens": 1000},
    max_model_len=8192,
    trust_remote_code=True,
    enforce_eager=True,
    dtype="bfloat16",
    disable_sliding_window=True,
    gpu_memory_utilization=0.90,
    max_num_seqs=256,
    quantization="awq_marlin",
)

VllmAwqAsync = ModelConfig(
    model_id="flowaicom/Flow-Judge-v0.1-AWQ",
    model_type=ModelType.VLLM_ASYNC,
    generation_params={"temperature": 0.1, "top_p": 0.95, "max_tokens": 1000},
    max_model_len=8192,
    trust_remote_code=True,
    enforce_eager=True,
    dtype="bfloat16",
    disable_sliding_window=True,
    disable_log_requests=False,
)

Hf = ModelConfig(
    model_id="flowaicom/Flow-Judge-v0.1",
    model_type=ModelType.TRANSFORMERS,
    generation_params={
        "temperature": 0.1,
        "top_p": 0.95,
        "max_new_tokens": 1000,
        "do_sample": True,
    },
    device_map="auto",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)

HfNoFlashAttn = ModelConfig(
    model_id="flowaicom/Flow-Judge-v0.1",
    model_type=ModelType.TRANSFORMERS,
    generation_params={
        "temperature": 0.1,
        "top_p": 0.95,
        "max_new_tokens": 1000,
        "do_sample": True,
    },
    device_map="auto",
    torch_dtype="bfloat16",
)


LlamafileConfig = ModelConfig(
    model_id="sariola/flow-judge-llamafile",
    model_type=ModelType.LLAMAFILE,
    generation_params={
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": 2000,
        "context_size": 16384,
        "gpu_layers": 34,
        "thread_count": 32,
        "batch_size": 32,
        "max_concurrent_requests": 1,
        "stop": ["<|endoftext|>"]
    },
    model_filename="flow-judge.llamafile",
    cache_dir=os.path.expanduser("~/.cache/flow-judge"),
    port=8085,
    disable_kv_offload=False,
    llamafile_kvargs="",
)

MODEL_CONFIGS = {
    "Vllm": Vllm,
    "VllmAwq": VllmAwq,
    "Hf": Hf,
    "HfNoFlashAttn": HfNoFlashAttn,
    "VllmAwqAsync": VllmAwqAsync,
    "Llamafile": LlamafileConfig,
}


def get_available_configs():
    """Get the available model configs."""
    return list(MODEL_CONFIGS.keys())
