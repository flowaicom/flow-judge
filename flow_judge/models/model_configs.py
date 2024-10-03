from typing import Any

from .model_types import ModelType


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
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


MODEL_CONFIGS = {
    "Flow-Judge-v0.1-AWQ": ModelConfig(
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
    ),
    "Flow-Judge-v0.1": ModelConfig(
        model_id="flowaicom/Flow-Judge-v0.1",
        model_type=ModelType.VLLM,
        generation_params={
            "temperature": 0.1,
            "top_p": 0.95,
            "max_tokens": 1000,
        },
        max_model_len=8192,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        disable_sliding_window=True,
        gpu_memory_utilization=0.90,
        max_num_seqs=256,
    ),
    "Flow-Judge-v0.1_HF": ModelConfig(
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
    ),
    "Flow-Judge-v0.1_HF_no_flsh_attn": ModelConfig(
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
    ),
    "Flow-Judge-v0.1-AWQ-Async": ModelConfig(
        model_id="flowaicom/Flow-Judge-v0.1-AWQ",
        model_type=ModelType.VLLM_ASYNC,
        generation_params={"temperature": 0.1, "top_p": 0.95, "max_tokens": 1000},
        max_model_len=8192,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        disable_sliding_window=True,
        disable_log_requests=False,
    ),
}


def get_available_configs():
    """Get the available model configs."""
    return list(MODEL_CONFIGS.keys())
