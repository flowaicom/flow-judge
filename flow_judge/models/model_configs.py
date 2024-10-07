from typing import Any, TYPE_CHECKING
from enum import Enum
from .model_types import ModelType

class Engine(Enum):
    VLLM = "vllm"
    VLLM_ASYNC = "vllm_async"
    HF = "hf"  # HF stands for Hugging Face (Transformers)
    LLAMAFILE = "llamafile"

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
