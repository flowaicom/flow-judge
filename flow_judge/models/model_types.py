from enum import Enum


class ModelType(Enum):
    """Enum for the type of model."""

    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    VLLM_ASYNC = "vllm_async"
    REMOTE_HOSTING = "remote"
