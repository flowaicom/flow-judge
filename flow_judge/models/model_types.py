from enum import Enum


class ModelType(Enum):
    """Enum for the type of model."""

    TRANSFORMERS = "transformers"
    VLLM = "vllm"
