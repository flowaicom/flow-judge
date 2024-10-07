from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional


class BaseFlowJudgeModel(ABC):
    """Base class for all FlowJudge models."""

    def __init__(
        self, model_id: str, model_type: str, generation_params: dict[str, Any], **kwargs: Any
    ):
        """Initialize the base FlowJudge model."""
        self.metadata = {
            "model_id": model_id,
            "model_type": model_type,
            "generation_params": generation_params,
            "kwargs": kwargs,
        }

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        pass

    @abstractmethod
    def batch_generate(self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any) -> list[str]:
        """Generate responses for multiple prompts."""
        pass


class AsyncBaseFlowJudgeModel(ABC):
    """Base class for asynchronous FlowJudge models."""

    def __init__(
        self, model_id: str, model_type: str, generation_params: dict[str, Any], **kwargs: Any
    ):
        """Initialize the base asynchronous FlowJudge model."""
        self.metadata = {
            "model_id": model_id,
            "model_type": model_type,
            "generation_params": generation_params,
            "kwargs": kwargs,
        }

    @abstractmethod
    async def async_generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt asynchronously."""
        pass

    @abstractmethod
    async def async_batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        """Generate responses for multiple prompts asynchronously."""
        pass


class ModelType(Enum):
    """Enum for the type of model."""

    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    VLLM_ASYNC = "vllm_async"
    LLAMAFILE = "llamafile"  # Add this line


class Engine(Enum):
    VLLM = "vllm"
    VLLM_ASYNC = "vllm_async"
    HF = "hf"  # HF stands for Hugging Face (Transformers)
    LLAMAFILE = "llamafile"

class ModelConfig:
    """Base configuration for a model."""

    def __init__(
        self,
        model_id: str,
        model_type: ModelType,
        generation_params: Dict[str, Any],
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.generation_params = generation_params
        self.kwargs = kwargs