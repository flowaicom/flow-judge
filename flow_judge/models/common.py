from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel


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
    def _generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        pass

    @abstractmethod
    def _batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
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
    async def _async_generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt asynchronously."""
        pass

    @abstractmethod
    async def _async_batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        """Generate responses for multiple prompts asynchronously."""
        pass


class GenerationParams(BaseModel):
    """Configuration parameters for text generation."""

    temperature: float = 0.1
    top_p: float = 0.95
    max_new_tokens: int = 1000
    do_sample: bool = True


class VllmGenerationParams(GenerationParams):
    """Configuration parameters specific to VLLM text generation."""

    max_tokens: int | None = None
    stop_token_ids: list[int] = [32007, 32001, 32000]

    def __init__(self, **data):
        """Initialize VllmGenerationParams with given data.

        :param data: Keyword arguments to initialize the parameters.
        """
        super().__init__(**data)
        self.max_tokens = self.max_new_tokens
        del self.max_new_tokens
        del self.do_sample


class ModelType(Enum):
    """Enum for the type of model."""

    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    VLLM_ASYNC = "vllm_async"
    LLAMAFILE = "llamafile"


class Engine(Enum):
    """Enum for the type of engine used for text generation."""

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
        generation_params: dict[str, Any],
        **kwargs: Any,
    ):
        """Initialize ModelConfig with model details and generation parameters.

        :param model_id: Identifier for the model.
        :param model_type: Type of the model.
        :param generation_params: Parameters for text generation.
        :param kwargs: Additional keyword arguments.
        """
        self.model_id = model_id
        self.model_type = model_type
        self.generation_params = generation_params
        self.kwargs = kwargs
