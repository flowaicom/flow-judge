from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .adapters.base import BaseAPIAdapter


class BaseFlowJudgeModel(ABC):
    """Base class for all FlowJudge models."""

    def __init__(
        self, model_id: str, model_type: str, generation_params: dict[str, Any], **kwargs: Any
    ) -> None:
        """Initialize the base FlowJudge model."""
        self.metadata: dict[str, Any] = {
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
    ) -> None:
        """Initialize the base asynchronous FlowJudge model."""
        self.metadata: dict[str, Any] = {
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


class FlowJudgeRemoteModel(BaseFlowJudgeModel):
    """Flow judge model class for remote hosting."""

    def __init__(
        self,
        model_id: str,
        model_type: str,
        generation_params: dict[str, Any],
        api_adapter: BaseAPIAdapter,
        **remote_kwargs: Any,
    ):
        """Initialize the FlowJudge remote model class.

        :param model_id: The ID of the model.
        :param model_type: Type of the model based on ModelType.
        :param generation_params: Relevant generation params for the model type.
        :param remote_kwargs: Keyword arguments to initialize the parameters.
        """
        super().__init__(model_id, model_type, generation_params, **remote_kwargs)

        if not isinstance(api_adapter, BaseAPIAdapter):
            raise ValueError("Invalid Adapter type. Use BaseAPIAdapter.")

        self.api_adapter = api_adapter

    def generate(self, prompt: str) -> str:
        """Single generation request."""
        conversation = [{"role": "user", "content": prompt.strip()}]
        return self.api_adapter.fetch_response(conversation)

    def batch_generate(self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any) -> list[str]:
        """Batched generation request."""
        conversations = [[{"role": "user", "content": prompt.strip()}] for prompt in prompts]
        return self.api_adapter.fetch_batched_response(conversations)


class GenerationParams(BaseModel):
    """Configuration parameters for text generation."""

    temperature: float = Field(default=0.1, description="Sampling temperature")
    top_p: float = Field(default=0.95, description="Top-p sampling parameter")
    max_new_tokens: int = Field(
        default=1000, description="Maximum number of new tokens to generate"
    )
    do_sample: bool = Field(default=True, description="Whether to use sampling for generation")


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
    BASETEN_VLLM = "baseten_vllm"
    BASETEN_VLLM_ASYNC = "baseten_vllm_async"


class Engine(Enum):
    """Enum for the type of engine used for text generation."""

    VLLM: str = "vllm"
    VLLM_ASYNC: str = "vllm_async"
    HF: str = "hf"  # HF stands for Hugging Face (Transformers)
    LLAMAFILE: str = "llamafile"


class ModelConfig:
    """Base configuration for a model."""

    def __init__(
        self,
        model_id: str,
        model_type: ModelType,
        generation_params: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Initialize ModelConfig with model details and generation parameters.

        :param model_id: Identifier for the model.
        :param model_type: Type of the model.
        :param generation_params: Parameters for text generation.
        :param kwargs: Additional keyword arguments.
        """
        self.model_id: str = model_id
        self.model_type: ModelType = model_type
        self.generation_params: dict[str, Any] = generation_params
        self.kwargs: dict[str, Any] = kwargs
