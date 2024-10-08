from typing import Optional

from .base import AsyncBaseFlowJudgeModel, BaseFlowJudgeModel
from .huggingface import FlowJudgeHFModel
from .model_configs import MODEL_CONFIGS, ModelConfig, BaseModelConfig, RemoteModelConfig
from .model_types import ModelType
from .vllm import AsyncFlowJudgeVLLMModel, FlowJudgeVLLMModel, VLLMError
from .remote import FlowJudgeRemoteModel
from .adapters.base import BaseAPIAdapter

class ModelFactory:
    """Factory class for creating model instances based on the provided configuration."""

    @staticmethod
    def create_model(config: str | ModelConfig, api_adapter: Optional[BaseAPIAdapter] = None) -> BaseFlowJudgeModel:
        """Create and return a model based on the provided configuration."""
        if isinstance(config, str):
            if config not in MODEL_CONFIGS:
                raise ValueError(f"Unknown config: {config}")
            model_config = MODEL_CONFIGS[config]
        elif isinstance(config, BaseModelConfig):
            model_config = config
        else:
            raise ValueError(
                "'config' must be either a string (config name) or a ModelConfig instance"
            )

        if model_config.model_type == ModelType.TRANSFORMERS:
            return ModelFactory._create_transformers_model(model_config)
        elif model_config.model_type == ModelType.VLLM:
            return ModelFactory._create_vllm_model(model_config)
        elif model_config.model_type == ModelType.VLLM_ASYNC:
            return ModelFactory._create_vllm_async_model(model_config)
        elif model_config.model_type == ModelType.REMOTE_HOSTING:
            if not api_adapter:
                raise ValueError("API adapter is required for remote models.")
            return ModelFactory._create_remote_model(model_config, api_adapter)
        else:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")

    @staticmethod
    def _create_transformers_model(config: ModelConfig) -> FlowJudgeHFModel:
        """Create and return a Transformers-based model."""
        return FlowJudgeHFModel(
            model_id=config.model_id, generation_params=config.generation_params, **config.hf_kwargs
        )

    @staticmethod
    def _create_vllm_model(config: ModelConfig) -> FlowJudgeVLLMModel:
        """Create and return a vLLM-based model."""
        try:
            return FlowJudgeVLLMModel(
                model=config.model_id,
                generation_params=config.generation_params,
                **config.vllm_kwargs,
            )
        except VLLMError as e:
            raise ValueError(f"Failed to create vLLM model: {e.message}") from e

    @staticmethod
    def _create_vllm_async_model(config: ModelConfig) -> AsyncFlowJudgeVLLMModel:
        """Create and return an asynchronous vLLM-based model."""
        try:
            return AsyncFlowJudgeVLLMModel(
                model=config.model_id,
                generation_params=config.generation_params,
                **config.vllm_kwargs,
            )
        except VLLMError as e:
            raise ValueError(f"Failed to create asynchronous vLLM model: {e.message}") from e
        
    @staticmethod
    def _create_remote_model(config: RemoteModelConfig, api_adapter: BaseAPIAdapter) -> FlowJudgeRemoteModel:
        return FlowJudgeRemoteModel(
            model_id=config.model_id,
            model_type=config.model_type,
            generation_params=config.generation_params,
            api_adapter=api_adapter
        )

    @staticmethod
    def is_async_model(model: BaseFlowJudgeModel | AsyncBaseFlowJudgeModel) -> bool:
        """Check if the given model is an asynchronous model."""
        return isinstance(model, AsyncBaseFlowJudgeModel)
