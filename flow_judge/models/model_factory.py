from .model_configs import MODEL_CONFIGS, ModelConfig
from .model_types import ModelType
from .models import BaseFlowJudgeModel, FlowJudgeHFModel, FlowJudgeVLLMModel, VLLMError


class ModelFactory:
    """Factory class for creating model instances based on the provided configuration."""

    @staticmethod
    def create_model(config: str | ModelConfig) -> BaseFlowJudgeModel:
        """Create and return a model based on the provided configuration."""
        if isinstance(config, str):
            if config not in MODEL_CONFIGS:
                raise ValueError(f"Unknown config: {config}")
            model_config = MODEL_CONFIGS[config]
        elif isinstance(config, ModelConfig):
            model_config = config
        else:
            raise ValueError(
                "'config' must be either a string (config name) or a ModelConfig instance"
            )

        if model_config.model_type == ModelType.TRANSFORMERS:
            return ModelFactory._create_transformers_model(model_config)
        elif model_config.model_type == ModelType.VLLM:
            return ModelFactory._create_vllm_model(model_config)
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
