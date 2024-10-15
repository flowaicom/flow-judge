import logging
import warnings
from collections.abc import Coroutine
from typing import Any

from .adapters.baseten.adapter import AsyncBasetenAPIAdapter, BaseAPIAdapter, BasetenAPIAdapter
from .adapters.baseten.deploy import ensure_model_deployment, get_deployed_model_id
from .common import (
    AsyncBaseFlowJudgeModel,
    BaseFlowJudgeModel,
    ModelConfig,
    ModelType,
    VllmGenerationParams,
)

logger = logging.getLogger(__name__)


class BasetenModelConfig(ModelConfig):
    """Model config for the Baseten model class."""

    def __init__(
        self,
        generation_params: VllmGenerationParams,
        exec_async: bool = False,
        webhook_proxy_url: str | None = None,
        async_batch_size: int = 128,
        **kwargs: Any,
    ):
        """Initialize the Baseten model config.

        :param generation_params: VllmGenerationParams for text generation.
        :param exec_async: Whether to use async execution.
        :param webhook_proxy_url: Webhook URL for Baseten async execution.
        :param async_batch_size: Batch size for concurrent requests in async mode.
        :raises ValueError: If any input parameters are invalid.
        """
        model_id = kwargs.pop("_model_id", None)
        if model_id is None:
            model_id = get_deployed_model_id()
            if model_id is None:
                raise ValueError("Unable to retrieve Baseten's deployed model id.")

        model_type = ModelType.BASETEN_VLLM_ASYNC if exec_async else ModelType.BASETEN_VLLM

        if not isinstance(generation_params, VllmGenerationParams):
            raise ValueError("generation_params must be an instance of VllmGenerationParams")
        if async_batch_size <= 0:
            raise ValueError(f"async_batch_size must be > 0, got {async_batch_size}")
        if exec_async and webhook_proxy_url is None:
            raise ValueError("webhook_proxy_url is required for async execution")

        super().__init__(model_id, model_type, generation_params, **kwargs)
        self.webhook_proxy_url = webhook_proxy_url
        self.exec_async = exec_async
        self.async_batch_size = async_batch_size


class Baseten(BaseFlowJudgeModel, AsyncBaseFlowJudgeModel):
    """Combined FlowJudge Model class for Baseten sync and webhook async operations."""

    def __init__(
        self,
        api_adapter: BaseAPIAdapter | None = None,
        webhook_proxy_url: str | None = None,
        exec_async: bool = False,
        async_batch_size: int = 128,
        generation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize the Baseten Model class.

        :param api_adapter: API handling class for Baseten requests.
        :param webhook_proxy_url: The webhook url for the proxy when exec_async is True.
        :param exec_async: Whether to use async webhook execution.
        :param async_batch_size: Batch size for concurrent requests to Baseten in async.
        :param generation_params: Dictionary of parameters for text generation.
        :raises BasetenError: If Baseten deployment or model ID retrieval fails.
        :raises ValueError: If input parameters are invalid.
        """
        if not ensure_model_deployment():
            raise BasetenError(status_code=1, message="Baseten deployment is not available.")

        model_id = kwargs.pop("_model_id", None)
        if model_id is None:
            model_id = get_deployed_model_id()
            if model_id is None:
                raise BasetenError(
                    status_code=2,
                    message=(
                        "Unable to retrieve Baseten's deployed model id. "
                        "Please ensure the model is deployed or provide a custom '_model_id'."
                    ),
                )
        else:
            warnings.warn(
                f"You have entered a custom Baseten model id: '{model_id}'. "
                "Using other models may lead to unexpected behavior, and we do "
                "not handle GitHub issues for unsupported models. Proceed with caution.",
                UserWarning,
                stacklevel=2,
            )

        if exec_async and not webhook_proxy_url:
            raise ValueError("webhook_proxy_url is required for async Baseten execution.")

        if async_batch_size < 1:
            raise ValueError("async_batch_size must be greater than 0.")

        if api_adapter is not None and not isinstance(
            api_adapter, (BasetenAPIAdapter, AsyncBasetenAPIAdapter)
        ):
            raise BasetenError(
                status_code=3,
                message="Incompatible API adapter. Use BasetenAPIAdapter or AsyncBasetenAPIAdapter",
            )

        self.api_adapter = api_adapter or (
            AsyncBasetenAPIAdapter(model_id, webhook_proxy_url, async_batch_size)
            if exec_async
            else BasetenAPIAdapter(model_id)
        )

        generation_params = VllmGenerationParams(**(generation_params or {}))
        config = BasetenModelConfig(
            generation_params=generation_params,
            exec_async=exec_async,
            webhook_proxy_url=webhook_proxy_url,
            async_batch_size=async_batch_size,
            _model_id=model_id,
        )
        self.config = config

        super().__init__(model_id, config.model_type, config.generation_params, **kwargs)

        logger.info("Successfully initialized Baseten!")

    def _format_conversation(self, prompt: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": prompt.strip()}]

    def _generate(self, prompt: str) -> str:
        logger.info("Initiating single Baseten request")

        conversation = self._format_conversation(prompt)
        return self.api_adapter._fetch_response(conversation)

    def _batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        logger.info("Initiating batched Baseten requests")

        conversations = [self._format_conversation(prompt) for prompt in prompts]
        return self.api_adapter._fetch_batched_response(conversations)

    async def _async_generate(self, prompt: str) -> str:
        if self.config.exec_async:
            conversation = self._format_conversation(prompt)
            return await self.api_adapter._async_fetch_response(conversation)
        else:
            logger.error("Attempting to run an async request with a synchronous API adapter")

    async def _async_batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> Coroutine[Any, Any, list[str]]:
        if self.config.exec_async:
            conversations = [self._format_conversation(prompt) for prompt in prompts]
            return await self.api_adapter._async_fetch_batched_response(conversations)
        else:
            logger.error("Attempting to run an async request with a synchronous API adapter")


class BasetenError(Exception):
    """Custom exception for Baseten-related errors."""

    def __init__(self, status_code: int, message: str):
        """Initialize with a status code and message."""
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)
