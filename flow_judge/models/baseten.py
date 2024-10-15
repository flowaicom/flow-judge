import logging
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
        model_id: str,
        generation_params: VllmGenerationParams,
        exec_async: bool = False,
        webhook_proxy_url: str = None,
        async_batch_size: int = 128,
        **kwargs: Any,
    ):
        """Initialize the Baseten model config.

        : param: model_id: baseten model id.
        : param: generation_params: VllmGenerationParams.
        : param exec_async: Baseten async execution.
        : param webhook_proxy_url: Webhook URL for Baseten async execution.
        : param async_batch_size: batch size for concurrent requests to Baseten in async.
        """
        super().__init__(model_id, ModelType.BASETEN_VLLM, generation_params, **kwargs)
        self.webhook_proxy_url = webhook_proxy_url
        self.exec_async = exec_async
        self.async_batch_size = async_batch_size


class Baseten(BaseFlowJudgeModel, AsyncBaseFlowJudgeModel):
    """Combined FlowJudge Model class for Baseten sync and webhook async operations."""

    def __init__(
        self,
        api_adapter: BaseAPIAdapter = None,
        webhook_proxy_url: str = None,
        exec_async: bool = False,
        async_batch_size: int = 128,
        **kwargs: Any,
    ):
        """Initialize the Baseten Model class.

        : param api_adapter: api handling class for Baseten requests.
        : param webhook_proxy_url: The webhook url for the proxy when exec_async is set to True.
        : param exec_async: Async webhook execution.
        """
        if not ensure_model_deployment():
            raise BasetenError(status_code=1, message="Baseten deployment is not available.")

        baseten_model_id = get_deployed_model_id()

        if not baseten_model_id:
            raise BasetenError(
                status_code=2, message="Unable to retrieve Basten's deployed model id."
            )

        if exec_async and not webhook_proxy_url:
            raise ValueError("Webhook proxy url is required for async Baseten execution.")

        if async_batch_size < 1:
            raise ValueError("async_batch_size needs to be greater than 0.")

        if api_adapter is not None and not isinstance(
            api_adapter, (BasetenAPIAdapter, AsyncBasetenAPIAdapter)
        ):
            raise BasetenError(
                status_code=3,
                message="The provided API adapter is incompatible,"
                " accepted types are BasetenAPIAdapter or AsyncBasetenAPIAdapter",
            )

        if api_adapter is None:
            if exec_async is False:
                self.api_adapter = BasetenAPIAdapter(baseten_model_id)
            else:
                self.api_adapter = AsyncBasetenAPIAdapter(
                    baseten_model_id=baseten_model_id,
                    webhook_proxy_url=webhook_proxy_url,
                    batch_size=async_batch_size,
                )
        else:
            self.api_adapter = api_adapter

        # default params
        generation_params = VllmGenerationParams()
        config = BasetenModelConfig(
            baseten_model_id, generation_params, exec_async, webhook_proxy_url, async_batch_size
        )
        self.config = config

        super().__init__(baseten_model_id, ModelType.BASETEN_VLLM, generation_params, **kwargs)

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
