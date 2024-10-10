import logging

from typing import Any

from .adapters.baseten.deploy import ensure_model_deployment, get_deployed_model_id
from .common import BaseFlowJudgeModel, VllmGenerationParams, ModelType, ModelConfig
from .adapters.baseten.adapter import BaseAPIAdapter,BasetenAPIAdapter, AsyncBasetenAPIAdapter

logger = logging.getLogger(__name__)

class BasetenModelConfig(ModelConfig):
    def __init__(
            self, 
            model_id: str, 
            generation_params: VllmGenerationParams,
            exec_async: bool = False,
            webhook_proxy_url = None,
            **kwargs: Any
        ):        
        super().__init__(model_id, ModelType.BASETEN_VLLM, generation_params, **kwargs)
        self.webhook_proxy_url = webhook_proxy_url
        self.exec_async = exec_async


class Baseten(BaseFlowJudgeModel):
    """Combined FlowJudge Model class for Baseten sync and webhook async operations.
    
    Arguments:
        api_adapter ((BasetenAPIAdapter, AsyncBasetenAPIAdapter), Optional): api handling class for Baseten requests.
        webhook_proxy_url (str, Optional): The webhook url for the proxy when exec_async is set to True.
        exec_async (bool, Optional): Async webhook execution.
    """
    def __init__(
            self,
            api_adapter: BaseAPIAdapter = None,
            webhook_proxy_url = None,
            exec_async: bool = False,
            **kwargs: Any
        ):
        if not ensure_model_deployment():
            raise BasetenError(
                status_code=1,
                message="Baseten deployment is not available."
            )
        
        baseten_model_id = get_deployed_model_id()

        if not baseten_model_id:
            raise BasetenError(
                status_code=2,
                message="Unable to retrieve Basten's deployed model id."
            )
        
        if exec_async and not webhook_proxy_url:
            raise ValueError("Webhook proxy url is required for async Baseten execution.")
        
        if api_adapter is not None and not isinstance(api_adapter, (BasetenAPIAdapter, AsyncBasetenAPIAdapter)):
            raise BasetenError(
                status_code=3,
                message="The provided API adapter is incompatible, accepted types are BasetenAPIAdapter or AsyncBasetenAPIAdapter"
            )

        if api_adapter is None:
            if exec_async is False:
                self.api_adapter = BasetenAPIAdapter(baseten_model_id)
            else:
                self.api_adapter = AsyncBasetenAPIAdapter(
                    baseten_model_id=baseten_model_id,
                    webhook_proxy_url=webhook_proxy_url
                )
        else:
            self.api_adapter = api_adapter

        # default params
        generation_params = VllmGenerationParams()
        config = BasetenModelConfig(baseten_model_id, generation_params, exec_async, webhook_proxy_url)
        self.config = config

        super().__init__(baseten_model_id, ModelType.BASETEN_VLLM, generation_params, **kwargs)

        logger.info("Successfully initialized Baseten!")

    def _generate(self, prompt: str) -> str:
        logger.info("Initiating single Baseten request")

        conversation = [{"role": "user", "content": prompt.strip()}]
        return self.api_adapter.fetch_response(conversation)

    def _batch_generate(
            self, 
            prompts: list[str], 
            use_tqdm: bool = True, 
            **kwargs: Any
        ) -> list[str]:
        logger.info("Initiating batched Baseten requests")

        conversations = [[{"role": "user", "content": prompt.strip()}] for prompt in prompts]
        return self.api_adapter.fetch_batched_response(conversations)


class BasetenError(Exception):
    """Custom exception for Baseten-related errors."""

    def __init__(self, status_code: int, message: str):
        """Initialize with a status code and message."""
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)