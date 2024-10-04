import os
import requests
from typing import Any, Coroutine, Dict
from abc import ABC, abstractmethod
from .base import AsyncBaseFlowJudgeModel, BaseFlowJudgeModel

class BaseAPIAdapter(ABC):

    def __init__(self, baseten_model_id: str):
        self.baseten_model_id = baseten_model_id
        self.base_url = f"https://model-{baseten_model_id}.api.baseten.co/production"
        try:
            self.baseten_api_key = os.environ["BASETEN_API_KEY"]
        except KeyError:
            raise ValueError("BASETEN_API_KEY is not provided in the environment.")
        
    @abstractmethod
    def fetch_response(self, request_body: Dict[str, Any]) -> str:
        """Generate a response based on the given request."""
        pass

    @abstractmethod
    def fetch_batched_response(self, request_bodies: list[Dict[str, any]]) -> list[str]:
        """Generate responses for multiple requests."""
        pass

class APIAdapter(BaseAPIAdapter):
    """API utility class to execute sync requests from Baseten remote model hosting."""
    def __init__(self, baseten_model_id: str):
        super().__init__(baseten_model_id)

    def fetch_response(self, request_body: Dict[str, Any]) -> str:
        resp = requests.post(
            url=self.base_url + "/predict",
            headers={"Authorization": f"Api-Key {self.baseten_api_key}"},
            json=request_body
        )

        parsed_resp = resp.json()

        try:
            message = parsed_resp["choices"][0]["message"].strip()
            return message
        except Exception as e:
            # TODO: Log warning here (?)
            return ""
        
    def fetch_batched_response(self, request_bodies: list[Dict[str, Any]]) -> list[str]:
        resp = requests.post(
            url=self.base_url + "/predict",
            headers={"Authorization": f"Api-Key {self.baseten_api_key}"},
            json=request_bodies
        )

        parsed_resp = resp.json()
        return [choice["message"].strip() for choice in parsed_resp["choices"]]
    
    class AsyncAPIAdapter(BaseAPIAdapter):
        pass

class FlowJudgeRemoteModel(BaseFlowJudgeModel):

    def __init__(
            self, 
            model_id: str, 
            model_type: str, 
            generation_params: dict[str, Any],
            api_adapter: APIAdapter,
            **remote_kwargs: Any
        ):
        super().__init__(model_id, model_type, generation_params, **remote_kwargs)
        self.api_adapter = api_adapter
    
    def generate(self, prompt: str) -> str:
        pass

    def batch_generate(
            self, 
            prompts: list[str], 
            use_tqdm: bool = True, 
            **kwargs: Any
        ) -> list[str]:
        pass

class AsyncFlowJudgeRemoteModel(AsyncBaseFlowJudgeModel):
    
    async def async_generate(self, prompt: str) -> Coroutine[Any, Any, str]:
        pass

    async def async_batch_generate(
            self, 
            prompts: list[str], 
            use_tqdm: bool = True, 
            **kwargs: Any
        ) -> Coroutine[Any, Any, list[str]]:
        pass