from typing import Any

from .base import BaseFlowJudgeModel
from .adapters.base import BaseAPIAdapter

class FlowJudgeRemoteModel(BaseFlowJudgeModel):
    """
    Flow judge model class for remote hosting.
    Expects the api_adapter to return a str message for generate.
    Expects the api_adapter to return a list of str messages for batch generate
    """
    def __init__(
            self, 
            model_id: str, 
            model_type: str, 
            generation_params: dict[str, Any],
            api_adapter: BaseAPIAdapter,
            **remote_kwargs: Any
        ):
        super().__init__(model_id, model_type, generation_params, **remote_kwargs)

        if not isinstance(api_adapter, BaseAPIAdapter):
            raise ValueError("Invalid Adapter type. Use BaseAPIAdapter.")
        
        self.api_adapter = api_adapter
    
    def generate(self, prompt: str) -> str:
        conversation = [{"role": "user", "content": prompt.strip()}]
        return self.api_adapter.fetch_response(conversation)

    def batch_generate(
            self, 
            prompts: list[str], 
            use_tqdm: bool = True, 
            **kwargs: Any
        ) -> list[str]:
        conversations = [[{"role": "user", "content": prompt.strip()}] for prompt in prompts]
        return self.api_adapter.fetch_batched_response(conversations)