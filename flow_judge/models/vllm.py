from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from flow_judge.models.base import AsyncBaseFlowJudgeModel, BaseFlowJudgeModel


class FlowJudgeVLLMModel(BaseFlowJudgeModel):
    """FlowJudge model class for vLLM."""

    def __init__(self, model: str, generation_params: dict[str, Any], **vllm_kwargs: Any):
        """Initialize the FlowJudge vLLM model."""
        super().__init__(model, "vllm", generation_params, **vllm_kwargs)
        try:
            self.model = LLM(model=model, **vllm_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        except ImportError as e:
            raise VLLMError(
                status_code=1,
                message="Failed to import 'vllm' package. Make sure it is installed correctly.",
            ) from e

        self.generation_params = generation_params

    def generate(self, prompt: str) -> str:
        """Generate a response using the FlowJudge vLLM model."""
        conversation = [{"role": "user", "content": prompt.strip()}]
        params = SamplingParams(**self.generation_params)
        outputs = self.model.chat(conversation, sampling_params=params)
        return outputs[0].outputs[0].text.strip()

    def batch_generate(self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any) -> list[str]:
        """Generate responses for multiple prompts using the FlowJudge vLLM model."""
        conversations = [[{"role": "user", "content": prompt.strip()}] for prompt in prompts]

        # Apply chat template and tokenize
        prompt_token_ids = [
            self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=True)
            for conv in conversations
        ]

        params = SamplingParams(**self.generation_params)

        outputs = self.model.generate(
            prompt_token_ids=prompt_token_ids, sampling_params=params, use_tqdm=use_tqdm
        )
        return [output.outputs[0].text.strip() for output in outputs]


class AsyncFlowJudgeVLLMModel(AsyncBaseFlowJudgeModel):
    """Asynchronous FlowJudge model class for vLLM."""

    def __init__(self, model: str, generation_params: dict[str, Any], **vllm_kwargs: Any):
        """Initialize the Asynchronous FlowJudge vLLM model."""
        super().__init__(model, "vllm_async", generation_params, **vllm_kwargs)
        try:
            engine_args = AsyncEngineArgs(model=model, **vllm_kwargs)
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        except ImportError as e:
            raise VLLMError(
                status_code=1,
                message="Failed to import 'vllm' package. Make sure it is installed correctly.",
            ) from e

        self.generation_params = generation_params

    async def agenerate(self, prompt: str) -> str:
        """Generate a response using the Asynchronous FlowJudge vLLM model."""
        conversation = [{"role": "user", "content": prompt.strip()}]
        prompt = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        params = SamplingParams(**self.generation_params)
        request_id = f"req_{id(prompt)}"

        async for output in self.engine.generate(
            inputs=prompt, sampling_params=params, request_id=request_id
        ):
            if output.finished:
                return output.outputs[0].text.strip()

        return ""  # Return empty string if no output is generated

    async def abatch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        """Generate responses for multiple prompts asynchronously."""
        conversations = [[{"role": "user", "content": prompt.strip()}] for prompt in prompts]
        formatted_prompts = [
            self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]
        params = SamplingParams(**self.generation_params)

        results = []
        for prompt in formatted_prompts:
            request_id = f"req_{id(prompt)}"
            async for output in self.engine.generate(
                inputs=prompt, sampling_params=params, request_id=request_id
            ):
                if output.finished:
                    results.append(output.outputs[0].text.strip())
                    break

        return results

    async def abort(self, request_id: str) -> None:
        """Abort a specific request."""
        await self.engine.abort(request_id)

    def shutdown(self) -> None:
        """Shut down the background loop."""
        self.engine.shutdown_background_loop()


class VLLMError(Exception):
    """Custom exception for VLLM-related errors."""

    def __init__(self, status_code: int, message: str):
        """Initialize a VLLMError with a status code and message."""
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)
