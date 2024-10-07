from typing import Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from flow_judge.models.base import AsyncBaseFlowJudgeModel, BaseFlowJudgeModel
from flow_judge.models.model_configs import ModelConfig, VllmConfig, VllmAwqConfig, VllmAwqAsyncConfig
from flow_judge.models.model_types import ModelType

_VllmConfig = ModelConfig(
    model_id="flowaicom/Flow-Judge-v0.1",
    model_type=ModelType.VLLM,
    generation_params={"temperature": 0.1, "top_p": 0.95, "max_tokens": 1000},
    max_model_len=8192,
    trust_remote_code=True,
    enforce_eager=True,
    dtype="bfloat16",
    disable_sliding_window=True,
    gpu_memory_utilization=0.90,
    max_num_seqs=256,
)

_VllmAwqConfig = ModelConfig(
    model_id="flowaicom/Flow-Judge-v0.1-AWQ",
    model_type=ModelType.VLLM,
    generation_params={"temperature": 0.1, "top_p": 0.95, "max_tokens": 1000},
    max_model_len=8192,
    trust_remote_code=True,
    enforce_eager=True,
    dtype="bfloat16",
    disable_sliding_window=True,
    gpu_memory_utilization=0.90,
    max_num_seqs=256,
    quantization="awq_marlin",
)

_VllmAwqAsyncConfig = ModelConfig(
    model_id="flowaicom/Flow-Judge-v0.1-AWQ",
    model_type=ModelType.VLLM_ASYNC,
    generation_params={"temperature": 0.1, "top_p": 0.95, "max_tokens": 1000},
    max_model_len=8192,
    trust_remote_code=True,
    enforce_eager=True,
    dtype="bfloat16",
    disable_sliding_window=True,
    disable_log_requests=False,
)

import asyncio

class Vllm(BaseFlowJudgeModel, AsyncBaseFlowJudgeModel):
    """Combined FlowJudge model class for vLLM supporting both sync and async operations."""

    def __init__(self, model: str = None, generation_params: dict[str, Any] = None, quantized: bool = False, exec_async: bool = False, **kwargs: Any):
        """Initialize the FlowJudge vLLM model."""
        if exec_async:
            config = _VllmAwqAsyncConfig
        elif quantized:
            config = _VllmAwqConfig
        else:
            config = _VllmConfig

        model = model or config.model_id
        generation_params = generation_params or config.generation_params
        kwargs = {**config.vllm_kwargs, **kwargs}

        super().__init__(model, "vllm", generation_params, **kwargs)

        self.exec_async = exec_async
        self.generation_params = generation_params

        try:
            if not torch.cuda.is_available():
                raise VLLMError(
                    status_code=2,
                    message="GPU is not available. vLLM requires a GPU to run. Check https://docs.vllm.ai/en/latest/getting_started/installation.html for installation requirements.",
                )

            if exec_async:
                engine_args = AsyncEngineArgs(model=model, **kwargs)
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            else:
                self.model = LLM(model=model, **kwargs)

            self.tokenizer = AutoTokenizer.from_pretrained(model)
        except ImportError as e:
            raise VLLMError(
                status_code=1,
                message="Failed to import 'vllm' package. Make sure it is installed correctly.",
            ) from e

    def generate(self, prompt: str) -> str:
        """Generate a response using the FlowJudge vLLM model."""
        if self.exec_async:
            return asyncio.run(self._async_generate(prompt))

        conversation = [{"role": "user", "content": prompt.strip()}]
        params = SamplingParams(**self.generation_params)
        outputs = self.model.chat(conversation, sampling_params=params)
        return outputs[0].outputs[0].text.strip()

    def batch_generate(self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any) -> list[str]:
        """Generate responses for multiple prompts using the FlowJudge vLLM model."""
        if self.exec_async:
            return asyncio.run(self._async_batch_generate(prompts, use_tqdm, **kwargs))

        conversations = [[{"role": "user", "content": prompt.strip()}] for prompt in prompts]
        prompt_token_ids = [
            self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=True)
            for conv in conversations
        ]
        params = SamplingParams(**self.generation_params)
        outputs = self.model.generate(
            prompt_token_ids=prompt_token_ids, sampling_params=params, use_tqdm=use_tqdm
        )
        return [output.outputs[0].text.strip() for output in outputs]

    async def _async_generate(self, prompt: str) -> str:
        """Internal method for async generation."""
        conversation = [{"role": "user", "content": prompt.strip()}]
        prompt = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        params = SamplingParams(**self.generation_params)
        request_id = f"req_{id(prompt)}"

        async for output in self.engine.generate(inputs=prompt, sampling_params=params, request_id=request_id):
            if output.finished:
                return output.outputs[0].text.strip()

        return ""

    async def _async_batch_generate(self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any) -> list[str]:
        """Internal method for async batch generation."""
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
        """Abort a specific request (async only)."""
        if self.exec_async:
            await self.engine.abort(request_id)

    def shutdown(self) -> None:
        """Shut down the background loop (async only)."""
        if self.exec_async:
            self.engine.shutdown_background_loop()

# Keep the VLLMError class as is
class VLLMError(Exception):
    """Custom exception for VLLM-related errors."""

    def __init__(self, status_code: int, message: str):
        """Initialize a VLLMError with a status code and message."""
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)
