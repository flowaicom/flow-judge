import asyncio
from typing import Any, Dict, List

from flow_judge.models.common import (
    AsyncBaseFlowJudgeModel,
    BaseFlowJudgeModel,
    ModelConfig,
    ModelType,
    GenerationParams,
)

import warnings

try:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class VllmConfig(ModelConfig):
    def __init__(
        self,
        model_id: str,
        generation_params: GenerationParams,
        max_model_len: int = 8192,
        trust_remote_code: bool = True,
        enforce_eager: bool = True,
        dtype: str = "bfloat16",
        disable_sliding_window: bool = True,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 256,
        quantization: bool = True,
        exec_async: bool = False,
        **kwargs: Any,
    ):
        model_type = ModelType.VLLM_ASYNC if exec_async else ModelType.VLLM
        super().__init__(model_id, model_type, generation_params.model_dump(), **kwargs)
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager
        self.dtype = dtype
        self.disable_sliding_window = disable_sliding_window
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.quantization = quantization
        self.exec_async = exec_async


class Vllm(BaseFlowJudgeModel, AsyncBaseFlowJudgeModel):
    """Combined FlowJudge model class for vLLM supporting both sync and async operations."""

    def __init__(
        self,
        model: str = None,
        generation_params: Dict[str, Any] = None,
        quantized: bool = True,
        exec_async: bool = False,
        **kwargs: Any,
    ):
        """Initialize the FlowJudge vLLM model."""
        if not VLLM_AVAILABLE:
            raise VllmError(
                status_code=1,
                message="The 'vllm' package is not installed. Please install it by adding 'vllm' to your extras:\n"
                "pip install flow-judge[...,vllm]",
            )

        default_model_id = "flowaicom/Flow-Judge-v0.1"

        if model is not None and model != default_model_id:
            warnings.warn(
                f"The model '{model}' is not officially supported. "
                f"This library is designed for the '{default_model_id}' model. "
                "Using other models may lead to unexpected behavior, and we do not handle "
                "GitHub issues for unsupported models. Proceed with caution.",
                UserWarning
            )

        model = model or default_model_id
        # Only append "-AWQ" if it's the default model and quantization is enabled
        model_id = f"{model}-AWQ" if quantized and model == default_model_id else model

        generation_params = GenerationParams(**(generation_params or {}))

        config = VllmConfig(model_id=model_id, generation_params=generation_params, quantization=quantized, exec_async=exec_async, **kwargs)

        super().__init__(model, "vllm", config.generation_params, **kwargs)

        self.exec_async = exec_async
        self.generation_params = config.generation_params

        try:
            if not torch.cuda.is_available():
                raise VllmError(
                    status_code=2,
                    message="GPU is not available. vLLM requires a GPU to run. Check https://docs.vllm.ai/en/latest/getting_started/installation.html for installation requirements.",
                )

            engine_args = {
                "model": model_id,
                "max_model_len": config.max_model_len,
                "trust_remote_code": config.trust_remote_code,
                "enforce_eager": config.enforce_eager,
                "dtype": config.dtype,
                "disable_sliding_window": config.disable_sliding_window,
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "max_num_seqs": config.max_num_seqs,
                "quantization": "awq_marlin" if config.quantization else None,
                **kwargs,
            }

            if exec_async:
                engine_args["disable_log_requests"] = kwargs.get("disable_log_requests", False)
                self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
            else:
                self.model = LLM(**engine_args)

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except ImportError as e:
            raise VllmError(
                status_code=1,
                message="Failed to import 'vllm' package. Make sure it is installed correctly.",
            ) from e

    def _generate(self, prompt: str) -> str:
        """Generate a response using the FlowJudge vLLM model."""
        if self.exec_async:
            return asyncio.run(self._async_generate(prompt))

        conversation = [{"role": "user", "content": prompt.strip()}]
        params = SamplingParams(**self.generation_params)
        outputs = self.model.chat(conversation, sampling_params=params)
        return outputs[0].outputs[0].text.strip()

    def _batch_generate(
        self, prompts: List[str], use_tqdm: bool = True, **kwargs: Any
    ) -> List[str]:
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

        return ""

    async def _async_batch_generate(
        self, prompts: List[str], use_tqdm: bool = True, **kwargs: Any
    ) -> List[str]:
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


class VllmError(Exception):
    """Custom exception for Vllm-related errors."""

    def __init__(self, status_code: int, message: str):
        """Initialize a VllmError with a status code and message."""
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)
