import asyncio
import os
import warnings
from typing import Any

from flow_judge.models.common import (
    AsyncBaseFlowJudgeModel,
    BaseFlowJudgeModel,
    ModelConfig,
    ModelType,
    VllmGenerationParams,
)

try:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class VllmConfig(ModelConfig):
    """Configuration class for vLLM models.

    This class extends ModelConfig to provide specific configuration options for vLLM models,
    including both synchronous and asynchronous execution modes.
    """

    _DEFAULT_MODEL_ID = "flowaicom/Flow-Judge-v0.1"

    def __init__(
        self,
        generation_params: VllmGenerationParams,
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
        """Initialize the VllmConfig.

        For a full list of parameters and their descriptions, see the Vllm class docstring.
        """
        model_id = kwargs.pop("_model_id", self._DEFAULT_MODEL_ID)
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
    """Combined FlowJudge model class for vLLM supporting both sync and async operations.

    This class provides an interface to use vLLM for text generation. It supports both
    synchronous and asynchronous operations, quantization, and various configuration options.

    For a comprehensive list of all available engine arguments and their descriptions,
    please refer to the official vLLM documentation:
    @https://docs.vllm.ai/en/stable/models/engine_args.html
    """

    _DEFAULT_MODEL_ID = "flowaicom/Flow-Judge-v0.1"

    def __init__(
        self,
        generation_params: dict[str, Any] | None = None,
        quantized: bool = True,
        exec_async: bool = False,
        **kwargs: Any,
    ):
        """Initialize the FlowJudge vLLM model.

        :param generation_params: Dictionary of parameters for text generation. Can include:
            - temperature: float (default: 0.1)
            - top_p: float (default: 0.95)
            - max_tokens: int (default: 1000)
            - stop_token_ids: list[int] (default: [32007, 32001, 32000])
        :param quantized: Whether to use quantization (default: True).
        :param exec_async: Whether to execute asynchronously (default: False).
        :param kwargs: Additional keyword arguments for engine configuration. Can include:
            - max_model_len: int (default: 8192)
            - trust_remote_code: bool (default: True)
            - enforce_eager: bool (default: True)
            - dtype: str (default: "bfloat16")
            - disable_sliding_window: bool (default: True)
            - gpu_memory_utilization: float (default: 0.90)
            - max_num_seqs: int (default: 256)
            - tensor_parallel_size: int (default: 1)
            - swap_space: int (default: 4)
            - max_num_batched_tokens: int (default: 2048)
            - max_paddings: int (default: 256)
            - disable_log_requests: bool (default: False)
            - revision: str (default: None)
            - tokenizer: str (default: None)
            - tokenizer_mode: str (default: "auto")
            - download_dir: str (default: None)
            - load_format: str (default: "auto")
            - seed: int (default: 0)
            - block_size: int (default: 16)
            - enable_prefix_caching: bool (default: False)
            - use_v2_block_manager: bool (default: False)
            - enable_lora: bool (default: False)
            - max_loras: int (default: 1)
            - max_lora_rank: int (default: 16)
            - lora_extra_vocab_size: int (default: 256)
            - lora_dtype: str (default: "auto")
            - max_cpu_loras: int (default: None)
            - fully_sharded_loras: bool (default: False)
            - device: str (default: "auto")
            - otlp_traces_endpoint: str (default: None)
            - collect_detailed_traces: str (default: None)

        :raises VllmError: If the 'vllm' package is not installed or initialization fails.
        """
        if not VLLM_AVAILABLE:
            raise VllmError(
                status_code=1,
                message=(
                    "The 'vllm' package is not installed. "
                    "Please install it by adding 'vllm' to your extras:\n"
                    "pip install flow-judge[vllm]"
                ),
            )

        model_id = kwargs.pop("_model_id", self._DEFAULT_MODEL_ID)

        if model_id != self._DEFAULT_MODEL_ID:
            warnings.warn(
                (
                    f"The model '{model_id}' is not officially supported. "
                    f"This library is designed for the '{self._DEFAULT_MODEL_ID}' model. "
                    "Using other models may lead to unexpected behavior, and we do "
                    "not handle GitHub issues for unsupported models. Proceed with caution."
                ),
                UserWarning,
                stacklevel=2,
            )

        model_id = (
            f"{model_id}-AWQ" if quantized and model_id == self._DEFAULT_MODEL_ID else model_id
        )

        generation_params = VllmGenerationParams(**(generation_params or {}))

        download_dir = kwargs.get("download_dir", None)

        # Translate max_new_tokens to max_tokens for vLLM
        if "max_new_tokens" in generation_params.model_dump():
            generation_params.max_tokens = generation_params.max_new_tokens
            del generation_params.max_new_tokens

        config = VllmConfig(
            generation_params=generation_params,
            quantization=quantized,
            exec_async=exec_async,
            _model_id=model_id,
            **kwargs,
        )

        super().__init__(model_id, "vllm", config.generation_params, **kwargs)

        self.exec_async = exec_async
        self.generation_params = config.generation_params

        try:
            if not torch.cuda.is_available():
                raise VllmError(
                    status_code=2,
                    message="GPU is not available. vLLM requires a GPU to run."
                    " Check https://docs.vllm.ai/en/latest/getting_started/installation.html"
                    " for installation requirements.",
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

            if download_dir:
                os.environ["HF_HOME"] = download_dir
                engine_args["download_dir"] = download_dir

            if exec_async:
                engine_args["disable_log_requests"] = kwargs.get("disable_log_requests", False)
                self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
            else:
                self.model = LLM(**engine_args)

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            raise VllmError(
                status_code=3,
                message=f"An unexpected error occurred while initializing vLLM: {str(e)}",
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
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
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
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
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

    async def aclose(self):
        """Asynchronous cleanup method."""
        self.shutdown()
        await asyncio.sleep(0.1)  # Give a moment for background tasks to clean up


class VllmError(Exception):
    """Custom exception for Vllm-related errors."""

    def __init__(self, status_code: int, message: str):
        """Initialize a VllmError with a status code and message."""
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)
