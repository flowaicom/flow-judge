import logging
import os
import warnings
from typing import Any

from flow_judge.models.common import BaseFlowJudgeModel, GenerationParams, ModelConfig, ModelType

try:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from tqdm import tqdm

logger = logging.getLogger(__name__)


class HfConfig(ModelConfig):
    """Configuration class for Hugging Face models."""

    _DEFAULT_MODEL_ID = "flowaicom/Flow-Judge-v0.1"

    def __init__(
        self,
        generation_params: GenerationParams,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        flash_attn: bool = True,
        **kwargs: Any,
    ):
        """Initialize HfConfig with model details and Hugging Face specific parameters.

        :param generation_params: Parameters for text generation.
        :param device_map: Device mapping strategy.
        :param torch_dtype: PyTorch data type for the model.
        :param flash_attn: Whether to use flash attention.
        :param kwargs: Additional keyword arguments.
        """
        model_id = kwargs.pop("_model_id", self._DEFAULT_MODEL_ID)
        super().__init__(model_id, ModelType.TRANSFORMERS, generation_params.model_dump(), **kwargs)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.flash_attn = flash_attn
        self.kwargs = kwargs


class Hf(BaseFlowJudgeModel):
    """FlowJudge model class for Hugging Face Transformers."""

    _DEFAULT_MODEL_ID = "flowaicom/Flow-Judge-v0.1"

    def __init__(
        self,
        generation_params: dict[str, Any] | None = None,
        flash_attn: bool = True,
        **kwargs: Any,
    ):
        """Initialize the FlowJudge Hugging Face Transformers model.

        :param generation_params: Dictionary of parameters for text generation.
        :param flash_attn: Whether to use flash attention.
        :param kwargs: Additional keyword arguments, including:
            - _model_id: Identifier for the model. If None, uses the default model.
            // ... other kwargs ...
        """
        if not HF_AVAILABLE:
            raise HfError(
                status_code=1,
                message="The required Hugging Face packages are not installed. "
                "Please install them by adding 'hf' to your extras:\n"
                "pip install flow-judge[hf]",
            )

        model_id = kwargs.pop("_model_id", self._DEFAULT_MODEL_ID)

        if model_id != self._DEFAULT_MODEL_ID:
            warnings.warn(
                f"The model '{model_id}' is not officially supported. "
                f"This library is designed for the '{self._DEFAULT_MODEL_ID}' model. "
                "Using other models may lead to unexpected behavior, and we do not handle "
                "GitHub issues for unsupported models. Proceed with caution.",
                UserWarning,
                stacklevel=2,
            )

        generation_params = GenerationParams(**(generation_params or {}))

        config = HfConfig(
            generation_params=generation_params, flash_attn=flash_attn, _model_id=model_id, **kwargs
        )

        super().__init__(model_id, "transformers", config.generation_params, **kwargs)

        try:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

            logger.info(
                "Downloading the model from Hugging Face Hub using hf-transfer"
                "for faster downloads...",
            )
            snapshot_download(repo_id=model_id)

            model_kwargs = {
                "device_map": config.device_map,
                "torch_dtype": getattr(torch, config.torch_dtype),
            }
            if config.flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            # Include any additional kwargs that might be relevant for model initialization
            for key, value in config.kwargs.items():
                if (
                    key not in model_kwargs
                    and key in AutoModelForCausalLM.from_pretrained.__code__.co_varnames
                ):
                    model_kwargs[key] = value

            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.generation_params = generation_params.model_dump()
            self.config = config

            if self.device == "cpu":
                logger.warning("Running Hf on CPU may result in longer inference times.")

            self.batch_size = 1  # Default to 1, will be updated in batch_generate

        except Exception as e:
            raise HfError(
                status_code=2,
                message=f"An error occurred while initializing the Hugging Face model: {str(e)}\n"
                "Please make sure you have installed all required dependencies by adding 'hf' "
                "to your extras:\npip install flow-judge[hf]",
            ) from e

    def _determine_batch_size(self, prompts: list[str]) -> int:
        """Determine an appropriate batch size based on available GPU memory and eval_inputs.

        This method attempts to find the largest batch size that can be processed without
        running out of GPU memory.

        :param prompts: List of input prompts to be processed.
        :return: The determined optimal batch size.
        """
        if self.device == "cpu":
            return 1  # Default to 1 for CPU

        batch_size = 1
        max_length = self.generation_params.get("max_model_len", 8192)
        max_new_tokens = self.generation_params.get("max_new_tokens", 1024)

        while True:
            try:
                # Prepare a batch of inputs using the longest input
                longest_input = max(prompts, key=lambda x: len(self.tokenizer.encode(x)))
                # Check if the longest input exceeds max_length
                input_length = len(self.tokenizer.encode(longest_input))
                if input_length > max_length:
                    logger.warning(
                        f"Input length {input_length} exceeds max_length {max_length}. "
                        f"Truncated inputs can result in suboptimal performance."
                    )

                inputs = [longest_input] * batch_size
                encoded_inputs = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)

                # Simulate generation
                with torch.no_grad():
                    _ = self.model.generate(
                        **encoded_inputs, max_new_tokens=max_new_tokens, do_sample=False
                    )

                # If successful, double the batch size and try again
                batch_size *= 2
                torch.cuda.empty_cache()
                del encoded_inputs
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    optimal_batch_size = max(1, batch_size // 2)
                    logger.info(f"Automatically determined batch size: {optimal_batch_size}")
                    torch.cuda.empty_cache()
                    return optimal_batch_size
                else:
                    raise

    def _prepare_generation_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Combines generation params, passed kwargs, and relevant config kwargs.

        :param kwargs: Additional keyword arguments for generation.
        :return: A dictionary of prepared generation kwargs.
        """
        generation_kwargs = {**self.generation_params, **kwargs}
        for key, value in self.config.kwargs.items():
            if key not in generation_kwargs and key in self.model.generate.__code__.co_varnames:
                generation_kwargs[key] = value
        return generation_kwargs

    def _generate(self, prompt: str) -> str:
        """Generate a response using the FlowJudge Hugging Face Transformers model."""
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        generation_kwargs = self._prepare_generation_kwargs()
        outputs = self.model.generate(**inputs, **generation_kwargs)

        generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return generated_text.strip()

    def _batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        """Generate responses for multiple prompts using batching."""
        all_results = []

        self.batch_size = self._determine_batch_size(prompts)

        batches = [
            prompts[i : i + self.batch_size] for i in range(0, len(prompts), self.batch_size)
        ]

        generation_kwargs = self._prepare_generation_kwargs(**kwargs)

        for batch in tqdm(batches, disable=not use_tqdm, desc="Processing batches"):
            chat_prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch
            ]

            inputs = self.tokenizer(
                chat_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            input_tok_lens = [len(input) for input in inputs["input_ids"]]

            outputs = self.model.generate(**inputs, **generation_kwargs)

            batch_results = []
            for output, input_tok_len in zip(outputs, input_tok_lens, strict=True):
                result = self.tokenizer.decode(output[input_tok_len:], skip_special_tokens=False)
                batch_results.append(result.strip())

            all_results.extend(batch_results)

        return all_results


class HfError(Exception):
    """Custom exception for Hugging Face-related errors."""

    def __init__(self, status_code: int, message: str):
        """Initialize an HfError with a status code and message."""
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)
