import logging
import os
from typing import Any, Dict, Optional

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from flow_judge.models.common import BaseFlowJudgeModel, ModelConfig, ModelType

logger = logging.getLogger(__name__)

class HfConfig(ModelConfig):
    def __init__(
        self,
        model_id: str,
        generation_params: Dict[str, Any],
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        attn_implementation: Optional[str] = None,
        flash_attn: bool = True,
        **kwargs: Any
    ):
        super().__init__(model_id, ModelType.TRANSFORMERS, generation_params, **kwargs)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation if flash_attn else None
        self.flash_attn = flash_attn

class Hf(BaseFlowJudgeModel):
    """FlowJudge model class for Hugging Face Transformers."""

    def __init__(self, model_id: str = None, generation_params: dict[str, Any] = None, flash_attn: bool = True, **kwargs: Any):
        """Initialize the FlowJudge Hugging Face Transformers model."""
        default_model_id = "flowaicom/Flow-Judge-v0.1"
        model_id = model_id or default_model_id

        default_generation_params = {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_new_tokens": 1000,
            "do_sample": True,
        }
        generation_params = generation_params or default_generation_params

        config = HfConfig(
            model_id=model_id,
            generation_params=generation_params,
            flash_attn=flash_attn,
            attn_implementation="flash_attention_2" if flash_attn else None,
            **kwargs
        )

        super().__init__(model_id, "transformers", generation_params, **kwargs)

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # Download the entire repository
        logger.info(
            "Downloading the model from Hugging Face Hub using hf-transfer for faster downloads..."
        )
        snapshot_download(repo_id=model_id)

        model_kwargs = {
            "device_map": config.device_map,
            "torch_dtype": getattr(torch, config.torch_dtype),
        }
        if config.attn_implementation:
            model_kwargs["attn_implementation"] = config.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_params = generation_params

        if self.device == "cpu":
            logger.warning(
                "Running the FlowJudgeHFModel on CPU may result in longer inference times."
            )

        self.batch_size = 1  # Default to 1, will be updated in batch_generate

    def _determine_batch_size(self, prompts: list[str]) -> int:
        """Determine an appropriate batch size based on available GPU memory and eval_inputs."""
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

    def _generate(self, prompt: str) -> str:
        """Generate a response using the FlowJudge Hugging Face Transformers model."""
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        outputs = self.model.generate(**inputs, **self.generation_params)

        # Decode only the generated part (excluding the input)
        generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return generated_text.strip()

    def _batch_generate(self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any) -> list[str]:
        """Generate responses for multiple prompts using batching."""
        all_results = []

        # Determine optimal batch size using the current prompts
        self.batch_size = self._determine_batch_size(prompts)

        # Create batches using the automatically determined batch size
        batches = [
            prompts[i : i + self.batch_size] for i in range(0, len(prompts), self.batch_size)
        ]

        # Process each batch
        for batch in tqdm(batches, disable=not use_tqdm, desc="Processing batches"):
            # Apply chat template to all prompts in the batch
            chat_prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch
            ]

            # Tokenize all prompts in the batch
            inputs = self.tokenizer(
                chat_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            input_tok_lens = [len(input) for input in inputs["input_ids"]]
            # Generate outputs
            outputs = self.model.generate(**inputs, **self.generation_params)

            # Decode outputs
            batch_results = []
            for output, input_tok_len in zip(outputs, input_tok_lens, strict=True):
                result = self.tokenizer.decode(output[input_tok_len:], skip_special_tokens=False)
                batch_results.append(result.strip())

            all_results.extend(batch_results)

        return all_results
