import logging
import os
from typing import Any

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from flow_judge.models.base import BaseFlowJudgeModel
from flow_judge.utils.prompt_formatter import format_user_prompt

logger = logging.getLogger(__name__)


class FlowJudgeHFModel(BaseFlowJudgeModel):
    """FlowJudge model class for Hugging Face Transformers."""

    def __init__(self, model_id: str, generation_params: dict[str, Any], **hf_kwargs: Any):
        """Initialize the FlowJudge Hugging Face Transformers model."""
        super().__init__(model_id, "transformers", generation_params, **hf_kwargs)

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # Download the entire repository
        logger.info(
            "Downloading the model from Hugging Face Hub using hf-transfer for faster downloads..."
        )
        snapshot_download(repo_id=model_id)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **hf_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **hf_kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_params = generation_params

        if self.device == "cpu":
            logger.warning(
                "Running the FlowJudgeHFModel on CPU may result in longer inference times."
            )

        self.batch_size = self._determine_batch_size()

    def _determine_batch_size(self) -> int:
        """Determine an appropriate batch size based on available GPU memory."""
        if self.device == "cpu":
            return 1  # Default to 1 for CPU

        # Start with a batch size of 1 and double it until we run out of memory
        batch_size = 1
        max_length = self.generation_params.get("max_model_len", 8192)

        # Create a sample prompt using the format_user_prompt function
        sample_prompt_variables = {
            "INPUTS": "Sample input for batch size determination" * 10,
            "OUTPUT": "Sample output for batch size determination" * 5,
            "EVALUATION_CRITERIA": "Sample evaluation criteria" * 3,
            "RUBRIC": "Sample rubric" * 5,
        }
        sample_input = format_user_prompt(sample_prompt_variables)

        while True:
            try:
                # Prepare a batch of inputs
                inputs = [sample_input] * batch_size
                encoded_inputs = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)

                # Perform a forward pass
                with torch.no_grad():
                    _ = self.model(**encoded_inputs)

                # If successful, double the batch size and try again
                batch_size *= 2
                torch.cuda.empty_cache()  # Clear GPU memory
                del encoded_inputs  # Remove references to tensors
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # We've found the limit, so use the last successful batch size
                    optimal_batch_size = max(1, batch_size // 2)
                    logger.info(f"Automatically determined batch size: {optimal_batch_size}")
                    torch.cuda.empty_cache()  # Final cleanup
                    return optimal_batch_size
                else:
                    # If it's not an OOM error, re-raise the exception
                    raise

    def generate(self, prompt: str) -> str:
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

    def batch_generate(self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any) -> list[str]:
        """Generate responses for multiple prompts using batching."""
        all_results = []

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

            # Generate outputs
            outputs = self.model.generate(**inputs, **self.generation_params)

            # Decode outputs
            batch_results = []
            for i, output in enumerate(outputs):
                input_length = len(self.tokenizer.encode(chat_prompts[i], add_special_tokens=False))
                result = self.tokenizer.decode(output[input_length:], skip_special_tokens=True)
                batch_results.append(result.strip())

            all_results.extend(batch_results)

        return all_results
