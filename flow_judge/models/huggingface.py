from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flow_judge.models.base import BaseFlowJudgeModel


class FlowJudgeHFModel(BaseFlowJudgeModel):
    """FlowJudge model class for Hugging Face Transformers."""

    def __init__(self, model_id: str, generation_params: dict[str, Any], **hf_kwargs: Any):
        """Initialize the FlowJudge Hugging Face Transformers model."""
        super().__init__(model_id, "transformers", generation_params, **hf_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **hf_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_params = generation_params

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

    def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate responses for multiple prompts."""
        # Apply chat template to all prompts
        chat_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]

        # Tokenize all prompts
        inputs = self.tokenizer(
            chat_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # Generate outputs
        outputs = self.model.generate(**inputs, **self.generation_params)

        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Remove the input prompts from the generated texts
        results = []
        for i, _ in enumerate(generated_texts):
            input_length = len(self.tokenizer.encode(chat_prompts[i], add_special_tokens=False))
            result = self.tokenizer.decode(outputs[i][input_length:], skip_special_tokens=True)
            results.append(result.strip())

        return results
