from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


class BaseFlowJudgeModel(ABC):
    """Base class for all FlowJudge models."""

    def __init__(self, model_id: str, model_type: str, generation_params: dict, **kwargs):
        """Initialize the base FlowJudge model."""
        self.metadata = {
            "model_id": model_id,
            "model_type": model_type,
            "generation_params": generation_params,
            "kwargs": kwargs,
        }

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        pass

    @abstractmethod
    def batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: int | float | str
    ) -> list[str]:
        """Generate responses for multiple prompts."""
        pass


class FlowJudgeHFModel(BaseFlowJudgeModel):
    """FlowJudge model class for Hugging Face Transformers."""

    def __init__(self, model_id: str, generation_params: dict, **hf_kwargs):
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

    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
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


class FlowJudgeVLLMModel(BaseFlowJudgeModel):
    """FlowJudge model class for vLLM."""

    def __init__(self, model: str, generation_params: dict, **vllm_kwargs):
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

    def batch_generate(self, prompts: list[str], use_tqdm: bool = True, **kwargs) -> list[str]:
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


class VLLMError(Exception):
    """Custom exception for VLLM-related errors."""

    def __init__(self, status_code: int, message: str):
        """Initialize a VLLMError with a status code and message."""
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)
