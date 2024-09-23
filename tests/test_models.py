from unittest.mock import MagicMock, patch

import pytest

from flow_judge.models import FlowJudgeHFModel, FlowJudgeVLLMModel


@pytest.fixture
def mock_hf_model():
    """Fixture to create a mock HF model for testing."""
    with patch("flow_judge.models.huggingface.AutoModelForCausalLM") as mock_model, patch(
        "flow_judge.models.huggingface.AutoTokenizer"
    ) as mock_tokenizer:
        model = FlowJudgeHFModel("test-model", {"temperature": 0.7})
        model.model = mock_model.from_pretrained.return_value
        model.tokenizer = mock_tokenizer.from_pretrained.return_value
        yield model


def test_hf_model_generate(mock_hf_model):
    """Test the generate method of the HF model."""
    mock_hf_model.tokenizer.apply_chat_template.return_value = "Test prompt"
    mock_hf_model.model.generate.return_value = [MagicMock()]
    mock_hf_model.tokenizer.decode.return_value = "Generated text"

    result = mock_hf_model.generate("Test input")
    assert result == "Generated text"


@pytest.fixture
def mock_vllm_model():
    """Fixture to create a mock VLLM model for testing."""
    with patch("flow_judge.models.vllm.LLM") as mock_llm, patch(
        "flow_judge.models.vllm.AutoTokenizer"
    ) as mock_tokenizer:
        model = FlowJudgeVLLMModel("test-model", {"temperature": 0.7})
        model.model = mock_llm.return_value
        model.tokenizer = mock_tokenizer.from_pretrained.return_value
        yield model


def test_vllm_model_generate(mock_vllm_model):
    """Test the generate method of the VLLM model."""
    mock_output = MagicMock()
    mock_output.outputs[0].text = "Generated text"
    mock_vllm_model.model.chat.return_value = [mock_output]

    result = mock_vllm_model.generate("Test input")
    assert result == "Generated text"
