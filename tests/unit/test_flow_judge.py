import shutil
from unittest.mock import patch

import pytest

from flow_judge.eval_data_types import EvalInput, EvalOutput
from flow_judge.flow_judge import FlowJudge
from flow_judge.metrics import RESPONSE_CORRECTNESS_BINARY, CustomMetric, RubricItem
from flow_judge.models.common import BaseFlowJudgeModel
from flow_judge.utils.prompt_formatter import USER_PROMPT_TEMPLATE, format_rubric, format_vars


class MockFlowJudgeModel(BaseFlowJudgeModel):
    """Mock model for testing."""

    def __init__(self, model_id, model_type, generation_params):
        """Initialize the mock model."""
        super().__init__(model_id, model_type, generation_params)

    def _generate(self, prompt):
        """Generate a mock response."""
        return "<feedback>Test feedback</feedback>\n<score>1</score>"

    def _batch_generate(self, prompts, use_tqdm=True):
        """Generate mock responses for a list of prompts."""
        return ["<feedback>Test feedback</feedback>\n<score>1</score>" for _ in prompts]

    def generate(self, prompt):
        """Generate a mock response."""
        return self._generate(prompt)

    def batch_generate(self, prompts, use_tqdm=True):
        """Generate mock responses for a list of prompts."""
        return self._batch_generate(prompts, use_tqdm=use_tqdm)


@pytest.fixture
def mock_model():
    """Fixture to create a mock model for testing."""
    return MockFlowJudgeModel("test-model", "mock", {"temperature": 0.7})


def test_flow_judge_initialization(mock_model):
    """Test the initialization of FlowJudge."""
    judge = FlowJudge(metric=RESPONSE_CORRECTNESS_BINARY, model=mock_model)
    assert isinstance(judge, FlowJudge)
    assert judge.metric == RESPONSE_CORRECTNESS_BINARY
    assert judge.model == mock_model


def test_flow_judge_initialization_invalid_metric():
    """Test FlowJudge initialization with invalid metric."""
    with pytest.raises(ValueError):
        FlowJudge(metric="invalid_metric", model=mock_model)


def test_flow_judge_evaluate(mock_model):
    """Test the evaluate method of FlowJudge."""
    judge = FlowJudge(metric=RESPONSE_CORRECTNESS_BINARY, model=mock_model)
    eval_input = EvalInput(
        inputs=[{"query": "Test query"}, {"reference_answer": "Test reference"}],
        output={"response": "Test response"},
    )
    result = judge.evaluate(eval_input)
    assert isinstance(result, EvalOutput)
    assert result.feedback == "Test feedback"
    assert result.score == 1


def test_flow_judge_batch_evaluate(mock_model):
    """Test the batch_evaluate method of FlowJudge."""
    judge = FlowJudge(metric=RESPONSE_CORRECTNESS_BINARY, model=mock_model)
    eval_inputs = [
        EvalInput(
            inputs=[{"query": "Test query 1"}, {"reference_answer": "Test reference 1"}],
            output={"response": "Test response 1"},
        ),
        EvalInput(
            inputs=[{"query": "Test query 2"}, {"reference_answer": "Test reference 2"}],
            output={"response": "Test response 2"},
        ),
    ]
    results = judge.batch_evaluate(eval_inputs, save_results=False)
    assert len(results) == 2
    for result in results:
        assert isinstance(result, EvalOutput)
        assert result.feedback == "Test feedback"
        assert result.score == 1


@pytest.mark.parametrize("save_results", [True, False])
def test_flow_judge_evaluate_save_results(mock_model, tmp_path, save_results):
    """Test saving results in the evaluate method."""
    judge = FlowJudge(
        metric=RESPONSE_CORRECTNESS_BINARY, model=mock_model, output_dir=str(tmp_path)
    )
    eval_input = EvalInput(
        inputs=[{"query": "Test query"}, {"reference_answer": "Test reference"}],
        output={"response": "Test response"},
    )
    with patch("flow_judge.flow_judge.write_results_to_disk") as mock_write:
        judge.evaluate(eval_input, save_results=save_results)
        if save_results:
            mock_write.assert_called_once()
        else:
            mock_write.assert_not_called()


def test_custom_metric():
    """Test creating and using a custom metric."""
    custom_metric = CustomMetric(
        name="custom_metric",
        criteria="Custom criteria",
        rubric=[RubricItem(score=0, description="Bad"), RubricItem(score=1, description="Good")],
        required_inputs=["custom_input"],
        required_output="custom_output",
    )
    assert custom_metric.name == "custom_metric"
    assert custom_metric.criteria == "Custom criteria"
    assert len(custom_metric.rubric) == 2
    assert custom_metric.required_inputs == ["custom_input"]
    assert custom_metric.required_output == "custom_output"


def test_eval_input_validation(mock_model):
    """Test EvalInput validation."""
    judge = FlowJudge(metric=RESPONSE_CORRECTNESS_BINARY, model=mock_model)

    # Valid input
    valid_input = EvalInput(
        inputs=[{"query": "Test query"}, {"reference_answer": "Test reference"}],
        output={"response": "Test response"},
    )
    assert judge.evaluate(valid_input)

    # Invalid input - missing required input
    invalid_input = EvalInput(
        inputs=[{"query": "Test query"}], output={"response": "Test response"}
    )
    with pytest.raises(ValueError):
        judge.evaluate(invalid_input)

    # Invalid input - wrong output key
    invalid_output = EvalInput(
        inputs=[{"query": "Test query"}, {"reference_answer": "Test reference"}],
        output={"wrong_key": "Test response"},
    )
    with pytest.raises(ValueError):
        judge.evaluate(invalid_output)


def test_format_vars():
    """Test format_vars function."""
    variables = [{"question": "What is 2+2?"}, {"context": "Math basics"}]
    formatted = format_vars(variables)
    expected = """<question>
What is 2+2?
</question>
<context>
Math basics
</context>"""
    assert expected == formatted


def test_format_rubric():
    """Test format_rubric function."""
    rubric = [RubricItem(score=1, description="Good"), RubricItem(score=0, description="Poor")]
    formatted = format_rubric(rubric)
    expected = """- Score 0: Poor
- Score 1: Good"""
    assert expected == formatted


def test_format_prompt(mock_model):
    """Test FlowJudge._format_prompt."""
    eval_input = EvalInput(
        inputs=[{"query": "Test query"}, {"reference_answer": "Test reference"}],
        output={"response": "Test response"},
    )

    judge = FlowJudge(metric=RESPONSE_CORRECTNESS_BINARY, model=mock_model)
    prompt = judge._format_prompt(eval_input)

    expected_prompt = USER_PROMPT_TEMPLATE.format(
        INPUTS=format_vars(eval_input.inputs),
        OUTPUT=format_vars([eval_input.output]),
        EVALUATION_CRITERIA=RESPONSE_CORRECTNESS_BINARY.criteria,
        RUBRIC=format_rubric(RESPONSE_CORRECTNESS_BINARY.rubric),
    )
    assert prompt == expected_prompt


def test_eval_output_parse_fail_on_parse_error():
    """Test EvalOutput.parse with fail_on_parse_error."""
    # Invalid response without proper tags
    invalid_response = "This is an invalid response without proper tags"

    # Test with fail_on_parse_error=False (default behavior)
    result = EvalOutput.parse(invalid_response)
    assert isinstance(result, EvalOutput)
    assert result.feedback == "Error"
    assert result.score == -1

    # Test with fail_on_parse_error=True
    with pytest.raises(ValueError):
        EvalOutput.parse(invalid_response, fail_on_parse_error=True)

    # Test with valid response
    valid_response = "<feedback>Good job!</feedback><score>5</score>"
    result = EvalOutput.parse(valid_response)
    assert isinstance(result, EvalOutput)
    assert result.feedback == "Good job!"
    assert result.score == 5


@pytest.fixture(autouse=True)
def cleanup(request, tmp_path):
    """Cleanup files and directories created during the test."""
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)
