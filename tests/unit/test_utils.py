import pytest

from flow_judge.eval_data_types import EvalInput
from flow_judge.metrics import CustomMetric, RubricItem
from flow_judge.utils.prompt_formatter import (
    USER_PROMPT_NO_INPUTS_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    format_rubric,
    format_user_prompt,
    format_vars,
)
from flow_judge.utils.validators import validate_eval_input


def test_format_vars():
    """Test the format_vars function."""
    variables = [{"question": "What is 2+2?"}, {"context": "Math basics"}]
    formatted = format_vars(variables)
    expected = "<question>\nWhat is 2+2?\n</question>\n<context>\nMath basics\n</context>"
    assert formatted == expected


def test_format_rubric():
    """Test the format_rubric function."""
    rubric = [RubricItem(score=0, description="Bad"), RubricItem(score=1, description="Good")]
    formatted = format_rubric(rubric)
    expected = "- Score 0: Bad\n- Score 1: Good"
    assert formatted == expected


def test_format_user_prompt():
    """Test the format_user_prompt function."""
    # Test with inputs
    variables_with_inputs = {
        "INPUTS": "Test input",
        "OUTPUT": "Test output",
        "EVALUATION_CRITERIA": "Test criteria",
        "RUBRIC": "Test rubric",
    }
    formatted_with_inputs = format_user_prompt(variables_with_inputs)
    assert USER_PROMPT_TEMPLATE.format(**variables_with_inputs) == formatted_with_inputs

    # Test without inputs
    variables_without_inputs = {
        "INPUTS": "",
        "OUTPUT": "Test output",
        "EVALUATION_CRITERIA": "Test criteria",
        "RUBRIC": "Test rubric",
    }
    formatted_without_inputs = format_user_prompt(variables_without_inputs)
    assert (
        USER_PROMPT_NO_INPUTS_TEMPLATE.format(**variables_without_inputs)
        == formatted_without_inputs
    )


def test_validate_eval_input():
    """Test the validate_eval_input function."""
    metric = CustomMetric(
        name="Test Metric",
        criteria="Test criteria",
        rubric=[RubricItem(score=0, description="Bad"), RubricItem(score=1, description="Good")],
        required_inputs=["test_input"],
        required_output="test_output",
    )
    valid_input = EvalInput(
        inputs=[{"test_input": "Test value"}], output={"test_output": "Test output"}
    )
    validate_eval_input(valid_input, metric)  # Should not raise an exception

    invalid_input = EvalInput(
        inputs=[{"wrong_input": "Test value"}], output={"test_output": "Test output"}
    )
    with pytest.raises(ValueError):
        validate_eval_input(invalid_input, metric)
