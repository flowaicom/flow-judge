import json
import os
import tempfile

import pytest
from pydantic import ValidationError

from flow_judge import EvalInput, FlowJudge
from flow_judge.formatting import (
    USER_PROMPT_NO_INPUTS_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    format_rubric,
    format_user_prompt,
    format_vars,
)
from flow_judge.metrics import RESPONSE_CORRECTNESS_BINARY, CustomMetric, RubricItem
from flow_judge.models.models import BaseFlowJudgeModel
from flow_judge.parsing import EvalOutput


class MockFlowJudgeModel(BaseFlowJudgeModel):
    """Mock model for testing."""

    def __init__(self, model_id: str, model_type: str, generation_params: dict):
        """Initialize the mock model."""
        super().__init__(model_id, model_type, generation_params)
        self.generate_return_value = (
            "<feedback>This is a test feedback.</feedback>\n<score>2</score>"
        )
        self.batch_generate_return_value = [
            "<feedback>This is a test feedback.</feedback>\n<score>2</score>"
        ] * 2

    def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        return self.generate_return_value

    def batch_generate(self, prompts: list[str], use_tqdm: bool = True, **kwargs) -> list[str]:
        """Generate responses for multiple prompts."""
        return self.batch_generate_return_value


@pytest.fixture
def mock_model():
    """Fixture to create a mock model for testing."""
    return MockFlowJudgeModel("test-model", "mock", {"temperature": 0.7})


def test_flow_judge_evaluate(mock_model):
    """Test the evaluate method of FlowJudge."""
    judge = FlowJudge(metric=RESPONSE_CORRECTNESS_BINARY, model=mock_model)

    eval_input = EvalInput(
        inputs=[{"question": "What is the capital of France?"}],
        output="The capital of France is Paris.",
    )

    result = judge.evaluate(eval_input)

    assert result.feedback == "This is a test feedback."
    assert result.score == 2


def test_flow_judge_batch_evaluate(mock_model):
    """Test the batch_evaluate method of FlowJudge."""
    judge = FlowJudge(metric=RESPONSE_CORRECTNESS_BINARY, model=mock_model)

    eval_inputs = [
        EvalInput(
            inputs=[{"question": "What is the capital of France?"}],
            output="The capital of France is Paris.",
        ),
        EvalInput(inputs=[{"question": "What is 2+2?"}], output="2+2 equals 4."),
    ]

    results = judge.batch_evaluate(eval_inputs)

    assert len(results) == 2
    for result in results:
        assert result.feedback == "This is a test feedback."
        assert result.score == 2


def test_custom_metric():
    """Test creating and using a custom metric."""
    custom_metric = CustomMetric(
        name="Test Metric",
        criteria="Test criteria",
        rubric=[
            RubricItem(score=0, description="Bad"),
            RubricItem(score=1, description="Good"),
        ],
    )

    assert custom_metric.name == "Test Metric"
    assert custom_metric.criteria == "Test criteria"
    assert len(custom_metric.rubric) == 2


def test_eval_input_validation():
    """Test EvalInput validation."""
    valid_input = EvalInput(inputs=[{"question": "Test?"}], output="Test output")
    assert valid_input.inputs[0]["question"] == "Test?"
    assert valid_input.output == "Test output"

    with pytest.raises(ValidationError):
        EvalInput(inputs="invalid", output="Test output")


def test_eval_output_parsing():
    """Test EvalOutput parsing."""
    valid_response = "<feedback>Test feedback</feedback>\n<score>1</score>"
    parsed = EvalOutput.parse(valid_response)
    assert parsed.feedback == "Test feedback"
    assert parsed.score == 1

    with pytest.raises(ValueError):
        EvalOutput.parse("Invalid response")


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


def test_format_user_prompt():
    """Test format_user_prompt function."""
    variables = {
        "INPUTS": "Test input",
        "OUTPUT": "Test output",
        "EVALUATION_CRITERIA": "Test criteria",
        "RUBRIC": "Test rubric",
    }

    # Test with inputs
    expected_with_inputs = USER_PROMPT_TEMPLATE.format(**variables)
    formatted_with_inputs = format_user_prompt(variables)
    assert formatted_with_inputs == expected_with_inputs

    # Test without inputs
    variables["INPUTS"] = ""
    expected_without_inputs = USER_PROMPT_NO_INPUTS_TEMPLATE.format(**variables)
    formatted_without_inputs = format_user_prompt(variables)
    assert formatted_without_inputs == expected_without_inputs


@pytest.fixture
def flow_judge_instance(mock_model):
    """Fixture to create a FlowJudge instance for testing."""
    instance = FlowJudge(metric=RESPONSE_CORRECTNESS_BINARY, model=mock_model)
    yield instance
    # Add any additional cleanup for FlowJudge if necessary
    del instance


def test_write_results_to_jsonl_single(flow_judge_instance):
    """Test write_results_to_jsonl with a single EvalInput and EvalOutput."""
    eval_input = EvalInput(
        inputs=[{"question": "What is the capital of France?"}],
        output="The capital of France is Paris.",
    )
    eval_output = EvalOutput(feedback="Good answer.", score=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        flow_judge_instance.write_results_to_jsonl([eval_input], [eval_output], tmpdir)

        # Construct expected metric folder name
        metric_name = "response_correctness_binary"
        metric_folder = os.path.join(tmpdir, metric_name)
        assert os.path.isdir(metric_folder), "Metric folder was not created."

        # List files in metric folder
        files = os.listdir(metric_folder)
        assert len(files) == 2, "Metadata and results files were not created."

        # Identify metadata and results files
        metadata_files = [f for f in files if f.startswith("metadata_")]
        results_files = [f for f in files if f.startswith("results_")]
        assert len(metadata_files) == 1, "Metadata file not found."
        assert len(results_files) == 1, "Results file not found."

        metadata_path = os.path.join(metric_folder, metadata_files[0])
        results_path = os.path.join(metric_folder, results_files[0])

        # Verify metadata content
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.loads(f.readline())
            assert metadata["model_id"] == "test-model", "Model ID mismatch in metadata."
            assert metadata["model_type"] == "mock", "Model type mismatch in metadata."
            assert "timestamp" in metadata, "Timestamp missing in metadata."
            assert "generation_params" in metadata, "Generation parameters missing in metadata."

        # Verify results content
        with open(results_path, encoding="utf-8") as f:
            result = json.loads(f.readline())
            assert result["sample"]["inputs"] == eval_input.inputs, "Inputs mismatch in results."
            assert result["sample"]["output"] == eval_input.output, "Output mismatch in results."
            assert result["feedback"] == eval_output.feedback, "Feedback mismatch in results."
            assert result["score"] == eval_output.score, "Score mismatch in results."

    # Ensure that the temporary directory and its contents are deleted
    assert not os.path.exists(tmpdir), "Temporary directory was not deleted."


def test_write_results_to_jsonl_batch(flow_judge_instance):
    """Test write_results_to_jsonl with multiple EvalInputs and EvalOutputs."""
    eval_inputs = [
        EvalInput(
            inputs=[{"question": "What is the capital of France?"}],
            output="The capital of France is Paris.",
        ),
        EvalInput(inputs=[{"question": "What is 2+2?"}], output="2+2 equals 4."),
    ]
    eval_outputs = [
        EvalOutput(feedback="Good answer.", score=1),
        EvalOutput(feedback="Correct.", score=1),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        flow_judge_instance.write_results_to_jsonl(eval_inputs, eval_outputs, tmpdir)

        # Construct expected metric folder name
        metric_name = "response_correctness_binary"
        metric_folder = os.path.join(tmpdir, metric_name)
        assert os.path.isdir(metric_folder), "Metric folder was not created."

        # List files in metric folder
        files = os.listdir(metric_folder)
        assert len(files) == 2, "Metadata and results files were not created."

        # Identify metadata and results files
        metadata_files = [f for f in files if f.startswith("metadata_")]
        results_files = [f for f in files if f.startswith("results_")]
        assert len(metadata_files) == 1, "Metadata file not found."
        assert len(results_files) == 1, "Results file not found."

        metadata_path = os.path.join(metric_folder, metadata_files[0])
        results_path = os.path.join(metric_folder, results_files[0])

        # Verify metadata content
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.loads(f.readline())
            assert metadata["model_id"] == "test-model", "Model ID mismatch in metadata."
            assert metadata["model_type"] == "mock", "Model type mismatch in metadata."
            assert "timestamp" in metadata, "Timestamp missing in metadata."
            assert "generation_params" in metadata, "Generation parameters missing in metadata."

        # Verify results content
        with open(results_path, encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2, "Number of results does not match number of EvalOutputs."
            for line, eval_input, eval_output in zip(lines, eval_inputs, eval_outputs):
                result = json.loads(line)
                assert (
                    result["sample"]["inputs"] == eval_input.inputs
                ), "Inputs mismatch in results."
                assert (
                    result["sample"]["output"] == eval_input.output
                ), "Output mismatch in results."
                assert result["feedback"] == eval_output.feedback, "Feedback mismatch in results."
                assert result["score"] == eval_output.score, "Score mismatch in results."

    # Ensure that the temporary directory and its contents are deleted
    assert not os.path.exists(tmpdir), "Temporary directory was not deleted."


def test_write_results_to_jsonl_empty(flow_judge_instance):
    """Test write_results_to_jsonl with empty EvalInputs and EvalOutputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        flow_judge_instance.write_results_to_jsonl([], [], tmpdir)

        # Construct expected metric folder name
        metric_name = "response_correctness_binary"
        metric_folder = os.path.join(tmpdir, metric_name)
        assert os.path.isdir(metric_folder), "Metric folder was not created."

        # List files in metric folder
        files = os.listdir(metric_folder)
        assert len(files) == 2, "Metadata and results files were not created."

        # Identify metadata and results files
        metadata_files = [f for f in files if f.startswith("metadata_")]
        results_files = [f for f in files if f.startswith("results_")]
        assert len(metadata_files) == 1, "Metadata file not found."
        assert len(results_files) == 1, "Results file not found."

        metadata_path = os.path.join(metric_folder, metadata_files[0])
        results_path = os.path.join(metric_folder, results_files[0])

        # Verify metadata content
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.loads(f.readline())
            assert metadata["model_id"] == "test-model", "Model ID mismatch in metadata."
            assert metadata["model_type"] == "mock", "Model type mismatch in metadata."
            assert "timestamp" in metadata, "Timestamp missing in metadata."
            assert "generation_params" in metadata, "Generation parameters missing in metadata."

        # Verify results content is empty
        with open(results_path, encoding="utf-8") as f:
            contents = f.read()
            assert contents == "", "Results file is not empty."

    # Ensure that the temporary directory and its contents are deleted
    assert not os.path.exists(tmpdir), "Temporary directory was not deleted."


def test_write_results_to_jsonl_invalid_paths(flow_judge_instance):
    """Test write_results_to_jsonl with invalid output directory paths."""
    eval_input = EvalInput(
        inputs=[{"question": "What is the capital of France?"}],
        output="The capital of France is Paris.",
    )
    eval_output = EvalOutput(feedback="Good answer.", score=1)

    # Attempt to write to a directory without write permissions
    invalid_dir = "/root/invalid_directory"
    with pytest.raises(PermissionError):
        flow_judge_instance.write_results_to_jsonl([eval_input], [eval_output], invalid_dir)
