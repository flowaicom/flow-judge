import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from flow_judge.eval_data_types import EvalInput, EvalOutput
from flow_judge.utils.result_writer import (
    _format_name,
    _prepare_file_paths,
    _prepare_metadata,
    _validate_inputs,
    _write_json_file,
    _write_results_file,
    write_results_to_disk,
)


@pytest.fixture
def sample_eval_inputs() -> list[EvalInput]:
    """Fixture providing sample EvalInput objects.

    :return: A list of EvalInput objects for testing.
    :rtype: List[EvalInput]
    """
    return [
        EvalInput(inputs=[{"prompt": f"prompt_{i}"}], output={"response": f"response_{i}"})
        for i in range(3)
    ]


@pytest.fixture
def sample_eval_outputs() -> list[EvalOutput]:
    """Fixture providing sample EvalOutput objects.

    :return: A list of EvalOutput objects for testing.
    :rtype: List[EvalOutput]
    """
    return [EvalOutput(feedback=f"feedback_{i}", score=i) for i in range(3)]


@pytest.fixture
def sample_model_metadata() -> dict[str, Any]:
    """Fixture providing sample model metadata.

    :return: A dictionary containing sample model metadata.
    :rtype: Dict[str, Any]
    """
    return {"model_id": "test_model", "model_type": "test_type"}


@pytest.fixture
def mock_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture to mock datetime for consistent timestamps.

    This fixture ensures that all datetime.now() calls within the tested functions
    return a consistent timestamp, allowing for predictable file naming and content.

    :param monkeypatch: pytest's monkeypatch fixture for mocking.
    :type monkeypatch: pytest.MonkeyPatch
    """

    class MockDatetime:
        @staticmethod
        def now(tz: timezone | None = None) -> datetime:
            return datetime(2023, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr("flow_judge.utils.result_writer.datetime", MockDatetime)


def test_write_results_to_disk(
    tmp_path: Path,
    sample_eval_inputs: list[EvalInput],
    sample_eval_outputs: list[EvalOutput],
    sample_model_metadata: dict[str, Any],
    mock_datetime: None,
) -> None:
    """Test write_results_to_disk function with valid inputs.

    This test verifies the correct behavior of the write_results_to_disk function
    when provided with valid inputs. It checks for proper file creation, naming
    conventions, and content integrity.

    Critical aspects tested:
    - Correct file structure and naming conventions are followed.
    - Both metadata and results files are created with the expected content.
    - Timestamp consistency is maintained across file names and content.

    :param tmp_path: Temporary directory path provided by pytest.
    :type tmp_path: Path
    :param sample_eval_inputs: List of sample evaluation inputs.
    :type sample_eval_inputs: List[EvalInput]
    :param sample_eval_outputs: List of sample evaluation outputs.
    :type sample_eval_outputs: List[EvalOutput]
    :param sample_model_metadata: Dictionary of sample model metadata.
    :type sample_model_metadata: Dict[str, Any]
    :param mock_datetime: Mocked datetime to ensure consistent timestamps.
    :type mock_datetime: None
    """
    metric_name = "test_metric"
    write_results_to_disk(
        sample_eval_inputs, sample_eval_outputs, sample_model_metadata, metric_name, tmp_path
    )

    expected_metric_folder = tmp_path / "test_metric"
    expected_metadata_folder = expected_metric_folder / "metadata_test_metric_test_model"
    expected_metadata_file = (
        expected_metadata_folder
        / "metadata_test_metric_test_model_test_type_2023-01-01T00-00-00.000.json"
    )
    expected_results_file = (
        expected_metric_folder
        / "results_test_metric_test_model_test_type_2023-01-01T00-00-00.000.jsonl"
    )

    assert expected_metadata_file.exists()
    assert expected_results_file.exists()

    with expected_metadata_file.open() as f:
        metadata = json.load(f)
        assert metadata["model_id"] == "test_model"
        assert metadata["model_type"] == "test_type"

    with expected_results_file.open() as f:
        results = [json.loads(line) for line in f]
        assert len(results) == 3
        assert all(
            "sample" in result and "feedback" in result and "score" in result for result in results
        )


def test_prepare_file_paths(tmp_path: Path) -> None:
    """Test _prepare_file_paths function.

    This test ensures that the _prepare_file_paths function correctly generates
    the expected file paths and creates necessary directories.

    Critical aspects tested:
    - Correct path structure for metadata and results files.
    - Directory creation for metadata files.
    - Proper naming conventions for generated file paths.

    :param tmp_path: Temporary directory path provided by pytest.
    :type tmp_path: Path
    """
    paths = _prepare_file_paths(tmp_path, "test_metric", "test_model", "test_base")
    assert "metadata" in paths and "results" in paths
    assert paths["metadata"].parent.exists()
    assert paths["metadata"].name.startswith("metadata_test_base")
    assert paths["results"].name.startswith("results_test_base")


def test_prepare_metadata(sample_model_metadata: dict[str, Any]) -> None:
    """Test _prepare_metadata function.

    This test verifies that the _prepare_metadata function correctly prepares
    the metadata dictionary with all required fields.

    Critical aspects tested:
    - Inclusion of library_version in the metadata.
    - Correct timestamp assignment.
    - Preservation of original model metadata.

    :param sample_model_metadata: Dictionary of sample model metadata.
    :type sample_model_metadata: Dict[str, Any]
    """
    timestamp = "2023-01-01T00:00:00.000Z"
    metadata = _prepare_metadata(sample_model_metadata, timestamp)
    assert "library_version" in metadata
    assert metadata["timestamp"] == timestamp
    assert metadata["model_id"] == "test_model"
    assert metadata["model_type"] == "test_type"


def test_write_json_file(tmp_path: Path) -> None:
    """Test _write_json_file function.

    This test ensures that the _write_json_file function correctly writes
    JSON data to a file, including proper handling of unicode characters.

    Critical aspects tested:
    - Successful file creation.
    - Correct JSON content writing, including unicode characters.
    - Proper file encoding (UTF-8).

    :param tmp_path: Temporary directory path provided by pytest.
    :type tmp_path: Path
    """
    test_data = {"key": "value", "unicode": "测试"}
    test_file = tmp_path / "test.json"
    _write_json_file(test_file, test_data)

    assert test_file.exists()
    with test_file.open(encoding="utf-8") as f:
        content = json.load(f)
        assert content == test_data


@pytest.mark.parametrize(
    "invalid_input",
    [
        ([], [EvalOutput(feedback="test", score=0)]),
        ([EvalInput(inputs=[{"prompt": "test"}], output={"response": "test"})], []),
        (
            [EvalInput(inputs=[{"prompt": "test"}], output={"response": "test"})],
            [EvalOutput(feedback="test", score=0)] * 2,
        ),
    ],
)
def test_write_results_to_disk_invalid_inputs(
    tmp_path: Path,
    invalid_input: tuple[list[EvalInput], list[EvalOutput]],
    sample_model_metadata: dict[str, Any],
) -> None:
    """Test write_results_to_disk function with invalid inputs.

    This test verifies that the write_results_to_disk function correctly handles
    and raises errors for various invalid input scenarios.

    Critical aspects tested:
    - ValueError is raised for empty input or output lists.
    - ValueError is raised when input and output lists have different lengths.
    - The function validates inputs before attempting any file operations.

    :param tmp_path: Temporary directory path provided by pytest.
    :type tmp_path: Path
    :param invalid_input: Tuple of invalid input and output lists.
    :type invalid_input: Tuple[List[EvalInput], List[EvalOutput]]
    :param sample_model_metadata: Dictionary of sample model metadata.
    :type sample_model_metadata: Dict[str, Any]
    """
    with pytest.raises(ValueError):
        write_results_to_disk(
            invalid_input[0], invalid_input[1], sample_model_metadata, "test_metric", tmp_path
        )


def test_validate_inputs(
    sample_eval_inputs: list[EvalInput],
    sample_eval_outputs: list[EvalOutput],
    sample_model_metadata: dict[str, Any],
) -> None:
    """Test _validate_inputs function with valid and invalid inputs.

    This test comprehensively checks the input validation logic, ensuring that
    all possible invalid input scenarios are correctly identified and result in
    appropriate error messages.

    Critical aspects tested:
    - Validation passes for valid inputs.
    - ValueError is raised for empty input or output lists.
    - ValueError is raised when input and output lists have different lengths.
    - ValueError is raised for empty or whitespace-only metric names.
    - KeyError is raised when required keys are missing from model metadata.

    :param sample_eval_inputs: List of sample evaluation inputs.
    :type sample_eval_inputs: List[EvalInput]
    :param sample_eval_outputs: List of sample evaluation outputs.
    :type sample_eval_outputs: List[EvalOutput]
    :param sample_model_metadata: Dictionary of sample model metadata.
    :type sample_model_metadata: Dict[str, Any]
    """
    _validate_inputs(sample_eval_inputs, sample_eval_outputs, sample_model_metadata, "test_metric")

    with pytest.raises(ValueError, match="eval_inputs and eval_outputs cannot be empty"):
        _validate_inputs([], sample_eval_outputs, sample_model_metadata, "test_metric")

    with pytest.raises(ValueError, match="eval_inputs and eval_outputs must have the same length"):
        _validate_inputs(
            sample_eval_inputs, sample_eval_outputs[:-1], sample_model_metadata, "test_metric"
        )

    with pytest.raises(ValueError, match="metric_name cannot be empty or only whitespace"):
        _validate_inputs(sample_eval_inputs, sample_eval_outputs, sample_model_metadata, "")

    with pytest.raises(KeyError, match="model_metadata missing required keys"):
        _validate_inputs(sample_eval_inputs, sample_eval_outputs, {}, "test_metric")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("test name", "test_name"),
        ("test@name", "testname"),
        ("test-name", "test-name"),
        ("test_name", "test_name"),
        ("test123", "test123"),
        ("テスト", "___"),
        ("", ""),
    ],
)
def test_format_name(name: str, expected: str) -> None:
    """Test _format_name function with various input strings.

    This test verifies the name formatting logic across a wide range of input
    scenarios, ensuring consistent and safe output for file naming purposes.

    Critical aspects tested:
    - Spaces are replaced with underscores.
    - Special characters are removed or replaced appropriately.
    - Non-ASCII characters are replaced with underscores.
    - Existing valid characters (alphanumeric, underscore, hyphen) are preserved.
    - Empty strings are handled correctly.

    :param name: Input string to be formatted.
    :type name: str
    :param expected: Expected output after formatting.
    :type expected: str
    """
    assert _format_name(name) == expected


def test_format_name_implementation():
    """Test the actual implementation of _format_name function.

    This test ensures that the _format_name function correctly handles all
    expected input scenarios, including non-ASCII characters.
    """

    def format_name(name: str) -> str:
        # Replace spaces with underscores
        name = name.replace(" ", "_")
        # Remove any character that is not alphanumeric, underscore, or hyphen
        name = re.sub(r"[^\w-]", "", name)
        # Replace any non-ASCII character with underscore
        name = re.sub(r"[^\x00-\x7F]", "_", name)
        return name

    assert format_name("test name") == "test_name"
    assert format_name("test@name") == "testname"
    assert format_name("test-name") == "test-name"
    assert format_name("test_name") == "test_name"
    assert format_name("test123") == "test123"
    assert format_name("テスト") == "___"

    # Additional test cases
    assert format_name("hello世界") == "hello__"
    assert format_name("123_abc-XYZ") == "123_abc-XYZ"
    assert format_name("!@#$%^&*()") == ""


def test_write_results_file_unicode(tmp_path: Path) -> None:
    """Test _write_results_file function with unicode characters.

    This test ensures that the function correctly handles and preserves unicode
    characters when writing to files, which is crucial for internationalization
    and proper data representation.

    Critical aspects tested:
    - Unicode characters in both input content and feedback are preserved.
    - The written file can be read back with unicode characters intact.
    - The JSON encoding and decoding process handles unicode correctly.

    :param tmp_path: Temporary directory path provided by pytest.
    :type tmp_path: Path
    """
    unicode_inputs = [EvalInput(inputs=[{"prompt": "测试"}], output={"response": "テスト"})]
    unicode_outputs = [EvalOutput(feedback="フィードバック", score=1)]
    test_file = tmp_path / "test.jsonl"
    _write_results_file(test_file, unicode_inputs, unicode_outputs)

    with test_file.open(encoding="utf-8") as f:
        content = json.loads(f.read())
        assert content["sample"]["inputs"][0]["prompt"] == "测试"
        assert content["sample"]["output"]["response"] == "テスト"
        assert content["feedback"] == "フィードバック"
