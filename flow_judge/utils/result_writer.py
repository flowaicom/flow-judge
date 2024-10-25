import json
import logging
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

import flow_judge
from flow_judge.eval_data_types import EvalInput, EvalOutput

logger = logging.getLogger(__name__)


def write_results_to_disk(
    eval_inputs: list[EvalInput],
    eval_outputs: list[EvalOutput],
    model_metadata: dict[str, Any],
    metric_name: str,
    output_dir: str | Path,
    append: bool = False,
) -> None:
    """Write evaluation results, inputs, and metadata to separate JSONL files.

    This function processes evaluation data and writes it to disk in a structured format.
    It creates separate files for metadata and results, organizing them in directories
    based on the metric name and model ID.

    Args:
        eval_inputs: List of evaluation inputs.
        eval_outputs: List of evaluation outputs.
        model_metadata: Dictionary containing model metadata.
        metric_name: Name of the metric being evaluated.
        output_dir: Directory to write output files.
        append: If True, append results to existing file. If False, overwrite. Default is False.

    Raises:
        ValueError: If inputs are invalid, empty, or lists have different lengths.
        KeyError: If required keys are missing from model_metadata.
        OSError: If there are file system related errors during writing.

    Note:
        - Ensures eval_inputs and eval_outputs have the same length.
        - Creates necessary directories if they don't exist.
        - Handles special characters in metric_name and model_id for file naming.
        - Overwrites existing files with the same name without warning.
    """
    _validate_inputs(eval_inputs, eval_outputs, model_metadata, metric_name)

    fmt_metric_name = _format_name(metric_name)
    fmt_model_id = _format_name(model_metadata["model_id"])
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3]
    base_filename = f"{fmt_metric_name}_{fmt_model_id}_{model_metadata['model_type']}_{timestamp}"
    paths = _prepare_file_paths(output_dir, fmt_metric_name, fmt_model_id, base_filename)
    metadata = _prepare_metadata(model_metadata, timestamp)

    try:
        _write_json_file(paths["metadata"], metadata)

        mode = "a" if append else "w"
        with paths["results"].open(mode, encoding="utf-8") as f:
            for eval_input, eval_output in zip(eval_inputs, eval_outputs, strict=True):
                result = {
                    "sample": eval_input.model_dump(),
                    "feedback": eval_output.feedback,
                    "score": eval_output.score,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info(f"Results {'appended to' if append else 'saved to'} {paths['results']}")
    except OSError as e:
        logger.error(f"Error writing files: {e}")
        raise


def _validate_inputs(
    eval_inputs: list[EvalInput],
    eval_outputs: list[EvalOutput],
    model_metadata: dict[str, Any],
    metric_name: str,
) -> None:
    """Validate input parameters for the write_results_to_disk function.

    Args:
        eval_inputs: List of evaluation inputs.
        eval_outputs: List of evaluation outputs.
        model_metadata: Dictionary containing model metadata.
        metric_name: Name of the metric being evaluated.

    Raises:
        ValueError: If eval_inputs or eval_outputs are empty, have different lengths,
                    or if metric_name is empty or only whitespace.
        KeyError: If required keys ('model_id', 'model_type') are missing from
                  model_metadata.

    Note:
        This function does not validate the content of eval_inputs or eval_outputs,
        only their presence and length.
    """
    if not eval_inputs or not eval_outputs:
        raise ValueError("eval_inputs and eval_outputs cannot be empty")
    if len(eval_inputs) != len(eval_outputs):
        raise ValueError("eval_inputs and eval_outputs must have the same length")
    if not metric_name or not metric_name.strip():
        raise ValueError("metric_name cannot be empty or only whitespace")
    required_keys = {"model_id", "model_type"}
    missing_keys = required_keys - set(model_metadata.keys())
    if missing_keys:
        raise KeyError(f"model_metadata missing required keys: {missing_keys}")


def _format_name(name: str) -> str:
    """Format a name for use in file paths by removing special characters.

    Args:
        name: The name to format.

    Returns:
        A formatted string safe for use in file paths.

    Note:
        This function replaces spaces with underscores, removes non-alphanumeric
        characters (except underscore and hyphen), and replaces non-ASCII
        characters with underscores.
    """
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Remove any character that is not alphanumeric, underscore, or hyphen
    name = re.sub(r"[^\w\-]", "", name)
    # Replace any non-ASCII character with underscore
    name = re.sub(r"[^\x00-\x7F]", "_", name)
    return name


def _prepare_file_paths(
    output_dir: str | Path,
    fmt_metric_name: str,
    fmt_model_id: str,
    base_filename: str,
) -> dict[str, Path]:
    """Prepare file paths for metadata and results files.

    Args:
        output_dir: Base output directory.
        fmt_metric_name: Formatted metric name.
        fmt_model_id: Formatted model ID.
        base_filename: Base filename for output files.

    Returns:
        A dictionary containing paths for metadata and results files.

    Note:
        This function creates the necessary directories if they don't exist.
        It does not check if the resulting file paths already exist.
    """
    output_dir = Path(output_dir)
    metric_folder = output_dir / fmt_metric_name
    metadata_folder = metric_folder / f"metadata_{fmt_metric_name}_{fmt_model_id}"
    metadata_folder.mkdir(parents=True, exist_ok=True)

    return {
        "metadata": metadata_folder / f"metadata_{base_filename}.json",
        "results": metric_folder / f"results_{base_filename}.jsonl",
    }


def _prepare_metadata(model_metadata: dict[str, Any], timestamp: str) -> dict[str, Any]:
    """Prepare metadata dictionary for writing.

    Args:
        model_metadata: Dictionary containing model metadata.
        timestamp: Timestamp string.

    Returns:
        A dictionary containing prepared metadata.

    Note:
        - Adds 'library_version' and 'timestamp' to the metadata.
        - Converts Pydantic BaseModel instances to dictionaries.
        - Converts Enum instances to their values.
        - Does not deep copy the input model_metadata.
    """
    metadata = {
        "library_version": f"{flow_judge.__version__}",
        "timestamp": timestamp,
        **model_metadata,
    }
    for key, item in metadata.items():
        if isinstance(item, BaseModel):
            metadata[key] = item.model_dump()
        elif isinstance(item, Enum):
            metadata[key] = item.value
    return metadata


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    """Write data to a JSON file.

    Args:
        path: Path to the output file.
        data: Data to write to the file.

    Raises:
        OSError: If there's an error writing to the file.

    Note:
        - Uses UTF-8 encoding.
        - Overwrites the file if it already exists.
        - Ensures non-ASCII characters are preserved in the output.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_results_file(
    path: Path, eval_inputs: list[EvalInput], eval_outputs: list[EvalOutput], append: bool = False
) -> None:
    """Write results to a JSONL file.

    Args:
        path: Path to the output file.
        eval_inputs: List of evaluation inputs.
        eval_outputs: List of evaluation outputs.
        append: If True, append to the file. If False, overwrite. Default is False.

    Raises:
        OSError: If there's an error writing to the file.
        ValueError: If eval_inputs and eval_outputs have different lengths.

    Note:
        - Uses UTF-8 encoding.
        - Appends to the file if append is True, otherwise overwrites.
        - Each line in the file is a JSON object representing one result.
        - Ensures non-ASCII characters are preserved in the output.
    """
    if len(eval_inputs) != len(eval_outputs):
        raise ValueError("eval_inputs and eval_outputs must have the same length")

    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for input_data, eval_output in zip(eval_inputs, eval_outputs, strict=True):
            result = {
                "sample": input_data.model_dump(),
                "feedback": eval_output.feedback,
                "score": eval_output.score,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
