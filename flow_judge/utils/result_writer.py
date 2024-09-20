import json
import logging
import os
import re
from datetime import datetime, timezone

from flow_judge.eval_data_types import EvalInput, EvalOutput

logger = logging.getLogger(__name__)


def write_results_to_disk(
    eval_inputs: list[EvalInput],
    eval_outputs: list[EvalOutput],
    model_metadata: dict,
    metric_name: str,
    output_dir: str,
):
    """Write evaluation results, inputs, and metadata to separate JSONL files."""
    fmt_metric_name = re.sub(r"\s", "_", re.sub(r"\(|\)", "", metric_name.lower()))
    fmt_model_id = model_metadata["model_id"].replace("/", "__")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3]
    metadata = {
        "library_version": "flow_judge.__version__",  # Replace with actual version import
        "timestamp": timestamp,
        **model_metadata,
    }

    metric_folder = os.path.join(output_dir, fmt_metric_name)
    os.makedirs(metric_folder, exist_ok=True)

    base_filename = f"{fmt_metric_name}_{fmt_model_id}_{model_metadata['model_type']}_{timestamp}"
    metadata_path = os.path.join(metric_folder, f"metadata_{base_filename}.jsonl")
    results_path = os.path.join(metric_folder, f"results_{base_filename}.jsonl")

    # Write metadata file
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata) + "\n")

    # Write results file
    with open(results_path, "w", encoding="utf-8") as f:
        for input_data, eval_output in zip(eval_inputs, eval_outputs):
            result = {
                "sample": input_data.model_dump(),
                "feedback": eval_output.feedback,
                "score": eval_output.score,
            }
            f.write(json.dumps(result) + "\n")

    logger.info(f"Results saved to {results_path}")
