from flow_judge.eval_data_types import EvalInput
from flow_judge.metrics.metric import CustomMetric, Metric


def validate_eval_input(eval_input: EvalInput, metric: Metric | CustomMetric):
    """Validate that the EvalInput matches the required inputs and output in the metric."""
    input_keys = {list(input_dict.keys())[0] for input_dict in eval_input.inputs}
    output_key = list(eval_input.output.keys())[0]
    required_inputs = set(metric.required_inputs)

    if input_keys != required_inputs:
        raise ValueError(f"Input keys {input_keys} do not match required inputs {required_inputs}")

    if metric.required_output:
        if not hasattr(eval_input, "output"):
            raise ValueError(f"Required output '{metric.required_output}' is missing")
        elif metric.required_output != output_key:
            raise ValueError(
                f"""Output key '{output_key}' does not match \
                required output '{metric.required_output}'"""
            )
