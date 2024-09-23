from flow_judge.metrics import RESPONSE_CORRECTNESS_BINARY, CustomMetric, RubricItem


def test_response_correctness_binary():
    """Test the RESPONSE_CORRECTNESS_BINARY metric."""
    metric = RESPONSE_CORRECTNESS_BINARY
    assert metric.name == "Response Correctness (Binary)"
    assert len(metric.rubric) == 2
    assert metric.required_inputs == ["query", "reference_answer"]
    assert metric.required_output == "response"


def test_custom_metric():
    """Test the CustomMetric class."""
    custom_metric = CustomMetric(
        name="Test Metric",
        criteria="Test criteria",
        rubric=[RubricItem(score=0, description="Bad"), RubricItem(score=1, description="Good")],
        required_inputs=["test_input"],
        required_output="test_output",
    )
    assert custom_metric.name == "Test Metric"
    assert custom_metric.criteria == "Test criteria"
    assert len(custom_metric.rubric) == 2
    assert custom_metric.required_inputs == ["test_input"]
    assert custom_metric.required_output == "test_output"
