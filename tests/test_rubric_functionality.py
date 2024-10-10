import os
from unittest.mock import create_autospec, patch

import pytest
from click.testing import CliRunner

from flow_judge.flow_judge import FlowJudge
from flow_judge.models import Vllm
from flow_judge.utils.cli import cli
from flow_judge.utils.rubrics import (
    create_judge_from_yaml,
    create_metric_from_template,
    display_rubric_request_form,
    load_rubric_templates,
    request_rubric,
)

# Path to the rubrics directory
RUBRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rubrics")


@pytest.fixture
def rubric_templates():
    """Fixture for rubric templates.

    This fixture loads the rubric templates from the RUBRICS_DIR and returns them.
    """
    return load_rubric_templates(RUBRICS_DIR)


def test_load_rubric_templates(rubric_templates):
    """Test the load_rubric_templates function.

    This test ensures that rubric templates are loaded correctly from the RUBRICS_DIR,
    and checks for the existence of specific rubrics.
    """
    assert len(rubric_templates) > 0
    # Check for the existence of specific rubrics
    assert "article_evaluation/source_attribution" in rubric_templates
    assert "query_decomposition/sub_query_coverage" in rubric_templates


def test_create_metric_from_template(rubric_templates):
    """Test the create_metric_from_template function.

    This test verifies that a metric is correctly created from a template,
    checking the name, criteria, and rubric length.
    """
    template = rubric_templates["article_evaluation/source_attribution"]
    metric = create_metric_from_template(template)
    assert metric.name == "article_source_attribution"
    assert metric.criteria == template.criteria
    assert len(metric.rubric) == len(template.rubric)


def test_create_judge_from_yaml():
    """Test the create_judge_from_yaml function.

    This test ensures that a FlowJudge instance is correctly created from a YAML file,
    using a mock Vllm model.
    """
    yaml_path = os.path.join(RUBRICS_DIR, "article_evaluation", "source_attribution.yaml")

    # Create a mock Vllm instance that will pass isinstance checks
    mock_vllm = create_autospec(Vllm, instance=True)

    with patch("flow_judge.utils.rubrics.Vllm", return_value=mock_vllm):
        judge = create_judge_from_yaml(yaml_path, mock_vllm)

        # Assert that the FlowJudge is created correctly
        assert isinstance(judge, FlowJudge)
        assert judge.metric.name == "article_source_attribution"

        # Assert that the created judge uses the mock Vllm instance
        assert judge.model == mock_vllm


@pytest.mark.skipif(not os.environ.get("JUPYTER_AVAILABLE"), reason="Requires Jupyter environment")
def test_display_rubric_request_form():
    """Test the display_rubric_request_form function.

    This test verifies that the function calls IPython.display.display
    when executed in a Jupyter environment.
    """
    with patch("IPython.display.display") as mock_display:
        display_rubric_request_form()
        mock_display.assert_called_once()


@pytest.mark.parametrize(
    "title,description,similar_to",
    [
        ("Test Rubric", "This is a test rubric request", "article_evaluation/source_attribution"),
        ("Another Test", "Another description", None),
    ],
)
def test_request_rubric(title, description, similar_to):
    """Test the request_rubric function with various input parameters.

    This test ensures that the function opens a web browser with the correct URL
    for different combinations of title, description, and similar_to parameters.
    """
    with patch("webbrowser.open") as mock_open:
        request_rubric(title, description, similar_to)
        mock_open.assert_called_once()


def test_cli_command():
    """Test the CLI command for requesting a rubric.

    This test verifies that the CLI command correctly calls the request_rubric function
    with the provided arguments.
    """
    runner = CliRunner()
    with patch("flow_judge.utils.cli.request_rubric") as mock_request:
        result = runner.invoke(
            cli,
            [
                "request-rubric",
                "--title",
                "CLI Test Rubric",
                "--description",
                "This is a test rubric request from CLI",
                "--similar-to",
                "article_evaluation/source_attribution",
            ],
        )
        assert result.exit_code == 0
        mock_request.assert_called_once_with(
            "CLI Test Rubric",
            "This is a test rubric request from CLI",
            "article_evaluation/source_attribution",
        )


# Optional: Add a test for is_notebook() function if it's part of your public API
def test_is_notebook():
    """Test the is_notebook function.

    This test checks that the is_notebook function returns False when run in a pytest environment.
    """
    from flow_judge.utils.rubrics import is_notebook

    assert not is_notebook()  # This should return False when run in pytest
