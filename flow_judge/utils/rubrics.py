import os
import webbrowser
from urllib.parse import quote

import yaml
from IPython.display import HTML, display
from pydantic import BaseModel

from flow_judge.flow_judge import FlowJudge
from flow_judge.metrics import CustomMetric
from flow_judge.metrics import RubricItem as MetricRubricItem
from flow_judge.models import Hf, Llamafile, Vllm


class RubricItem(BaseModel):
    """Represents an item in a rubric with a score and description."""

    score: int
    description: str


class RubricTemplate(BaseModel):
    """Template for a complete rubric including metadata and evaluation criteria."""

    name: str
    description: str
    criteria: str
    rubric: list[RubricItem]
    required_inputs: list[str]
    required_output: str


def load_rubric_from_yaml(file_path: str) -> RubricTemplate:
    """Load a rubric template from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        RubricTemplate: Loaded rubric template.
    """
    with open(file_path) as file:
        data = yaml.safe_load(file)
        for item in data["rubric"]:
            item["score"] = int(item["score"])
        return RubricTemplate(**data)


def load_rubric_templates(directory: str) -> dict[str, RubricTemplate]:
    """Load all rubric templates from a directory.

    Args:
        directory (str): Path to the directory containing YAML files.

    Returns:
        dict[str, RubricTemplate]: Dictionary of loaded rubric templates.
    """
    templates = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".yaml"):
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                template_name = os.path.splitext(relative_path)[0]
                templates[template_name] = load_rubric_from_yaml(os.path.join(root, file))
    return templates


def create_metric_from_template(template: RubricTemplate) -> CustomMetric:
    """Create a CustomMetric from a RubricTemplate.

    Args:
        template (RubricTemplate): The rubric template to convert.

    Returns:
        CustomMetric: The created metric.
    """
    return CustomMetric(
        name=template.name,
        criteria=template.criteria,
        rubric=[
            MetricRubricItem(score=item.score, description=item.description)
            for item in template.rubric
        ],
        required_inputs=template.required_inputs,
        required_output=template.required_output,
    )


def create_judge_from_yaml(file_path: str, model: Vllm | Hf | Llamafile) -> FlowJudge:
    """Create a FlowJudge instance from a YAML file and a model.

    Args:
        file_path (str): Path to the YAML file containing the rubric.
        model (Vllm | Hf | Llamafile): The model to use for judging.

    Returns:
        FlowJudge: The created FlowJudge instance.
    """
    template = load_rubric_from_yaml(file_path)
    metric = create_metric_from_template(template)
    return FlowJudge(metric=metric, model=model)


def is_notebook() -> bool:
    """Check if the current environment is a Jupyter notebook.

    Returns:
        bool: True if in a notebook, False otherwise.
    """
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False


def request_rubric(
    title: str,
    description: str,
    similar_to: str | None = None,
    custom_fields: dict[str, str] | None = None,
):
    """Create a new rubric request and open it in the browser or display it in a notebook.

    Args:
        title (str): Title of the rubric request.
        description (str): Description of the rubric request.
        similar_to (str | None): Name of a similar existing rubric.
        custom_fields (dict[str, str] | None): Additional custom fields for the request.
    """
    issue_title = f"Rubric Request: {title}"
    templates = load_rubric_templates("example_rubrics")
    similar_template = templates.get(similar_to, None) if similar_to else None

    issue_body = f"""
## Rubric Request

**Title:** {title}

**Description:**
{description}

## Similar Rubric
{f"This request is similar to the existing rubric: `{similar_to}`" if similar_to else "N/A"}

## Proposed Structure
```yaml
name: {title.lower().replace(' ', '_')}
description: {description}
criteria: [TO BE FILLED]
rubric:
  - score: 0
    description: [TO BE FILLED]
  - score: 1
    description: [TO BE FILLED]
required_inputs: {similar_template.required_inputs if similar_template else '[TO BE FILLED]'}
required_output: {similar_template.required_output if similar_template else '[TO BE FILLED]'}
```

## Additional Information
{yaml.dump(custom_fields) if custom_fields else "Please provide any additional context"
 " or requirements for this rubric."}

## Existing Rubrics for Reference
{yaml.dump({name: template.description for name, template in templates.items()})}
"""

    encoded_body = quote(issue_body)
    url = f"https://github.com/flowaicom/flow-judge/issues/new?title={quote(issue_title)}&body={encoded_body}&labels=enhancement,rubric-request"

    if is_notebook():
        display(
            HTML(
                f"""
        <a href="{url}" target="_blank">
            <button style="background-color: #4CAF50; border: none; color: white; padding: "
            "15px 32px; text-align: center; text-decoration: none; display: inline-block; "
            "font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px; "
            "box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);">
                Create New Rubric Request
            </button>
        </a>
        """
            )
        )
    else:
        webbrowser.open(url)
        print("Browser opened with the rubric request creation page.")


def display_rubric_request_form():
    """Display an interactive form for creating rubric requests in a Jupyter notebook."""
    templates = load_rubric_templates("example_rubrics")
    options = "".join([f'<option value="{name}">{name}</option>' for name in templates.keys()])

    form_html = f"""
    <form id="rubricForm" style="max-width: 500px; margin: 20px auto; padding: 20px; border: "
    "1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="margin-bottom: 15px;">
            <label for="title" style="display: block; margin-bottom: 5px;">Rubric Title:</label>
            <input type="text" id="title" name="title" required style="width: 100%; padding: 8px;"
            " border: 1px solid #ccc; border-radius: 4px;">
        </div>
        <div style="margin-bottom: 15px;">
            <label for="description" style="display: block; margin-bottom: 5px;">Description:"
            "</label>
            <textarea id="description" name="description" required style="width: 100%; "
            "height: 100px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;"></textarea>
        </div>
        <div style="margin-bottom: 15px;">
            <label for="similarTo" style="display: block; margin-bottom: 5px;">Similar to existing"
            " rubric:</label>
            <select id="similarTo" name="similarTo" style="width: 100%; padding: 8px; border:"
            " 1px solid #ccc; border-radius: 4px;">
                <option value="">Select a rubric</option>
                {options}
            </select>
        </div>
        <button type="submit" style="background-color: #4CAF50; border: none; color: white;"
        " padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block;"
        " font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 4px;">
            Create Rubric Request
        </button>
    </form>
    <div id="result"></div>
    <script>
    document.getElementById('rubricForm').addEventListener('submit', function(e) {{
        e.preventDefault();
        var title = document.getElementById('title').value;
        var description = document.getElementById('description').value;
        var similarTo = document.getElementById('similarTo').value;

        IPython.notebook.kernel.execute(`request_rubric("${{title}}", "${{description}}","
        " "${{similarTo}}")`);
    }});
    </script>
    """
    display(HTML(form_html))
