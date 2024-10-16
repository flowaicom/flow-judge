## Example Rubrics and Requesting New Ones

We provide a collection of example rubrics in the `example_rubrics` directory. These rubrics are written in YAML format for easy customization and integration into your evaluation workflows.

### Browsing Example Rubrics

You can find example rubrics for various evaluation tasks in the `example_rubrics` directory. Each rubric is stored as a YAML file and includes the following information:

- `name`: A unique identifier for the rubric
- `description`: A brief description of what the rubric evaluates
- `criteria`: The main evaluation criteria
- `rubric`: A list of scoring options with descriptions
- `required_inputs`: The inputs required for the evaluation
- `required_output`: The output to be evaluated

### Requesting New Rubrics

To request a new rubric or modifications to existing ones, you can use our built-in tools:

#### From CLI:

```bash
flow-judge create-rubric-request
```

This interactive command will guide you through the process of creating a new rubric request.

#### From Jupyter Notebook:

```python
from flow_judge.notebook_utils import display_rubric_request_form

display_rubric_request_form()
```

This will display an interactive form in your notebook for creating a new rubric request.

#### Programmatically:

```python
from flow_judge.rubric_utils import request_rubric

request_rubric(
    title="Your Rubric Title",
    description="Brief description of the rubric",
    similar_to="existing_rubric_name",  # Optional
    custom_fields={"key": "value"}  # Optional
)
```

This will open a pre-filled GitHub issue in your browser, making it easy to submit your request. The issue will include:

- A proposed structure for the new rubric
- Reference to a similar existing rubric (if specified)
- A list of all existing rubrics for context
- Any additional custom fields you've provided

By using these tools, you can easily contribute to the growth and improvement of the `flow-judge` library's evaluation capabilities.


```python
from flow_judge.rubric_loader import create_judge_from_yaml

judge = create_judge_from_yaml('path/to/rubric.yaml', model_type='vllm')
result = judge.evaluate(eval_input)
```
