# `flow-judge`
<p align="center">
  <img src="img/flow_judge_banner.png" alt="Flow Judge Banner">
</p>

<p align="center" style="font-family: 'Courier New', Courier, monospace;">
  <strong>
    <a href="https://www.flow-ai.com/judge">Technical Report</a> |
    <a href="https://huggingface.co/collections/flowaicom/flow-judge-v01-66e6af5fc3b3a128bde07dec">Model Weights</a> |
    <a href="https://github.com/flowaicom/lm-evaluation-harness/tree/Flow-Judge-v0.1_evals/lm_eval/tasks/flow_judge_evals">Evaluation Code</a> |
    <a href="https://github.com/flowaicom/flow-judge/tree/main/examples">Examples</a>
  </strong>
</p>

<p align="center" style="font-family: 'Courier New', Courier, monospace;">
  <code>flow-judge</code> is a lightweight library for evaluating LLM applications with <code>Flow-Judge-v0.1</code>.
</p>

## Model
`Flow-Judge-v0.1` is an open, small yet powerful language model evaluator trained on a synthetic dataset containing LLM system evaluation data by Flow AI.

You can learn more about the unique features of our model in the [technical report](https://www.flow-ai.com/blog/flow-judge#flow-judge-an-open-small-language-model-for-llm-system-evaluations).


## Features of the library

- Support for multiple model types: Hugging Face Transformers and vLLM
- Extensible architecture for custom metrics
- Pre-defined evaluation metrics
- Ease of custom metric and rubric creation
- Batched evaluation for efficient processing
- Integrations with most popular frameworks like Llama Index

## Installation

Install flow-judge using pip:

```bash
pip install -e ".[vllm,hf]"
pip install 'flash_attn>=2.6.3' --no-build-isolation
```

Extras available:
- `dev` for development dependencies
- `hf` for Hugging Face Transformers support
- `vllm` for vLLM support

## Quick Start

Here's a simple example to get you started:

```python
from flow_judge.models.model_factory import ModelFactory
from flow_judge.flow_judge import EvalInput, FlowJudge
from flow_judge.metrics import RESPONSE_FAITHFULNESS_5POINT
from IPython.display import Markdown, display

# Create a model using ModelFactory
model = ModelFactory.create_model("Flow-Judge-v0.1-AWQ")

# Initialize the judge
faithfulness_judge = FlowJudge(
    metric=RESPONSE_FAITHFULNESS_5POINT,
    model=model
)

# Sample to evaluate
query = """..."""
context = """...""""
response = """..."""

# Create an EvalInput
# We want to evaluate the response to the customer issue based on the context and the user instructions
eval_input = EvalInput(
    inputs=[
        {"query": query},
        {"context": context},
    ],
    output={"response": response},
)

# Run the evaluation
result = faithfulness_judge.evaluate(eval_input, save_results=False)

# Display the result
display(Markdown(f"__Feedback:__\n{result.feedback}\n\n__Score:__\n{result.score}"))
```

## Usage

### Supported Model Types

- Hugging Face Transformers (`hf_transformers`)
- vLLM (`vllm`)

### Evaluation Metrics

Flow-Judge-v0.1 was trained to handle any custom metric that can be expressed as a combination of evaluation criteria and rubric, and required inputs and outputs.

#### Pre-defined Metrics

For convenience, `flow-judge` library comes with pre-defined metrics such as `RESPONSE_CORRECTNESS` or `RESPONSE_FAITHFULNESS`. You can check the full list by running:

```python
from flow_judge.metrics import list_all_metrics

list_all_metrics()
```

### Batched Evaluations

For efficient processing of multiple inputs, you can use the `batch_evaluate` method:

```python
# Read the sample data
import json
from flow_judge.models.model_factory import ModelFactory
from flow_judge.flow_judge import EvalInput, FlowJudge
from flow_judge.metrics import RESPONSE_FAITHFULNESS_5POINT
from IPython.display import Markdown, display

# Create a model using ModelFactory
model = ModelFactory.create_model("Flow-Judge-v0.1-AWQ")

# Initialize the judge
faithfulness_judge = FlowJudge(
    metric=RESPONSE_FAITHFULNESS_5POINT,
    model=model
)

# Load data
with open("sample_data/csr_assistant.json", "r") as f:
    data = json.load(f)

# Create a list of inputs and outputs
inputs_batch = [
    [
        {"query": sample["query"]},
        {"context": sample["context"]},
    ]
    for sample in data
]
outputs_batch = [{"response": sample["response"]} for sample in data]

# Create a list of EvalInput
eval_inputs_batch = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch, outputs_batch)]

# Run the batch evaluation
results = faithfulness_judge.batch_evaluate(eval_inputs_batch, save_results=False)

# Visualizing the results
for i, result in enumerate(results):
    display(Markdown(f"__Sample {i+1}:__"))
    display(Markdown(f"__Feedback:__\n{result.feedback}\n\n__Score:__\n{result.score}"))
    display(Markdown("---"))
```

## Advanced Usage

### Model configurations

We currently support vLLM engine (recommended) and Hugging Face Transformers.

We are working on adding API-based usage as well as better options for CPU.

### Custom Metrics

Create your own evaluation metrics:

```python
from flow_judge.metrics import CustomMetric, RubricItem

custom_metric = CustomMetric(
    name="My Custom Metric",
    criteria="Evaluate based on X, Y, and Z.",
    rubric=[
        RubricItem(score=0, description="Poor performance"),
        RubricItem(score=1, description="Good performance"),
    ],
    required_inputs=["query"],
    required_output="response"
)

judge = FlowJudge(metric=custom_metric, config="Flow-Judge-v0.1-AWQ")
```

### Integrations

We support an integration with `Llama Index` evaluation module. See an tutorial [here](https://github.com/flowaicom/flow-judge/blob/main/examples/4_llama_index_evaluators.ipynb).

> Note that we are currently working on adding more integrations with other frameworks in the near future.
## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/flowaicom/flow-judge.git
   cd flow-judge
   ```

2. Create a virtual environment:
    ```bash
    virtualenv ./.venv
    ```
    or

    ```bash
    python -m venv ./.venv
    ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
   or
   ```bash
   pip install -e ".[dev,vllm]"
   ```
   for vLLM support.

5. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

6. Run pre-commit on all files:
   ```bash
   pre-commit run --all-files
   ```

7. You're now ready to start developing! You can run the main script with:
   ```bash
   python -m flow_judge
   ```

Remember to always activate your virtual environment when working on the project. To deactivate the virtual environment when you're done, simply run:
```bash
deactivate
```

## Running Tests

To run the tests for Flow-Judge, follow these steps:

1. Navigate to the root directory of the project in your terminal.

2. Run the tests using pytest:
   ```bash
   pytest tests/
   ```

   This will discover and run all the tests in the `tests/` directory.

3. If you want to run a specific test file, you can do so by specifying the file path:
   ```bash
   pytest tests/test_flow_judge.py
   ```

4. For more verbose output, you can use the `-v` flag:
   ```bash
   pytest -v tests/
   ```
## Contributing

Contributions to `flow-judge` are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure that your code adheres to the project's coding standards and passes all tests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Flow-Judge is developed and maintained by the Flow AI team. We appreciate the contributions and feedback from the AI community in making this tool more robust and versatile.
