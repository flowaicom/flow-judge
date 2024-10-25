# `flow-judge`

<p align="center">
  <img src="img/flow_judge_banner.png" alt="Flow Judge Banner">
</p>

<p align="center" style="font-family: 'Courier New', Courier, monospace;">
  <strong>
    <a href="https://www.flow-ai.com/judge">Technical Report</a> |
    <a href="https://huggingface.co/collections/flowaicom/flow-judge-v01-66e6af5fc3b3a128bde07dec">Model Weights</a> |
    <a href="https://huggingface.co/spaces/flowaicom/Flow-Judge-v0.1">HuggingFace Space</a> |
    <a href="https://github.com/flowaicom/lm-evaluation-harness/tree/Flow-Judge-v0.1_evals/lm_eval/tasks/flow_judge_evals">Evaluation Code</a> |
    <a href="https://github.com/flowaicom/flow-judge/tree/main/examples">Tutorials</a>
  </strong>
</p>

<p align="center" style="font-family: 'Courier New', Courier, monospace;">
  <code>flow-judge</code> is a lightweight library for evaluating LLM applications with <code>Flow-Judge-v0.1</code>.
</p>

<p align="center">
<a href="https://github.com/flowaicom/flow-judge/stargazers/" target="_blank">
    <img src="https://img.shields.io/github/stars/flowaicom/flow-judge?style=social&label=Star&maxAge=3600" alt="GitHub stars">
</a>
<a href="https://github.com/flowaicom/flow-judge/releases" target="_blank">
    <img src="https://img.shields.io/github/v/release/flowaicom/flow-judge?color=white" alt="Release">
</a>
<a href="https://www.youtube.com/@flowaicom" target="_blank">
    <img alt="YouTube Channel Views" src="https://img.shields.io/youtube/channel/views/UCo2qL1nIQRHiPc0TF9xbqwg?style=social">
</a>
<a href="https://github.com/flowaicom/flow-judge/actions/workflows/test-and-lint.yml" target="_blank">
    <img src="https://github.com/flowaicom/flow-judge/actions/workflows/test-and-lint.yml/badge.svg" alt="Build">
</a>
<a href="https://codecov.io/gh/flowaicom/flow-judge" target="_blank">
    <img src="https://codecov.io/gh/flowaicom/flow-judge/branch/main/graph/badge.svg?token=AEGC7W3DGE" alt="Code coverage">
</a>
<a href="https://github.com/flowaicom/flow-judge/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/static/v1?label=license&message=Apache%202.0&color=white" alt="License">
</a>
<a href="https://app.fossa.com/projects/git%2Bgithub.com%2Fflowaicom%2Fflow-judge?ref=badge_shield" alt="FOSSA Status"><img src="https://app.fossa.com/api/projects/git%2Bgithub.com%2Fflowaicom%2Fflow-judge.svg?type=shield"/></a>
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
- `dev` to install development dependencies
- `hf` to install Hugging Face Transformers dependencies
- `vllm` to install vLLM dependencies
- `llamafile` to install Llamafile dependencies
- `baseten` to install Baseten dependencies

## Quick Start

Here's a simple example to get you started:

```python
from flow_judge import Vllm, Llamafile, Hf, EvalInput, FlowJudge
from flow_judge.metrics import RESPONSE_FAITHFULNESS_5POINT
from IPython.display import Markdown, display

# If you are running on an Ampere GPU or newer, create a model using VLLM
model = Vllm()

# If you have other applications open taking up VRAM, you can use less VRAM by setting gpu_memory_utilization to a lower value.
# model = Vllm(gpu_memory_utilization=0.70)

# Or if not running on Ampere GPU or newer, create a model using no flash attn and Hugging Face Transformers
# model = Hf(flash_attn=False)

# Or create a model using Llamafile if not running an Nvidia GPU & running a Silicon MacOS for example
# model = Llamafile()

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

### Inference Options

The library supports multiple inference backends to accommodate different hardware configurations and performance needs:

1. **vLLM**:
   - Best for NVIDIA GPUs with Ampere architecture or newer (e.g., RTX 3000 series, A100, H100)
   - Offers the highest performance and throughput
   - Requires CUDA-compatible GPU

   ```python
   from flow_judge import Vllm

   model = Vllm()
   ```

2. **Hugging Face Transformers**:
   - Compatible with a wide range of hardware, including older NVIDIA GPUs
   - Supports CPU inference (slower but universally compatible)
   - It is slower than vLLM but generally compatible with more hardware.

    If you are running on an Ampere GPU or newer:
   ```python
   from flow_judge import Hf

   model = Hf()
   ```

   If you are not running on an Ampere GPU or newer, disable flash attention:
   ```python
   from flow_judge import Hf

   model = Hf(flash_attn=False)
   ```

3. **Llamafile**:
   - Ideal for non-NVIDIA hardware, including Apple Silicon
   - Provides good performance on CPUs
   - Self-contained, easy to deploy option

   ```python
   from flow_judge import Llamafile

   model = Llamafile()
   ```

4. **Baseten**:
    - Remote execution.
    - Machine independent.
    - Improved concurrency patterns for larger workloads.

  ```python
  from flow_judge import Baseten

  model = Baseten()
  ```
  For detailed information on using Baseten, visit the [Baseten readme](https://github.com/flowaicom/flow-judge/blob/feat/baseten-integration/flow_judge/models/adapters/baseten/README.md).

Choose the inference backend that best matches your hardware and performance requirements. The library provides a unified interface for all these options, making it easy to switch between them as needed.


### Evaluation Metrics

`Flow-Judge-v0.1` was trained to handle any custom metric that can be expressed as a combination of evaluation criteria and rubric, and required inputs and outputs.

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
from flow_judge import Vllm, EvalInput, FlowJudge
from flow_judge.metrics import RESPONSE_FAITHFULNESS_5POINT
from IPython.display import Markdown, display

# Initialize the model
model = Vllm()

# Initialize the judge
faithfulness_judge = FlowJudge(
    metric=RESPONSE_FAITHFULNESS_5POINT,
    model=model
)

# Load some sampledata
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

> [!WARNING]
> There exists currently a reported issue with Phi-3 models that produces gibberish outputs with contexts longer than 4096 tokens, including input and output. This issue has been recently fixed in the transformers library so we recommend using the `Hf()` model configuration for longer contexts at the moment. For more details, refer to: [#33129](https://github.com/huggingface/transformers/pull/33129) and [#6135](https://github.com/vllm-project/vllm/issues/6135)


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

We support an integration with Llama Index evaluation module and Haystack:
- [Llama Index tutorial](https://github.com/flowaicom/flow-judge/blob/main/examples/4_llama_index_evaluators.ipynb)
- [Haystack tutorial](https://github.com/flowaicom/flow-judge/blob/main/examples/5_evaluate_haystack_rag_pipeline.ipynb)

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

6. Make sure you have trufflehog installed:
   ```bash
   # make trufflehog available in your path
   # macos
   brew install trufflehog
   # linux
   curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh | sh -s -- -b /usr/local/bin
   # nix
   nix profile install nixpkgs#trufflehog
   ```

7. Run pre-commit on all files:
   ```bash
   pre-commit run --all-files
   ```

8. You're now ready to start developing! You can run the main script with:
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

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fflowaicom%2Fflow-judge.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fflowaicom%2Fflow-judge?ref=badge_large)

## Acknowledgments

Flow-Judge is developed and maintained by the Flow AI team. We appreciate the contributions and feedback from the AI community in making this tool more robust and versatile.
