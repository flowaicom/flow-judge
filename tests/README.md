
# Tests for Flow Judge

This directory contains the test suite for the Flow Judge project.

## Test Coverage

Below is the current test coverage visualization for the Flow Judge project:

<p align="center">
  <a href="https://codecov.io/gh/flowaicom/flow-judge" target="_blank">
    <img src="https://codecov.io/gh/flowaicom/flow-judge/graphs/icicle.svg?token=AEGC7W3DGE" alt="Codecov Icicle Graph">
  </a>
</p>


## Directory Structure
tests/
├── README.md
├── unit/
│ ├── test_flow_judge.py
│ ├── test_metrics.py
│ └── test_models.py
├── integration/
│ ├── test_evaluation_pipeline.py
│ └── test_model_integration.py
└── fixtures/
└── sample_data.json


## Running Tests

To run the entire test suite:
```sh
pytest
```


To run a specific test file:
```sh
pytest tests/unit/test_flow_judge.py
```

To run tests with coverage report:
```sh
pytest --cov=flow_judge --cov-report=term-missing
```


## Contributing

When adding new features or modifying existing ones, please make sure to add or update the corresponding tests. This helps maintain the project's reliability and makes it easier to catch potential issues early.

## Continuous Integration

Our CI pipeline automatically runs these tests on every pull request and push to the main branch. You can check the status of the latest runs in the GitHub Actions tab of the repository.
