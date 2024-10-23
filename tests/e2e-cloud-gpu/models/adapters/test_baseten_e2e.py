import logging
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.llama_dataset import download_llama_dataset
from pydantic import BaseModel

from flow_judge import Baseten
from flow_judge.integrations.llama_index import LlamaIndexFlowJudge
from flow_judge.metrics import CustomMetric, RubricItem

pytest_plugins = ("pytest_asyncio",)

logger = logging.getLogger(__name__)


class TestConfig(BaseModel):
    """Configuration for Baseten e2e tests."""

    api_key: str = os.getenv("BASETEN_API_KEY")
    model_id: str = os.getenv("BASETEN_MODEL_ID")
    webhook_url: str = os.getenv("BASETEN_WEBHOOK_URL")
    webhook_secret: str = os.getenv("BASETEN_WEBHOOK_SECRET")


@pytest.fixture(scope="module")
def test_config() -> TestConfig:
    """Fixture to load test configuration from environment variables.

    Returns:
        TestConfig: Configuration object for Baseten e2e tests.

    Raises:
        ValueError: If any required environment variable is missing.
    """
    try:
        return TestConfig()
    except ValueError as e:
        pytest.fail(f"Missing required environment variable: {str(e)}")


@pytest.fixture(scope="module")
def test_cache_dir() -> Path:
    """Create a temporary directory for test cache.

    Returns:
        Path: Path object pointing to the temporary directory.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="flow-judge-baseten-test-") as tmpdir:
        temp_path = Path(tmpdir)
        logger.info(f"Created temporary test cache directory: {temp_path}")
        yield temp_path
    logger.info(f"Cleaned up temporary test cache directory: {temp_path}")


@pytest.fixture
def correctness_metric() -> CustomMetric:
    """Creates a CustomMetric for evaluating the correctness of generated answers.

    Returns:
        CustomMetric: A metric object with evaluation criteria and rubric for
        assessing answer correctness.
    """
    evaluation_criteria = "Is the generated answer relevant to the user query and reference answer?"
    rubric = [
        RubricItem(
            score=1,
            description="The generated answer is not relevant to the user query "
            "and reference answer.",
        ),
        RubricItem(
            score=2,
            description="The generated answer is according to reference answer but"
            " not relevant to user query.",
        ),
        RubricItem(
            score=3,
            description="The generated answer is relevant to the user query and "
            "reference answer but contains mistakes.",
        ),
        RubricItem(
            score=4,
            description="The generated answer is relevant to the user query and "
            "has the exact same metrics as the reference answer, but it is not as concise.",
        ),
        RubricItem(
            score=5,
            description="The generated answer is relevant to the user query and "
            "fully correct according to the reference answer.",
        ),
    ]
    return CustomMetric(
        name="correctness",
        criteria=evaluation_criteria,
        rubric=rubric,
        required_inputs=["query", "reference"],
        required_output="response",
    )


def get_scores_distribution(scores: list[float]) -> dict[float, str]:
    """Calculates the distribution of scores as percentages.

    Args:
        scores (List[float]): A list of numerical scores.

    Returns:
        Dict[float, str]: A dictionary mapping scores to their percentage occurrence.
    """
    score_counts = Counter(scores)
    total_scores = len(scores)
    return {score: f"{(count / total_scores) * 100:.1f}%" for score, count in score_counts.items()}


def compare_distributions(
    actual: dict[float, str],
    expected: dict[float, str],
    tolerance: float = 20.0,
) -> bool:
    """Compares two score distributions within a given tolerance.

    Args:
        actual (Dict[float, str]): The actual score distribution.
        expected (Dict[float, str]): The expected score distribution.
        tolerance (float, optional): The maximum allowed difference between
            percentages. Defaults to 20.0.

    Returns:
        bool: True if the distributions are within the tolerance, False otherwise.
    """
    for score in set(actual.keys()) | set(expected.keys()):
        actual_pct = float(actual.get(score, "0%").rstrip("%"))
        expected_pct = float(expected.get(score, "0%").rstrip("%"))
        if abs(actual_pct - expected_pct) > tolerance:
            return False
    return True


async def batch_eval_runner(
    evaluators: dict[str, LlamaIndexFlowJudge],
    query_engine: Any,
    questions: list[str],
    reference: list[str] | None = None,
    num_workers: int = 2,
) -> dict[str, list[Any]]:
    """Runs batch evaluation using the provided evaluators and query engine.

    Args:
        evaluators (Dict[str, LlamaIndexFlowJudge]): Dictionary of evaluators.
        query_engine (Any): The query engine to use for generating responses.
        questions (List[str]): List of questions to evaluate.
        reference (Optional[List[str]], optional): List of reference answers.
            Defaults to None.
        num_workers (int, optional): Number of workers for parallel processing.
            Defaults to 2.

    Returns:
        Dict[str, List[Any]]: Evaluation results for each evaluator.
    """
    batch_runner = BatchEvalRunner(evaluators, workers=num_workers, show_progress=True)
    return await batch_runner.aevaluate_queries(
        query_engine, queries=questions, reference=reference
    )


@pytest.mark.asyncio
async def test_baseten_correctness_evaluation(
    test_config: TestConfig,
    correctness_metric: CustomMetric,
    test_cache_dir: Path,
) -> None:
    """Tests the correctness evaluation of Baseten model using LlamaIndexFlowJudge.

    Args:
        test_config (TestConfig): Test configuration object.
        correctness_metric (CustomMetric): The metric used for evaluation.
        test_cache_dir (Path): Temporary directory for test cache.

    Raises:
        AssertionError: If the evaluation score is outside the expected range or
            feedback is missing.
    """
    os.environ["HF_HOME"] = str(test_cache_dir)
    model = Baseten(
        _model_id=test_config.model_id,
        exec_async=True,
        webhook_proxy_url=test_config.webhook_url,
    )
    flow_judge_evaluator = LlamaIndexFlowJudge(model=model, metric=correctness_metric)

    # Download and prepare the dataset
    rag_dataset, documents = download_llama_dataset(
        "MiniTruthfulQADataset", str(test_cache_dir / "mini_truthful_qa")
    )

    # Select a single example for evaluation
    example = rag_dataset.examples[0]
    query, reference = example.query, example.reference_answer

    # Generate response using Baseten model
    response = await model._async_generate(query)

    result = await flow_judge_evaluator.aevaluate(
        query=query, reference=reference, response=response
    )

    assert result is not None, "Evaluation result is None"
    assert 2 <= int(result.score) <= 5, f"Score {result.score} is out of expected range"
    assert result.feedback is not None, "Feedback is missing"

    logger.info(f"Evaluation score: {result.score}")
    logger.info(f"Evaluation feedback: {result.feedback}")


@pytest.mark.asyncio
async def test_baseten_batch_evaluation(
    test_config: TestConfig,
    correctness_metric: CustomMetric,
    test_cache_dir: Path,
) -> None:
    """Performs a batch evaluation of queries using Baseten model and analyzes results.

    Args:
        test_config (TestConfig): Test configuration object.
        correctness_metric (CustomMetric): The metric used for evaluation.
        test_cache_dir (Path): Temporary directory for test cache.

    Raises:
        AssertionError: If the evaluation results do not meet expected criteria.
    """
    os.environ["HF_HOME"] = str(test_cache_dir)
    model = Baseten(
        _model_id=test_config.model_id,
        exec_async=True,
        webhook_proxy_url=test_config.webhook_url,
    )
    logger.info("Starting test_baseten_batch_evaluation")

    flow_judge_correctness = LlamaIndexFlowJudge(model=model, metric=correctness_metric)

    # Download and prepare the dataset
    rag_dataset, documents = download_llama_dataset(
        "MiniTruthfulQADataset", str(test_cache_dir / "mini_truthful_qa")
    )

    # Create the index and query engine
    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()

    # Prepare queries and references
    rag_subset = rag_dataset.examples[:10]
    queries = [example.query for example in rag_subset]
    references = [example.reference_answer for example in rag_subset]

    logger.info(f"Evaluating {len(queries)} queries")

    evaluators = {"correctness": flow_judge_correctness}

    eval_results = await batch_eval_runner(
        evaluators=evaluators,
        query_engine=query_engine,
        questions=queries,
        reference=references,
    )

    # Check results
    assert "correctness" in eval_results, "Correctness evaluator results missing"
    assert len(eval_results["correctness"]) == len(queries), "Incomplete evaluation results"

    for result in eval_results["correctness"]:
        assert result.score is not None, "Evaluation score is missing"
        assert result.feedback is not None, "Evaluation feedback is missing"

    # Calculate score distribution
    scores = [result.score for result in eval_results["correctness"]]
    actual_distribution = get_scores_distribution(scores)
    logger.info(f"Actual score distribution: {actual_distribution}")

    # Calculate average score
    average_score = statistics.mean(scores)
    logger.info(f"Average score: {average_score:.2f}")

    # Assert that the average score is within an acceptable range
    assert (
        3.0 <= average_score <= 4.5
    ), f"Average score {average_score:.2f} is outside the expected range of 3.0 to 4.5"

    # Check that we have a variety of scores
    unique_scores = set(scores)
    assert (
        len(unique_scores) >= 3
    ), f"Expected at least 3 different score values, but got {len(unique_scores)}"

    logger.info("test_baseten_batch_evaluation completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
