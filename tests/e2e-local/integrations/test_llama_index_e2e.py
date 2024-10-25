import asyncio
import json
import logging
import os
import tempfile
from collections import Counter
from pathlib import Path

import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.llama_dataset import download_llama_dataset

from flow_judge.integrations.llama_index import LlamaIndexFlowJudge
from flow_judge.metrics import CustomMetric, RubricItem
from flow_judge.models import Vllm

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="module")
def test_cache_dir():
    """Create a temporary directory for test cache.

    This fixture creates a temporary directory that is guaranteed to be
    writable and cleaned up after the tests.

    :yield: Path object pointing to the temporary directory
    :rtype: pathlib.Path
    """
    with tempfile.TemporaryDirectory(prefix="flow-judge-test-") as tmpdir:
        temp_path = Path(tmpdir)
        logging.info(f"Created temporary test cache directory: {temp_path}")
        yield temp_path
    logging.info(f"Cleaned up temporary test cache directory: {temp_path}")


@pytest.fixture
def query():
    """Provides a sample query about the Industrial Revolution's impact on urbanization.

    Returns:
        str: A query string about the Industrial Revolution's effects on 19th century England.
    """
    return """Analyze the impact of the Industrial Revolution on urbanization in 19th century
    England, focusing on demographic shifts, living conditions, and social reforms. Include
    specific examples and statistics to support your analysis."""


@pytest.fixture
def reference():
    """Provides a reference answer about the Industrial Revolution's impact on urbanization.

    Returns:
        str: A detailed reference answer covering demographic shifts, living conditions,
            and social reforms in 19th century England.
    """
    return """The Industrial Revolution in England during the 19th century had a profound impact
    on urbanization, resulting in significant demographic shifts, changes in living conditions,
    and the emergence of social reforms.

    1. Demographic Shifts:
       - Rural-to-urban migration: Between 1811 and 1851, the urban population in England and
         Wales increased from 35% to 54%.
       - Population growth: London's population grew from 1 million in 1800 to 6.7 million by 1900.
       - Example: Manchester's population increased tenfold, from 75,000 in 1801 to 750,000 by
         1901.

    2. Living Conditions:
       - Overcrowding: In 1851, the average number of people per house in Liverpool was 7.6,
         compared to 5.5 in rural areas.
       - Poor sanitation: In 1842, only 5% of working-class homes in Manchester had drainage
         systems.
       - High mortality rates: Life expectancy in industrial cities was 29 years, compared to 45
         years in rural areas.
       - Example: The cholera epidemic of 1832 killed over 20,000 people in England, with a
         disproportionate impact on urban areas.

    3. Social Reforms:
       - Public Health Act of 1848: Established local boards of health to improve sanitation and
         water supply.
       - Factory Acts (1833, 1844, 1847): Regulated working conditions and hours, especially for
         women and children.
       - Education Act of 1870: Introduced compulsory elementary education.
       - Example: The creation of model towns like Saltaire (1851) and Bournville (1879) aimed to
         provide better living conditions for workers.

    In conclusion, the Industrial Revolution led to rapid urbanization in 19th century England,
    causing significant demographic shifts and initially worsening living conditions. However,
    these challenges ultimately spurred social reforms that aimed to improve the quality of life
    for urban residents."""


@pytest.fixture
def response():
    """Provides a sample response to the query about the Industrial Revolution's impact.

    Returns:
        str: A detailed response covering demographic shifts, living conditions,
            and social reforms in 19th century England during the Industrial Revolution.
    """
    return """The Industrial Revolution in 19th century England significantly influenced
    urbanization, causing substantial changes in demographics, living conditions, and social
    structures.

    1. Demographic Shifts:
       - There was a notable rural-to-urban migration, with the urban population in England and
         Wales increasing from 35% to 60% between 1811 and 1851.
       - Cities grew rapidly. For example, London's population expanded from 1 million in 1800 to
         5.5 million by 1900.
       - Manchester's population increased from 75,000 in 1801 to 500,000 by 1901.

    2. Living Conditions:
       - Overcrowding was a major issue. In Liverpool, the average number of people per house in
         1851 was 8.2, compared to 6.0 in rural areas.
       - Sanitation was poor, with only 10% of working-class homes in Manchester having drainage
         systems in 1842.
       - Health problems were widespread. Life expectancy in industrial cities dropped to 25
         years, while it remained at 40 years in rural areas.
       - The cholera epidemic of 1832 exemplifies the health crisis, killing over 30,000 people in
         England, primarily in urban areas.

    3. Social Reforms:
       - The Public Health Act of 1850 was introduced to improve sanitation and water supply in
         urban areas.
       - Factory Acts were passed in 1833 and 1845 to regulate working conditions, particularly
         for women and children.
       - The Education Act of 1875 made elementary education compulsory, addressing the need for a
         more educated workforce.
       - Some industrialists created model towns, such as New Lanark (1851) and Port Sunlight
         (1879), to provide better living conditions for workers.

    These changes transformed England's urban landscape, creating challenges that eventually led
    to social and legislative reforms aimed at improving the quality of life for city dwellers.
    However, the full impact of these reforms wasn't felt until the early 20th century."""


@pytest.fixture
def contexts():
    """Provides a list of context strings about Amazon's history and business.

    Returns:
        list[str]: A list of factual statements about Amazon's founding, growth,
            and current status.
    """
    return [
        "Amazon started as an online bookstore in 1994, founded by Jeff Bezos in his garage in "
        "Bellevue, Washington.",
        "Over the years, Amazon expanded into various product categories beyond books, including "
        "electronics, clothing, furniture, food, toys, and more.",
        "Amazon's business model has diversified to include online retail, cloud computing services"
        " (Amazon Web Services), digital streaming, and artificial intelligence.",
        "In 1999, Amazon introduced its Marketplace feature, allowing third-party sellers to offer "
        "their products alongside Amazon's offerings.",
        "Amazon launched Amazon Prime in 2005, a subscription service offering free two-day"
        " shipping and other benefits to members.",
        "The company entered the e-reader market with the Kindle in 2007, revolutionizing digital "
        "book consumption.",
        "Amazon acquired Whole Foods Market in 2017, marking its significant entry into the "
        "brick-and-mortar grocery business.",
        "As of 2023, Amazon is one of the world's most valuable companies and a leader in"
        " e-commerce, cloud computing, and artificial intelligence technologies.",
    ]


@pytest.fixture
def correctness_metric():
    """Creates a CustomMetric for evaluating the correctness of generated answers.

    Returns:
        CustomMetric: A metric object with evaluation criteria and rubric for assessing
            answer correctness.
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


@pytest.mark.asyncio
async def test_correctness_evaluation(
    correctness_metric, query, reference, response, test_cache_dir
):
    """Tests the correctness evaluation of a generated response using LlamaIndexFlowJudge.

    Args:
        correctness_metric (CustomMetric): The metric used for evaluation.
        query (str): The input query.
        reference (str): The reference answer.
        response (str): The generated response to evaluate.
        test_cache_dir (Path): Temp dir that is sure to be okay during testing

    Raises:
        AssertionError: If the evaluation score is outside the expected range or feedback
            is missing.
    """
    os.environ["HF_HOME"] = str(test_cache_dir)
    model = Vllm(
        exec_async=True,
        gpu_memory_utilization=0.5,
        quantized=True,
        download_dir=str(test_cache_dir),
    )
    flow_judge_evaluator = LlamaIndexFlowJudge(
        model=model, metric=correctness_metric, save_results=True
    )
    result = await flow_judge_evaluator.aevaluate(
        query=query, reference=reference, response=response
    )
    if result:
        assert 2 <= int(result.score) <= 4
        assert result.feedback is not None
    else:
        raise AssertionError("Couldn't process 'result'")
    del model


def get_scores_distribution(scores: list[float]) -> dict[float, str]:
    """Calculates the distribution of scores as percentages.

    Args:
        scores (list[float]): A list of numerical scores.

    Returns:
        dict[float, str]: A dictionary mapping scores to their percentage occurrence.
    """
    score_counts = Counter(scores)
    total_scores = len(scores)
    return {score: f"{(count / total_scores) * 100:.1f}%" for score, count in score_counts.items()}


def compare_distributions(
    actual: dict[float, str], expected: dict[float, str], tolerance: float = 20.0
) -> bool:
    """Compares two score distributions within a given tolerance.

    Args:
        actual (dict[float, str]): The actual score distribution.
        expected (dict[float, str]): The expected score distribution.
        tolerance (float, optional): The maximum allowed difference between percentages.
            Defaults to 20.0.

    Returns:
        bool: True if the distributions are within the tolerance, False otherwise.
    """
    for score in set(actual.keys()) | set(expected.keys()):
        actual_pct = float(actual.get(score, "0%").rstrip("%"))
        expected_pct = float(expected.get(score, "0%").rstrip("%"))
        if abs(actual_pct - expected_pct) > tolerance:
            return False
    return True


@pytest.mark.asyncio
async def test_batch_evaluation(correctness_metric, query, reference, response, test_cache_dir):
    """Performs a batch evaluation of queries using LlamaIndexFlowJudge and analyzes results."""
    os.environ["HF_HOME"] = str(test_cache_dir)
    model = None
    flow_judge_correctness = None

    try:
        logging.info("Starting test_batch_evaluation")

        logging.info("Initializing Vllm model")
        model = Vllm(
            exec_async=True,
            gpu_memory_utilization=0.5,
            quantized=True,
            download_dir=str(test_cache_dir),
        )
        logging.info("Vllm model initialized")

        logging.info("Initializing LlamaIndexFlowJudge")
        flow_judge_correctness = LlamaIndexFlowJudge(
            model=model,
            metric=correctness_metric,
            output_dir=str(test_cache_dir),
            save_results=True,
        )
        logging.info("LlamaIndexFlowJudge initialized")

        logging.info("Downloading and preparing dataset")
        rag_dataset, documents = await asyncio.to_thread(
            download_llama_dataset, "MiniTruthfulQADataset", "./data/mini_truthful_qa"
        )
        logging.info("Dataset prepared")

        logging.info("Creating index and query engine")
        index = await asyncio.to_thread(VectorStoreIndex.from_documents, documents=documents)
        _ = index.as_query_engine()
        logging.info("Index and query engine created")

        rag_subset = rag_dataset.examples[:10]
        queries = [example.query for example in rag_subset]
        references = [example.reference_answer for example in rag_subset]

        # Use the fixtures to create a batch of 3
        queries = [query] * 3
        references = [reference] * 3
        responses = [response] * 3

        logging.info(f"Starting evaluation of {len(queries)} queries")
        eval_results = await flow_judge_correctness.aevaluate_batch(
            queries=queries, responses=responses, references=references, save_results=True
        )
        logging.info(f"Finished evaluation, got {len(eval_results)} results")

        # Check results
        assert len(eval_results) == len(queries)
        for result in eval_results:
            assert result.score is not None
            assert result.feedback is not None

        # Verify results were written to disk
        expected_metric_folder = test_cache_dir / correctness_metric.name
        expected_results_file = next(expected_metric_folder.glob("results_*.jsonl"), None)

        assert (
            expected_results_file is not None
        ), f"No results file found in {expected_metric_folder}"

        # Read and verify the results file
        with expected_results_file.open("r") as f:
            written_results = [json.loads(line) for line in f]

        logging.info("Full content of written_results:")
        logging.info(json.dumps(written_results, indent=2))

        assert len(written_results) == len(
            queries
        ), f"Expected {len(queries)} results, but found {len(written_results)}"

        for i, result in enumerate(written_results):
            logging.info(f"Validating result {i+1}")
            assert "sample" in result, f"Result {i+1} is missing 'sample' key"
            assert "inputs" in result["sample"], f"Result {i+1} is missing 'inputs' key in 'sample'"

            evaluation = result["sample"]["inputs"]
            logging.info(f"Evaluation {i+1} structure:")
            logging.info(json.dumps(evaluation, indent=2))

            assert (
                len(evaluation) == 2
            ), f"Expected 2 input components in result {i+1}, but found {len(evaluation)}"
            assert (
                "query" in evaluation[0]
            ), f"First input component in result {i+1} should be the query"
            assert (
                "reference" in evaluation[1]
            ), f"Second input component in result {i+1} should be the reference"

            assert (
                evaluation[0]["query"] == queries[i]
            ), f"Query in result {i+1} doesn't match the input query"

            assert "feedback" in result, f"Result {i+1} is missing 'feedback'"
            assert "score" in result, f"Result {i+1} is missing 'score'"
            assert isinstance(
                result["score"], (int, float)
            ), f"Score in result {i+1} should be a number"

        logging.info(f"Results successfully written to and verified in {expected_results_file}")

        logging.info("Results processing and verification completed")

    finally:
        if model:
            await model.aclose()

        if flow_judge_correctness and hasattr(flow_judge_correctness, "aclose"):
            await flow_judge_correctness.aclose()

        # Allow time for background tasks to complete
        await asyncio.sleep(1)

        # Cancel any remaining tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]

        # Wait for all tasks to be cancelled
        await asyncio.gather(*tasks, return_exceptions=True)

    logging.info("test_batch_evaluation completed successfully")
