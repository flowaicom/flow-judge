import logging
import os

import pytest

from flow_judge.models.llamafile import Llamafile

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define MEMORY_LIMIT (10 GB in bytes)
MEMORY_LIMIT = 10 * 1024 * 1024 * 1024  # 10 GB

"""
End-to-end test suite for the Llamafile class.

This test suite performs comprehensive tests on the Llamafile class,
including initialization, file operations, server management, and text generation.
These tests interact with the actual Llamafile system and may take longer to run
compared to unit tests.
"""


@pytest.fixture(scope="module")
def test_cache_dir():
    """Set up the test environment.

    This fixture creates a temporary directory to use as a cache for the
    Llamafile downloads and operations. It's used for all tests in the module.
    """
    cache_dir = "/tmp/flow-judge-test-cache"
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Test cache directory created at {cache_dir}")
    yield cache_dir
    # Clean up after all tests
    for file in os.listdir(cache_dir):
        os.remove(os.path.join(cache_dir, file))
    os.rmdir(cache_dir)
    logger.info(f"Test cache directory {cache_dir} cleaned up and removed")


@pytest.mark.memray(threshold=MEMORY_LIMIT)
def test_llamafile_initialization(test_cache_dir):
    """Test Llamafile initialization with various parameters.

    This test verifies that a Llamafile instance is correctly initialized
    with custom configuration parameters, including port, context size,
    GPU layers, thread count, batch size, concurrent requests, and
    generation parameters.
    """
    logger.info("Starting Llamafile initialization test")
    llamafile = Llamafile(
        cache_dir=test_cache_dir,
        port=9000,
        context_size=4096,
        gpu_layers=20,
        thread_count=4,
        batch_size=16,
        max_concurrent_requests=2,
        generation_params={
            "temperature": 0.8,
            "top_p": 0.9,
            "max_new_tokens": 500,
        },
    )

    assert llamafile.config.model_id == "flowaicom/Flow-Judge-v0.1-Llamafile"
    assert llamafile.config.port == 9000
    assert llamafile.config.context_size == 4096
    assert llamafile.config.gpu_layers == 20
    assert llamafile.config.thread_count == 4
    assert llamafile.config.batch_size == 16
    assert llamafile.config.max_concurrent_requests == 2
    assert llamafile.config.generation_params.temperature == 0.8
    assert llamafile.config.generation_params.top_p == 0.9
    assert llamafile.config.generation_params.max_new_tokens == 500
    logger.info("Llamafile initialization test completed successfully")


@pytest.mark.memray(threshold=MEMORY_LIMIT)
def test_download_llamafile(test_cache_dir):
    """Test downloading of Llamafile.

    This test ensures that the Llamafile is correctly downloaded to the
    specified cache directory and that the downloaded file has the correct
    permissions (executable).
    """
    logger.info("Starting Llamafile download test")
    llamafile = Llamafile(cache_dir=test_cache_dir)
    llamafile_path = llamafile.download_llamafile()

    assert os.path.exists(llamafile_path)
    assert os.access(llamafile_path, os.X_OK)
    logger.info(f"Llamafile successfully downloaded to {llamafile_path}")


@pytest.mark.memray(threshold=MEMORY_LIMIT)
def test_build_llamafile_command(test_cache_dir):
    """Test building of Llamafile command.

    This test verifies that the Llamafile command is correctly constructed
    based on the provided configuration parameters. It checks for the
    presence of essential command-line arguments in the built command.
    """
    logger.info("Starting Llamafile command building test")
    llamafile = Llamafile(
        cache_dir=test_cache_dir,
        port=9000,
        context_size=4096,
        gpu_layers=20,
        thread_count=4,
        batch_size=16,
        max_concurrent_requests=2,
        generation_params={"temperature": 0.8, "max_new_tokens": 500},
        quantized_kv=True,
        flash_attn=True,
    )

    llamafile_path = llamafile.download_llamafile()
    command = llamafile._build_llamafile_command(llamafile_path)

    expected_args = [
        llamafile_path,
        "--port 9000",
        "-c 4096",
        "-ngl 20",
        "--threads 4",
        "-b 16",
        "--parallel 2",
        "--temp 0.8",
        "-n 500",
        "-ctk q4_0",
        "-ctv q4_0",
        "-fa",
    ]
    for arg in expected_args:
        assert arg in command
    logger.info("Llamafile command successfully built and verified")


@pytest.mark.memray(threshold=MEMORY_LIMIT)
def test_start_stop_server(test_cache_dir):
    """Test starting and stopping of Llamafile server.

    This test verifies that the Llamafile server can be started and stopped
    correctly, and that the is_server_running method accurately reflects
    the server's state.
    """
    logger.info("Starting Llamafile server start/stop test")
    llamafile = Llamafile(cache_dir=test_cache_dir)

    llamafile.start_llamafile_server()
    assert llamafile.is_server_running()
    logger.info("Llamafile server successfully started")

    llamafile.stop_llamafile_server()
    assert not llamafile.is_server_running()
    logger.info("Llamafile server successfully stopped")


@pytest.mark.memray(threshold=MEMORY_LIMIT)
def test_generate(test_cache_dir):
    """Test text generation.

    This test verifies that the Llamafile can generate text responses
    to a given prompt. It checks that the generated response is a
    non-empty string.
    """
    logger.info("Starting text generation test")
    llamafile = Llamafile(cache_dir=test_cache_dir)

    with llamafile:
        response = llamafile._generate("Hello, world!")
        assert isinstance(response, str)
        assert len(response) > 0
    logger.info("Text generation test completed successfully")


@pytest.mark.memray(threshold=MEMORY_LIMIT)
def test_batch_generate(test_cache_dir):
    """Test batch text generation.

    This test verifies that the Llamafile can generate text responses
    for multiple prompts in a batch. It checks that the number of
    responses matches the number of prompts and that each response
    is a non-empty string.
    """
    logger.info("Starting batch text generation test")
    llamafile = Llamafile(cache_dir=test_cache_dir)

    with llamafile:
        prompts = ["Hello, world!", "How are you?", "What's the weather like?"]
        responses = llamafile._batch_generate(prompts)
        assert len(responses) == len(prompts)
        for i, response in enumerate(responses):
            assert isinstance(response, str)
            assert len(response) > 0
            logger.info(f"Response {i+1} generated successfully")
    logger.info("Batch text generation test completed successfully")
