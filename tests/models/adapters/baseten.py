import asyncio
import os
from unittest.mock import Mock, patch

import pytest
from openai import OpenAIError

from flow_judge.models.adapters.baseten.adapter import AsyncBasetenAPIAdapter, BasetenAPIAdapter


@pytest.fixture
def baseten_api_adapter():
    """Fixture to create a BasetenAPIAdapter instance with a test model_id and API key.

    Yields the adapter instance and cleans up the environment variable after the test.
    """
    model_id = "test_model_id"
    baseten_api_key = "test_api_key"
    os.environ["BASETEN_API_KEY"] = baseten_api_key
    adapter = BasetenAPIAdapter(model_id)
    yield adapter
    del os.environ["BASETEN_API_KEY"]


@pytest.fixture
def async_baseten_api_adapter():
    """Fixture to create an AsyncBasetenAPIAdapter instance with test values.

    Yields the adapter instance and cleans up the environment variables after the test.
    """
    model_id = "test_model_id"
    webhook_proxy_url = "https://test.webhook.com"
    batch_size = 2
    baseten_api_key = "test_api_key"
    os.environ["BASETEN_API_KEY"] = baseten_api_key
    os.environ["BASETEN_WEBHOOK_SECRET"] = "test_secret"
    adapter = AsyncBasetenAPIAdapter(model_id, webhook_proxy_url, batch_size)
    yield adapter
    del os.environ["BASETEN_API_KEY"]
    del os.environ["BASETEN_WEBHOOK_SECRET"]


def test_baseten_api_adapter_init(baseten_api_adapter):
    """Test case to ensure the BasetenAPIAdapter instance is initialized correctly.

    Sets values for baseten_model_id, baseten_api_key, and base_url.
    """
    assert baseten_api_adapter.baseten_model_id == "test_model_id"
    assert baseten_api_adapter.baseten_api_key == "test_api_key"
    assert baseten_api_adapter.base_url == "https://bridge.baseten.co/v1/direct"


def test_baseten_api_adapter_init_missing_api_key():
    """Test case to ensure a ValueError is raised when creating a BasetenAPIAdapter instance.

    For when a BASETEN_API_KEY environment variable is not set.
    """
    with pytest.raises(ValueError):
        BasetenAPIAdapter("test_model_id")


@patch("flow_judge.models.adapters.baseten.adapter.OpenAI")
def test_make_request(mock_openai, baseten_api_adapter):
    """Test case to ensure the _make_request method works as expected under different scenarios.

    Includes an OpenAI error, and other exceptions.
    """
    message = {"role": "user", "content": "test content"}
    request_messages = [message]

    # Test OpenAI error
    mock_openai.return_value.chat.completions.create.side_effect = OpenAIError("test error")
    result = baseten_api_adapter._make_request(request_messages)
    assert result is None

    # Test other exceptions
    mock_openai.return_value.chat.completions.create.side_effect = Exception("test exception")
    result = baseten_api_adapter._make_request(request_messages)
    assert result is None


@patch("flow_judge.models.adapters.baseten.adapter.BasetenAPIAdapter._make_request")
def test_fetch_response(mock_make_request, baseten_api_adapter):
    """Test case to ensure the _fetch_response method works as expected under different scenarios.

    Includes a successful response and exception handling.
    """
    message = {"role": "user", "content": "test content"}
    request_messages = [message]

    # Test successful response
    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(content="Hello, world!"))]
    mock_make_request.return_value = mock_completion
    result = baseten_api_adapter._fetch_response(request_messages)
    assert result == "Hello, world!"

    # Test exception handling
    mock_make_request.return_value = None
    result = baseten_api_adapter._fetch_response(request_messages)
    assert result == ""


@patch("flow_judge.models.adapters.baseten.adapter.BasetenAPIAdapter._make_request")
def test_fetch_batched_response(mock_make_request, baseten_api_adapter):
    """Test case to ensure the _fetch_batched_response method works as expected.

    Includes a successful batch response and exception handling.
    """
    message = {"role": "user", "content": "test content"}
    request_messages = [message, message]

    # Test successful batch response
    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(content="test response"))]
    mock_make_request.return_value = mock_completion

    result = baseten_api_adapter._fetch_batched_response(request_messages)
    assert result == ["test response"]

    # Exception handling and defaults
    mock_make_request.return_value = None
    result = baseten_api_adapter._fetch_batched_response(request_messages)
    assert result == [""]


@patch("flow_judge.models.adapters.baseten.webhook.ensure_baseten_webhook_secret")
def test_async_baseten_api_adapter_init(mock_ensure_webhook_secret, async_baseten_api_adapter):
    """Test case to ensure the AsyncBasetenAPIAdapter instance is initialized correctly.

    Including expected values for baseten_model_id, webhook_proxy_url, batch_size,
    baseten_api_key, and base_url.
    """
    mock_ensure_webhook_secret.return_value = True
    assert async_baseten_api_adapter.baseten_model_id == "test_model_id"
    assert async_baseten_api_adapter.webhook_proxy_url == "https://test.webhook.com"
    assert async_baseten_api_adapter.batch_size == 2
    assert async_baseten_api_adapter.baseten_api_key == "test_api_key"
    assert async_baseten_api_adapter.base_url == "https://model-test_model_id.api.baseten.co/production"


def test_async_baseten_api_adapter_init_missing_api_key():
    """Test case to ensure a ValueError is raised.

    On creation of AsyncBasetenAPIAdapter instance without a BASETEN_API_KEY variable set.
    """
    with pytest.raises(ValueError):
        AsyncBasetenAPIAdapter("test_model_id", "https://test.webhook.com", 2)


def test_async_baseten_api_adapter_init_missing_webhook_secret(monkeypatch):
    """Test case to ensure a ValueError is raised.

    On creation of AsyncBasetenAPIAdapter instance without a BASETEN_WEBHOOK_SECRET variable set.
    """
    monkeypatch.setenv("BASETEN_WEBHOOK_SECRET", "")
    with pytest.raises(ValueError):
        AsyncBasetenAPIAdapter("test_model_id", "https://test.webhook.com", 2)


@patch("aiohttp.ClientSession.post")
def test_async_make_request(mock_post, async_baseten_api_adapter):
    """Test case to ensure the _make_request method of AsyncBasetenAPIAdapter works as expected.

    Includes a successful request, request failure, and exception handling.
    """
    message = {"role": "user", "content": "test content"}
    request_messages = [message]

    # Test successful request
    mock_response = Mock()
    mock_response.json.return_value = {"request_id": "test_request_id"}
    mock_post.return_value.__aenter__.return_value.json.return_value = \
        mock_response.json.return_value
    result = asyncio.run(async_baseten_api_adapter._make_request(request_messages))
    assert result == "test_request_id"

    # Test request failure
    mock_response.json.return_value = {"error": "test error"}
    mock_post.return_value.__aenter__.return_value.json.return_value = \
        mock_response.json.return_value
    result = asyncio.run(async_baseten_api_adapter._make_request(request_messages))
    assert result is None

    # Test exception handling
    mock_response.json.return_value = {}
    result = asyncio.run(async_baseten_api_adapter._make_request(request_messages))
    assert result is None


@patch("flow_judge.models.adapters.baseten.adapter.AsyncBasetenAPIAdapter._make_request")
@patch("flow_judge.models.adapters.baseten.adapter.AsyncBasetenAPIAdapter._fetch_stream")
def test_async_fetch_response(mock_fetch_stream, mock_make_request, async_baseten_api_adapter):
    """Test case to ensure the _async_fetch_response method of AsyncBasetenAPIAdapter works.

    Includes a successful response.
    """
    message = {"role": "user", "content": "test content"}
    request_messages = [message]

    # Test successful response
    mock_make_request.return_value = "test_request_id"
    mock_fetch_stream.return_value = "Hello, world!"
    result = asyncio.run(async_baseten_api_adapter._async_fetch_response(request_messages))
    assert result == "Hello, world!"


@patch("flow_judge.models.adapters.baseten.adapter.AsyncBasetenAPIAdapter._make_request")
@patch("flow_judge.models.adapters.baseten.adapter.AsyncBasetenAPIAdapter._fetch_stream")
def test_async_fetch_batched_response(
    mock_fetch_stream,
    mock_make_request,
    async_baseten_api_adapter
):
    """Test case to ensure the _async_fetch_batched_response method of AsyncBasetenAPIAdapter works.

    Includes a successful batch response and batch with exceptions.
    """
    message = {"role": "user", "content": "test content"}
    request_messages = [message, message, message]

    # Test successful batch response
    mock_make_request.side_effect = ["request_id1", "request_id2", "request_id3"]
    mock_fetch_stream.side_effect = ["Hello, world!", "Hello, universe!", "Hello, galaxy!"]
    result = asyncio.run(async_baseten_api_adapter._async_fetch_batched_response(request_messages))
    assert result == ["Hello, world!", "Hello, universe!", "Hello, galaxy!"]

    # Test batch with exceptions
    mock_fetch_stream.side_effect = [
        "Hello, world!",
        asyncio.TimeoutError("timeout"),
        "Hello, galaxy!",
    ]
    mock_make_request.side_effect = ["request_id1", "request_id2", "request_id3"]
    result = asyncio.run(async_baseten_api_adapter._async_fetch_batched_response(request_messages))
    assert result == ["Hello, world!", "", "Hello, galaxy!"]

