import logging
import os
import sys
from collections.abc import Generator
from io import StringIO
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pytest import LogCaptureFixture, MonkeyPatch

from flow_judge.models.adapters.baseten.deploy import ensure_model_deployment
from flow_judge.models.adapters.baseten.webhook import ensure_baseten_webhook_secret
from flow_judge.models.baseten import (
    Baseten,
    BasetenError,
    BasetenModelConfig,
    ModelType,
    VllmGenerationParams,
)


@pytest.fixture
def capture_output() -> Generator[StringIO, None, None]:
    """Capture stdout for testing print statements.

    :yield: StringIO object containing captured output
    """
    captured_output = StringIO()
    sys.stdout = captured_output
    yield captured_output
    sys.stdout = sys.__stdout__


def test_ensure_model_deployment_key_exists(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Test ensure_model_deployment when BASETEN_API_KEY exists.

    :param monkeypatch: pytest's monkeypatch fixture
    :param caplog: pytest's log capture fixture
    """
    monkeypatch.setenv("BASETEN_API_KEY", "mock_api_key")

    with caplog.at_level(logging.INFO):
        with patch(
            "flow_judge.models.adapters.baseten.api_auth._validate_auth_status", return_value=True
        ):
            with patch(
                "flow_judge.models.adapters.baseten.deploy._initialize_model", return_value=True
            ):
                result = ensure_model_deployment()

    assert result is True, "Model deployment should succeed when API key exists"
    assert any(
        "Baseten authenticated" in record.message for record in caplog.records
    ), "Should log successful authentication"
    assert any(
        "Ensuring Flow Judge model deployment" in record.message for record in caplog.records
    ), "Should log deployment attempt"


def test_ensure_model_deployment_key_missing_non_interactive(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Test ensure_model_deployment when BASETEN_API_KEY is missing in non-interactive mode.

    :param monkeypatch: pytest's monkeypatch fixture
    :param caplog: pytest's log capture fixture
    """
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)

    captured_output = StringIO()
    sys.stdout = captured_output

    with caplog.at_level(logging.INFO):
        with patch(
            "flow_judge.models.adapters.baseten.api_auth._validate_auth_status", return_value=False
        ):
            with patch(
                "flow_judge.models.adapters.baseten.util.is_interactive", return_value=False
            ):
                result = ensure_model_deployment()

    sys.stdout = sys.__stdout__

    assert result is False, "Model deployment should fail when API key is missing"
    assert any(
        "Baseten authentication failed" in record.message for record in caplog.records
    ), "Should log failed authentication"
    assert any(
        "Non-interactive environment detected" in record.message for record in caplog.records
    ), "Should detect non-interactive environment"

    output = captured_output.getvalue()
    assert "To run Flow Judge remotely with Baseten, signup and generate API key" in output
    assert "Set the Baseten API key in `BASETEN_API_KEY` environment variable" in output


def test_ensure_model_deployment_key_missing_interactive(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Test ensure_model_deployment when BASETEN_API_KEY is missing in interactive mode.

    :param monkeypatch: pytest's monkeypatch fixture
    :param caplog: pytest's log capture fixture
    """
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)

    captured_output = StringIO()
    sys.stdout = captured_output

    with caplog.at_level(logging.INFO):
        with patch(
            "flow_judge.models.adapters.baseten.api_auth._validate_auth_status",
            side_effect=[False, True],
        ):
            with patch(
                "flow_judge.models.adapters.baseten.api_auth.is_interactive", return_value=True
            ):
                with patch(
                    "flow_judge.models.adapters.baseten.api_auth.getpass.getpass",
                    return_value="mock_api_key",
                ):
                    with patch(
                        "flow_judge.models.adapters.baseten.api_auth.truss.login"
                    ) as mock_login:
                        with patch(
                            "flow_judge.models.adapters.baseten.deploy._initialize_model",
                            return_value=True,
                        ):
                            result = ensure_model_deployment()

    sys.stdout = sys.__stdout__

    assert result is True, "Model deployment should succeed after entering valid API key"
    assert any(
        "Baseten authentication failed" in record.message for record in caplog.records
    ), "Should log initial failed authentication"
    assert any(
        "Prompting for API key in interactive environment" in record.message
        for record in caplog.records
    ), "Should prompt for API key"
    assert any(
        "Login successful" in record.message for record in caplog.records
    ), "Should log successful login"

    mock_login.assert_called_once_with("mock_api_key")

    output = captured_output.getvalue()
    assert "To run Flow Judge remotely with Baseten, signup and generate API key" in output


@pytest.fixture(autouse=True)
def mock_baseten_api_key() -> Generator[None, None, None]:
    """Fixture to mock the Baseten API key in the environment.

    This fixture ensures that all tests run with a mock API key,
    preventing accidental use of real credentials during testing.

    :yield: None
    """
    with patch.dict(os.environ, {"BASETEN_API_KEY": "mock_api_key"}):
        yield


@pytest.fixture(scope="session")
def valid_generation_params() -> VllmGenerationParams:
    """Fixture for creating a valid VllmGenerationParams instance.

    :return: A VllmGenerationParams instance with default values.
    :rtype: VllmGenerationParams
    :raises AssertionError: If the created instance is not of type VllmGenerationParams.

    .. note::
        This fixture does not test the VllmGenerationParams class itself.
        Separate unit tests should be written for VllmGenerationParams.

    .. warning::
        This fixture uses default values. It may not cover all possible
        valid configurations of VllmGenerationParams.
    """
    params = VllmGenerationParams()
    assert isinstance(params, VllmGenerationParams), "Failed to create valid VllmGenerationParams"
    return params


@given(
    exec_async=st.booleans(),
    webhook_proxy_url=st.one_of(
        st.none(),
        st.text(min_size=1, max_size=200).filter(lambda x: x.startswith("http")),
    ),
    async_batch_size=st.integers(min_value=1, max_value=1000),
)
def test_baseten_model_config_init_valid(
    valid_generation_params: VllmGenerationParams,
    exec_async: bool,
    webhook_proxy_url: str | None,
    async_batch_size: int,
) -> None:
    """Test the initialization of BasetenModelConfig with valid parameters.

    This test uses Hypothesis to generate a wide range of valid inputs.

    :param valid_generation_params: A fixture providing valid VllmGenerationParams.
    :param exec_async: Whether to use async execution.
    :param webhook_proxy_url: The webhook proxy URL to use.
    :param async_batch_size: The async batch size to use.
    :raises AssertionError: If any of the assertions fail or if unexpected
                            exceptions occur.

    .. note::
        - This test does not verify the behavior of the BasetenModelConfig
          class methods.
        - It only tests the initialization and attribute setting.

    .. warning::
        - This test assumes that the VllmGenerationParams fixture is correct.
        - It does not test the interaction between BasetenModelConfig and other
          components.
        - The generated values may not cover all possible real-world scenarios.
    """
    try:
        config = BasetenModelConfig(
            generation_params=valid_generation_params,
            exec_async=exec_async,
            webhook_proxy_url=webhook_proxy_url,
            async_batch_size=async_batch_size,
        )
    except ValueError as e:
        if exec_async and webhook_proxy_url is None:
            pytest.skip(
                "Skipping invalid combination of exec_async=True and webhook_proxy_url=None"
            )
        else:
            pytest.fail(f"Unexpected ValueError: {str(e)}")
    except Exception as e:
        pytest.fail(f"BasetenModelConfig initialization failed unexpectedly: {str(e)}")

    assert isinstance(config, BasetenModelConfig), "Not a BasetenModelConfig instance"
    expected_model_type = ModelType.BASETEN_VLLM_ASYNC if exec_async else ModelType.BASETEN_VLLM
    assert config.model_type == expected_model_type, (
        f"model_type mismatch: expected {expected_model_type}, " f"got {config.model_type}"
    )
    assert config.generation_params == valid_generation_params, "generation_params mismatch"
    assert (
        config.exec_async == exec_async
    ), f"exec_async mismatch: expected {exec_async}, got {config.exec_async}"
    assert config.webhook_proxy_url == webhook_proxy_url, (
        f"webhook_proxy_url mismatch: expected {webhook_proxy_url}, "
        f"got {config.webhook_proxy_url}"
    )
    assert config.async_batch_size == async_batch_size, (
        f"async_batch_size mismatch: expected {async_batch_size}, " f"got {config.async_batch_size}"
    )


@pytest.mark.parametrize(
    "invalid_input, expected_error",
    [
        ({"async_batch_size": 0}, "async_batch_size must be > 0, got 0"),
        ({"async_batch_size": -1}, "async_batch_size must be > 0, got -1"),
        (
            {"exec_async": True, "webhook_proxy_url": None},
            "webhook_proxy_url is required for async execution",
        ),
        (
            {"generation_params": {}},
            "generation_params must be an instance of VllmGenerationParams",
        ),
        (
            {"generation_params": None},
            "generation_params must be an instance of VllmGenerationParams",
        ),
    ],
)
def test_baseten_model_config_init_invalid(
    invalid_input: dict[str, Any],
    expected_error: str,
    valid_generation_params: VllmGenerationParams,
) -> None:
    """Test the initialization of BasetenModelConfig with invalid parameters.

    :param invalid_input: Dictionary of invalid input parameters.
    :param expected_error: Expected error message.
    :param valid_generation_params: A fixture providing valid VllmGenerationParams.
    :raises AssertionError: If the expected error is not raised or if unexpected
                            exceptions occur.
    """
    valid_params = {
        "generation_params": valid_generation_params,
        "exec_async": False,
        "webhook_proxy_url": "http://example.com",
        "async_batch_size": 128,
    }
    test_params = {**valid_params, **invalid_input}

    with patch("flow_judge.models.baseten.get_deployed_model_id", return_value="mock_model_id"):
        try:
            BasetenModelConfig(**valid_params)
        except Exception as e:
            pytest.fail(f"Valid parameters raised an unexpected exception: {str(e)}")

        with pytest.raises(ValueError, match=expected_error):
            BasetenModelConfig(**test_params)


@pytest.mark.asyncio
async def test_baseten_init_valid(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test valid initializations of the Baseten class.

    This test covers both synchronous and asynchronous initializations,
    as well as various error cases and custom model ID usage.

    :param monkeypatch: pytest's monkeypatch fixture
    :param caplog: pytest's log capture fixture
    """
    with (
        patch("flow_judge.models.baseten.get_deployed_model_id", return_value="mock_model_id"),
        patch("flow_judge.models.baseten.ensure_model_deployment", return_value=True),
        patch(
            "flow_judge.models.adapters.baseten.webhook.ensure_baseten_webhook_secret",
            return_value=True,
        ),
    ):
        # Test synchronous initialization
        try:
            baseten_sync = Baseten(exec_async=False, async_batch_size=128)
        except Exception as e:
            pytest.fail(f"Synchronous Baseten initialization failed unexpectedly: {str(e)}")

        assert isinstance(baseten_sync, Baseten)
        assert baseten_sync.config.model_id == "mock_model_id"
        assert baseten_sync.config.model_type == ModelType.BASETEN_VLLM
        assert baseten_sync.config.exec_async is False
        assert baseten_sync.config.async_batch_size == 128
        assert baseten_sync.config.webhook_proxy_url is None

        # Test asynchronous initialization with webhook secret
        monkeypatch.setenv("BASETEN_WEBHOOK_SECRET", "whsec_mockwebhooksecret")
        try:
            baseten_async = Baseten(
                exec_async=True,
                async_batch_size=256,
                webhook_proxy_url="https://example.com/webhook",
            )
        except Exception as e:
            pytest.fail(f"Asynchronous Baseten initialization failed unexpectedly: {str(e)}")

        assert isinstance(baseten_async, Baseten)
        assert baseten_async.config.model_id == "mock_model_id"
        assert baseten_async.config.model_type == ModelType.BASETEN_VLLM_ASYNC
        assert baseten_async.config.exec_async is True
        assert baseten_async.config.async_batch_size == 256
        assert baseten_async.config.webhook_proxy_url == "https://example.com/webhook"

        # Test asynchronous initialization without webhook secret
        monkeypatch.delenv("BASETEN_WEBHOOK_SECRET", raising=False)
    with (
        patch("flow_judge.models.baseten.get_deployed_model_id", return_value="mock_model_id"),
        patch("flow_judge.models.baseten.ensure_model_deployment", return_value=True),
        patch(
            "flow_judge.models.baseten.ensure_baseten_webhook_secret",
            return_value=False,
        ),
    ):
        with pytest.raises(
            BasetenError,
            match=(
                "Unable to retrieve Baseten's webhook secret. "
                "Please ensure the webhook secret is provided in "
                "BASETEN_WEBHOOK_SECRET environment variable."
            ),
        ):
            Baseten(
                exec_async=True,
                async_batch_size=256,
                webhook_proxy_url="https://example.com/webhook",
            )

    # Test error cases
    with pytest.raises(
        ValueError, match="webhook_proxy_url is required for async Baseten execution"
    ):
        Baseten(exec_async=True, async_batch_size=128)

    with pytest.raises(ValueError, match="async_batch_size must be greater than 0"):
        Baseten(exec_async=False, async_batch_size=0)

    with (
        patch("flow_judge.models.baseten.ensure_model_deployment", return_value=False),
        patch("flow_judge.models.baseten.get_deployed_model_id", return_value=None),
    ):
        with pytest.raises(BasetenError, match="Baseten deployment is not available"):
            Baseten(exec_async=False, async_batch_size=128)

    with (
        patch("flow_judge.models.baseten.ensure_model_deployment", return_value=True),
        patch("flow_judge.models.baseten.get_deployed_model_id", side_effect=[None, None]),
    ):
        with pytest.raises(BasetenError, match="Unable to retrieve Baseten's deployed model id"):
            Baseten(exec_async=False, async_batch_size=128)

    # Test with custom _model_id
    with patch("flow_judge.models.baseten.get_deployed_model_id", return_value=None):
        baseten_custom = Baseten(exec_async=False, async_batch_size=128, _model_id="custom_id")
        assert baseten_custom.config.model_id == "custom_id"


@pytest.mark.parametrize(
    "env_secret, stored_secret, is_interactive, expected_result, expected_logs, expect_output",
    [
        (
            "whsec_valid",
            None,
            False,
            True,
            ["Found BASETEN_WEBHOOK_SECRET in environment variables"],
            False,
        ),
        (None, "whsec_valid", False, True, ["Found stored webhook secret"], False),
        (None, None, False, False, ["Non-interactive environment detected"], True),
        (None, None, True, True, ["Prompting user for webhook secret"], True),
    ],
)
def test_ensure_baseten_webhook_secret(
    env_secret: str | None,
    stored_secret: str | None,
    is_interactive: bool,
    expected_result: bool,
    expected_logs: list[str],
    expect_output: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    """Test the ensure_baseten_webhook_secret function comprehensively.

    :param env_secret: The secret to set in the environment
    :param stored_secret: The secret to return as stored
    :param is_interactive: Whether the environment is interactive
    :param expected_result: The expected result of the function
    :param expected_logs: Expected log messages
    :param expect_output: Whether to expect console output
    :param monkeypatch: pytest's monkeypatch fixture
    :param caplog: pytest's log capture fixture
    """
    if env_secret:
        monkeypatch.setenv("BASETEN_WEBHOOK_SECRET", env_secret)
    else:
        monkeypatch.delenv("BASETEN_WEBHOOK_SECRET", raising=False)

    captured_output = StringIO()
    sys.stdout = captured_output

    with caplog.at_level(logging.INFO):
        with patch(
            "flow_judge.models.adapters.baseten.webhook._get_stored_secret"
        ) as mock_get_stored:
            with patch(
                "flow_judge.models.adapters.baseten.webhook.is_interactive"
            ) as mock_interactive:
                with patch("getpass.getpass", return_value="whsec_valid"):
                    with patch("flow_judge.models.adapters.baseten.webhook._save_webhook_secret"):
                        mock_get_stored.return_value = stored_secret
                        mock_interactive.return_value = is_interactive
                        result = ensure_baseten_webhook_secret()

    sys.stdout = sys.__stdout__

    assert result == expected_result, f"Expected {expected_result}, got {result}"
    for log in expected_logs:
        assert any(log in record.message for record in caplog.records), f"Missing log: {log}"

    if expect_output:
        if is_interactive:
            assert (
                "To run Flow Judge remotely with Baseten and enable async execution"
                in captured_output.getvalue()
            )
            assert (
                "Warning: The provided webhook secret is probably invalid."
                in captured_output.getvalue()
            )
        else:
            assert (
                "Set the Baseten webhook secret in the BASETEN_WEBHOOK_SECRET"
                in captured_output.getvalue()
            )
    else:
        assert captured_output.getvalue() == "", f"Unexpected output: {captured_output.getvalue()}"


def test_baseten_format_conversation() -> None:
    """Test the _format_conversation method of the Baseten class."""
    with (
        patch("flow_judge.models.baseten.get_deployed_model_id", return_value="mock_model_id"),
        patch("flow_judge.models.baseten.ensure_model_deployment", return_value=True),
    ):
        baseten = Baseten(_model_id="test_model")
        prompt = "Hello, world!"
        formatted = baseten._format_conversation(prompt)
        assert formatted == [{"role": "user", "content": "Hello, world!"}]

        prompt_with_whitespace = "  Hello, world!  "
        formatted = baseten._format_conversation(prompt_with_whitespace)
        assert formatted == [{"role": "user", "content": "Hello, world!"}]


@pytest.mark.asyncio
async def test_baseten_generate_methods(caplog: pytest.LogCaptureFixture) -> None:
    """Test the generate methods of the Baseten class.

    This test covers both synchronous and asynchronous generation methods,
    as well as error cases for calling async methods on a sync instance.

    :param caplog: pytest's log capture fixture
    """
    with (
        patch("flow_judge.models.baseten.get_deployed_model_id", return_value="mock_model_id"),
        patch("flow_judge.models.baseten.ensure_model_deployment", return_value=True),
        patch(
            "flow_judge.models.adapters.baseten.webhook.ensure_baseten_webhook_secret",
            return_value=True,
        ),
        patch.dict(os.environ, {"BASETEN_WEBHOOK_SECRET": "whsec_mockwebhooksecret"}),
    ):
        baseten = Baseten(_model_id="test_model")

        with patch.object(baseten.api_adapter, "_fetch_response", return_value="Response"):
            result = baseten._generate("Test prompt")
            assert result == "Response"

        with patch.object(
            baseten.api_adapter, "_fetch_batched_response", return_value=["Response1", "Response2"]
        ):
            result = baseten._batch_generate(["Prompt1", "Prompt2"])
            assert result == ["Response1", "Response2"]

        baseten_async = Baseten(
            _model_id="test_model", exec_async=True, webhook_proxy_url="http://test.com"
        )

        with patch.object(
            baseten_async.api_adapter, "_async_fetch_response", return_value="Async Response"
        ):
            result = await baseten_async._async_generate("Test prompt")
            assert result == "Async Response"

        with patch.object(
            baseten_async.api_adapter,
            "_async_fetch_batched_response",
            return_value=["Async1", "Async2"],
        ):
            result = await baseten_async._async_batch_generate(["Prompt1", "Prompt2"])
            assert result == ["Async1", "Async2"]

        # Test error case for async methods called on sync instance
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            await baseten._async_generate("Test prompt")
            assert (
                "Attempting to run an async request with a synchronous API adapter" in caplog.text
            )

        caplog.clear()
        with caplog.at_level(logging.ERROR):
            await baseten._async_batch_generate(["Prompt1", "Prompt2"])
            assert (
                "Attempting to run an async request with a synchronous API adapter" in caplog.text
            )


if __name__ == "__main__":
    pytest.main()
