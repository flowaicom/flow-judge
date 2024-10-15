import logging
import os
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from flow_judge.models.baseten import (
    Baseten,
    BasetenError,
    BasetenModelConfig,
    ModelType,
    VllmGenerationParams,
)


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
    :type valid_generation_params: VllmGenerationParams
    :param exec_async: Whether to use async execution.
    :type exec_async: bool
    :param webhook_proxy_url: The webhook proxy URL to use.
    :type webhook_proxy_url: Optional[str]
    :param async_batch_size: The async batch size to use.
    :type async_batch_size: int
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
    assert (
        config.model_type == expected_model_type
    ), f"model_type mismatch: expected {expected_model_type}, got {config.model_type}"
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

    unexpected_keys = set(invalid_input.keys()) - set(valid_params.keys())
    assert not unexpected_keys, f"Unexpected keys in test parameters: {unexpected_keys}"

    # Verify that valid_params alone don't raise an exception
    try:
        BasetenModelConfig(**valid_params)
    except Exception as e:
        pytest.fail(f"Valid parameters raised an unexpected exception: {str(e)}")

    with pytest.raises(ValueError, match=expected_error):
        BasetenModelConfig(**test_params)


@pytest.fixture
def mock_env_with_api_key() -> Generator[None, None, None]:
    """Fixture to mock environment with a valid Baseten API key.

    :yield: None
    """
    with patch.dict(os.environ, {"BASETEN_API_KEY": "mock_api_key_" + "x" * 30}):
        yield


@pytest.fixture
def mock_env_without_api_key() -> Generator[None, None, None]:
    """Fixture to mock environment without a Baseten API key.

    :yield: None
    """
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.mark.parametrize(
    "api_key, model_deployed, model_id, expected_error",
    [
        ("valid_key", True, "model_123", None),
        ("valid_key", False, None, BasetenError),
        ("invalid_key", False, None, BasetenError),
        (None, False, None, BasetenError),
    ],
)
def test_baseten_initialization(
    api_key: str | None,
    model_deployed: bool,
    model_id: str | None,
    expected_error: type[BasetenError] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Baseten initialization under various scenarios.

    :param api_key: The API key to use for initialization
    :param model_deployed: Whether the model is already deployed
    :param model_id: The model ID to return
    :param expected_error: The expected error type, if any
    :param monkeypatch: pytest's monkeypatch fixture
    """
    monkeypatch.setenv("BASETEN_API_KEY", api_key or "")

    with (
        patch("flow_judge.models.baseten.get_deployed_model_id") as mock_get_id,
        patch("flow_judge.models.baseten.ensure_model_deployment") as mock_deploy,
    ):
        mock_get_id.return_value = model_id
        mock_deploy.return_value = model_deployed

        if expected_error:
            with pytest.raises(expected_error):
                Baseten()
        else:
            baseten = Baseten()
            assert isinstance(baseten, Baseten)
            assert baseten.config.model_id == model_id


@pytest.mark.asyncio
async def test_baseten_init_valid(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test the initialization of Baseten class with valid parameters.

    This test ensures that the Baseten class can be properly initialized
    with valid parameters and that all attributes are set correctly.
    It uses patch to mock the ensure_model_deployment function and
    monkeypatch to set a mock BASETEN_WEBHOOK_SECRET.

    :param monkeypatch: pytest fixture for modifying the test environment
    :param caplog: pytest fixture for capturing log output
    :raises AssertionError: If any of the assertions fail or if unexpected
                            exceptions occur.

    .. note::
        - This test covers both synchronous and asynchronous initialization.
        - It does not test the actual functionality of the Baseten class methods.

    .. warning::
        - This test relies on mocked functions and environment variables, which
          may not reflect real-world behavior.
        - It does not test network-related issues or API interactions.
        - The test does not verify the correct implementation of the API adapters.
    """
    caplog.set_level(logging.DEBUG)
    with (
        patch("flow_judge.models.baseten.ensure_model_deployment", return_value=True),
        patch("flow_judge.models.baseten.get_deployed_model_id", return_value="mock_model_id"),
    ):
        # Test synchronous initialization
        try:
            baseten_sync = Baseten(
                exec_async=False,
                async_batch_size=128,
            )
        except Exception as e:
            pytest.fail(f"Synchronous Baseten initialization failed unexpectedly: {str(e)}")

        assert isinstance(baseten_sync, Baseten), "Created object is not a Baseten instance"
        assert baseten_sync.config.model_id == "mock_model_id", "model_id mismatch"
        assert baseten_sync.config.model_type == ModelType.BASETEN_VLLM, "model_type mismatch"
        assert baseten_sync.config.exec_async is False, "exec_async mismatch"
        assert baseten_sync.config.async_batch_size == 128, "async_batch_size mismatch"
        assert baseten_sync.config.webhook_proxy_url is None, "webhook_proxy_url should be None"

        # Set mock BASETEN_WEBHOOK_SECRET for async test
        monkeypatch.setenv("BASETEN_WEBHOOK_SECRET", "mock_secret")

        # Patch ensure_baseten_webhook_secret to simulate the warning
        with patch(
            "flow_judge.models.adapters.baseten.webhook.ensure_baseten_webhook_secret",
            side_effect=lambda: (
                logging.warning("Invalid BASETEN_WEBHOOK_SECRET in environment variable"),
                True,
            )[1],
        ):
            # Test asynchronous initialization
            try:
                baseten_async = Baseten(
                    exec_async=True,
                    async_batch_size=256,
                    webhook_proxy_url="https://example.com/webhook",
                )
            except Exception as e:
                pytest.fail(f"Asynchronous Baseten initialization failed unexpectedly: {str(e)}")

            assert any(
                "Invalid BASETEN_WEBHOOK_SECRET in environment variable" in record.message
                for record in caplog.records
            ), "Warning not found in logs"

            assert isinstance(baseten_async, Baseten), "Created object is not a Baseten instance"
            assert baseten_async.config.model_id == "mock_model_id", "model_id mismatch"
            assert (
                baseten_async.config.model_type == ModelType.BASETEN_VLLM_ASYNC
            ), "model_type mismatch"
            assert baseten_async.config.exec_async is True, "exec_async mismatch"
            assert baseten_async.config.async_batch_size == 256, "async_batch_size mismatch"
            assert (
                baseten_async.config.webhook_proxy_url == "https://example.com/webhook"
            ), "webhook_proxy_url mismatch"

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
        assert baseten_custom.config.model_id == "custom_id", "Custom _model_id not set correctly"


@pytest.mark.parametrize(
    "env_secret, stored_secret, user_input, expected_result",
    [
        ("whsec_valid", None, None, True),
        (None, "whsec_valid", None, True),
        (
            None,
            None,
            "whsec_valid",
            False,
        ),  # Changed to False because _is_interactive is mocked to False
        (
            None,
            None,
            "invalid",
            False,
        ),  # Changed to False because _is_interactive is mocked to False
        (None, None, "", False),
    ],
)
def test_ensure_baseten_webhook_secret(
    env_secret: str | None,
    stored_secret: str | None,
    user_input: str | None,
    expected_result: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the ensure_baseten_webhook_secret function.

    :param env_secret: The secret to set in the environment
    :param stored_secret: The secret to return as stored
    :param user_input: The user input to simulate
    :param expected_result: The expected result of the function
    :param monkeypatch: pytest's monkeypatch fixture
    """
    from flow_judge.models.adapters.baseten.webhook import ensure_baseten_webhook_secret

    if env_secret:
        monkeypatch.setenv("BASETEN_WEBHOOK_SECRET", env_secret)
    else:
        monkeypatch.delenv("BASETEN_WEBHOOK_SECRET", raising=False)

    with (
        patch("flow_judge.models.adapters.baseten.webhook._get_stored_secret") as mock_get_stored,
        patch("flow_judge.models.adapters.baseten.webhook._is_interactive", return_value=False),
        patch("getpass.getpass") as mock_getpass,
        patch("flow_judge.models.adapters.baseten.webhook._save_webhook_secret") as mock_save,
    ):
        mock_get_stored.return_value = stored_secret
        mock_getpass.return_value = user_input

        result = ensure_baseten_webhook_secret()

        assert result == expected_result

        if env_secret:
            mock_get_stored.assert_not_called()
        elif stored_secret:
            mock_get_stored.assert_called_once()
        else:
            mock_getpass.assert_not_called()  # Because _is_interactive is False
            mock_save.assert_not_called()


def test_baseten_format_conversation() -> None:
    """Test the _format_conversation method of the Baseten class."""
    baseten = Baseten(_model_id="test_model")
    prompt = "Hello, world!"
    formatted = baseten._format_conversation(prompt)
    assert formatted == [{"role": "user", "content": "Hello, world!"}]

    prompt_with_whitespace = "  Hello, world!  "
    formatted = baseten._format_conversation(prompt_with_whitespace)
    assert formatted == [{"role": "user", "content": "Hello, world!"}]


@pytest.mark.asyncio
async def test_baseten_generate_methods(caplog: pytest.LogCaptureFixture) -> None:
    """Test the generate methods of the Baseten class."""
    with patch.dict(os.environ, {"BASETEN_WEBHOOK_SECRET": "mock_secret"}):
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
