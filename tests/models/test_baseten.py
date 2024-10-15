from typing import Any
from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pytest import MonkeyPatch

from flow_judge.models.adapters.baseten.adapter import AsyncBasetenAPIAdapter, BasetenAPIAdapter
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
    model_id=st.text(min_size=1, max_size=100),
    exec_async=st.booleans(),
    webhook_proxy_url=st.one_of(
        st.none(), st.text(min_size=1, max_size=200).filter(lambda x: x.startswith("http"))
    ),
    async_batch_size=st.integers(min_value=1, max_value=1000),
)
def test_baseten_model_config_init_valid(
    valid_generation_params: VllmGenerationParams,
    model_id: str,
    exec_async: bool,
    webhook_proxy_url: str | None,
    async_batch_size: int,
) -> None:
    """Test the initialization of BasetenModelConfig with valid parameters.

    This test uses Hypothesis to generate a wide range of valid inputs.

    :param valid_generation_params: A fixture providing valid VllmGenerationParams.
    :type valid_generation_params: VllmGenerationParams
    :param model_id: The model ID to use for initialization.
    :type model_id: str
    :param exec_async: Whether to use async execution.
    :type exec_async: bool
    :param webhook_proxy_url: The webhook proxy URL to use.
    :type webhook_proxy_url: Optional[str]
    :param async_batch_size: The async batch size to use.
    :type async_batch_size: int
    :raises AssertionError: If any of the assertions fail or if unexpected exceptions occur.

    .. note::
        - This test does not verify the behavior of the BasetenModelConfig class methods.
        - It only tests the initialization and attribute setting.

    .. warning::
        - This test assumes that the VllmGenerationParams fixture is correct.
        - It does not test the interaction between BasetenModelConfig and other components.
        - The generated values may not cover all possible real-world scenarios.
    """
    try:
        config = BasetenModelConfig(
            model_id=model_id,
            generation_params=valid_generation_params,
            exec_async=exec_async,
            webhook_proxy_url=webhook_proxy_url,
            async_batch_size=async_batch_size,
        )
    except ValueError as e:
        if exec_async and webhook_proxy_url is None:
            pytest.skip(
                "Skipping invalid combination of exec_async=True\
                        and webhook_proxy_url=None"
            )
        else:
            pytest.fail(f"Unexpected ValueError: {str(e)}")
    except Exception as e:
        pytest.fail(
            f"BasetenModelConfig initialization failed unexpectedly:\
                     {str(e)}"
        )

    assert isinstance(
        config, BasetenModelConfig
    ), "Created object is not a BasetenModelConfig instance"
    assert (
        config.model_id == model_id
    ), f"model_id mismatch: expected {model_id}, got {config.model_id}"
    assert (
        config.model_type == ModelType.BASETEN_VLLM
    ), f"model_type mismatch: expected {ModelType.BASETEN_VLLM}, got {config.model_type}"
    assert config.generation_params == valid_generation_params, "generation_params mismatch"
    assert (
        config.exec_async == exec_async
    ), f"exec_async mismatch: expected {exec_async}, got {config.exec_async}"
    assert (
        config.webhook_proxy_url == webhook_proxy_url
    ), f"webhook_proxy_url mismatch: expected {webhook_proxy_url}, got {config.webhook_proxy_url}"
    assert (
        config.async_batch_size == async_batch_size
    ), f"async_batch_size mismatch: expected {async_batch_size}, got {config.async_batch_size}"


@pytest.mark.parametrize(
    "invalid_input, expected_error",
    [
        ({"async_batch_size": 0}, "async_batch_size should be greater than 0, got 0"),
        ({"async_batch_size": -1}, "async_batch_size should be greater than 0, got -1"),
        (
            {"exec_async": True, "webhook_proxy_url": None},
            "Webhook proxy url should be set for async execution",
        ),
        ({"model_id": ""}, "model_id should not be empty"),
        ({"model_id": None}, "model_id should not be empty"),
        (
            {"generation_params": None},
            "generation_params should be an instance of VllmGenerationParams",
        ),
        (
            {"generation_params": {}},
            "generation_params should be an instance of VllmGenerationParams",
        ),
    ],
)
def test_baseten_model_config_init_invalid(
    invalid_input: dict[str, Any],
    expected_error: str,
    valid_generation_params: VllmGenerationParams,
) -> None:
    """Test the initialization of BasetenModelConfig with invalid parameters.

    This test verifies that the BasetenModelConfig raises appropriate exceptions
    when initialized with invalid parameters.

    :param invalid_input: A dictionary of invalid parameters to test.
    :type invalid_input: Dict[str, Any]
    :param expected_error: The expected error message.
    :type expected_error: str
    :param valid_generation_params: A fixture providing valid VllmGenerationParams.
    :type valid_generation_params: VllmGenerationParams
    :raises AssertionError: If the expected exceptions are not raised or\
          if unexpected exceptions occur.

    .. note::
        - This test covers a subset of possible invalid inputs.
        - It does not test all possible combinations of invalid parameters.

    .. warning::
        - This test may not catch all possible error conditions.
        - It assumes that the error messages in the BasetenModelConfig class remain constant.
    """
    valid_params = {
        "model_id": "test_model_id",
        "generation_params": valid_generation_params,
        "exec_async": False,
        "webhook_proxy_url": "https://example.com/webhook",
        "async_batch_size": 64,
    }

    test_params = {**valid_params, **invalid_input}

    with pytest.raises(ValueError) as excinfo:
        BasetenModelConfig(**test_params)

    assert (
        str(excinfo.value) == expected_error
    ), f"Unexpected exception message:\
          {str(excinfo.value)}"


@pytest.mark.asyncio
async def test_baseten_init_valid(monkeypatch: MonkeyPatch) -> None:
    """Test the initialization of Baseten class with valid parameters.

    This test ensures that the Baseten class can be properly initialized
    with valid parameters and that all attributes are set correctly.
    It uses monkeypatch to mock the ensure_model_deployment and
    get_deployed_model_id functions.

    :param monkeypatch: Pytest's monkeypatch fixture for mocking.
    :type monkeypatch: MonkeyPatch
    :raises AssertionError: If any of the assertions fail or if unexpected exceptions occur.

    .. note::
        - This test covers both synchronous and asynchronous initialization.
        - It does not test the actual functionality of the Baseten class methods.

    .. warning::
        - This test relies on mocked functions, which may not reflect real-world behavior.
        - It does not test network-related issues or API interactions.
        - The test does not verify the correct implementation of the API adapters.
    """
    # Mock ensure_model_deployment and get_deployed_model_id
    monkeypatch.setattr("flow_judge.models.baseten.ensure_model_deployment", lambda: True)
    monkeypatch.setattr(
        "flow_judge.models.baseten.get_deployed_model_id", lambda: "mocked_model_id"
    )
    monkeypatch.setattr(
        "flow_judge.models.adapters.baseten.adapter.ensure_baseten_webhook_secret", lambda: True
    )

    # Mock the BASETEN_API_KEY environment variable
    monkeypatch.setenv("BASETEN_API_KEY", "mocked_api_key")

    # Test synchronous initialization
    try:
        baseten_sync = Baseten(
            exec_async=False,
            async_batch_size=128,
        )
    except Exception as e:
        pytest.fail(f"Synchronous Baseten initialization failed unexpectedly: {str(e)}")

    assert isinstance(baseten_sync, Baseten), "Created object is not a Baseten instance"
    assert isinstance(
        baseten_sync.api_adapter, BasetenAPIAdapter
    ), "api_adapter is not a BasetenAPIAdapter instance"
    assert (
        baseten_sync.config.model_id == "mocked_model_id"
    ), f"model_id mismatch: expected 'mocked_model_id', got {baseten_sync.config.model_id}"
    assert (
        baseten_sync.config.model_type == ModelType.BASETEN_VLLM
    ), f"model_type mismatch: expected {ModelType.BASETEN_VLLM}, \
            got {baseten_sync.config.model_type}"
    assert isinstance(
        baseten_sync.config.generation_params, VllmGenerationParams
    ), "generation_params is not a VllmGenerationParams instance"
    assert (
        baseten_sync.config.exec_async is False
    ), f"exec_async mismatch: expected False, got {baseten_sync.config.exec_async}"
    assert (
        baseten_sync.config.async_batch_size == 128
    ), f"async_batch_size mismatch: expected 128, got {baseten_sync.config.async_batch_size}"
    assert (
        baseten_sync.config.webhook_proxy_url is None
    ), f"webhook_proxy_url mismatch: expected None, got {baseten_sync.config.webhook_proxy_url}"

    # Test asynchronous initialization
    try:
        baseten_async = Baseten(
            exec_async=True, async_batch_size=256, webhook_proxy_url="https://example.com/webhook"
        )
    except Exception as e:
        pytest.fail(f"Asynchronous Baseten initialization failed unexpectedly: {str(e)}")

    assert isinstance(baseten_async, Baseten), "Created object is not a Baseten instance"
    assert isinstance(
        baseten_async.api_adapter, AsyncBasetenAPIAdapter
    ), "api_adapter is not an AsyncBasetenAPIAdapter instance"
    assert (
        baseten_async.config.exec_async is True
    ), f"exec_async mismatch: expected True, got {baseten_async.config.exec_async}"
    assert (
        baseten_async.config.async_batch_size == 256
    ), f"async_batch_size mismatch: expected 256, got {baseten_async.config.async_batch_size}"
    assert (
        baseten_async.config.webhook_proxy_url == "https://example.com/webhook"
    ), f"webhook_proxy_url mismatch: expected 'https://example.com/webhook',\
              got {baseten_async.config.webhook_proxy_url}"

    # Test error cases
    with pytest.raises(
        ValueError,
        match="Webhook proxy url is required for async Baseten execution",
    ):
        Baseten(exec_async=True, async_batch_size=128)

    with pytest.raises(ValueError, match="async_batch_size needs to be greater than 0"):
        Baseten(exec_async=False, async_batch_size=0)

    with pytest.raises(BasetenError, match="The provided API adapter is incompatible"):
        Baseten(api_adapter=Mock())  # type: ignore

    # Test when ensure_model_deployment returns False
    monkeypatch.setattr("flow_judge.models.baseten.ensure_model_deployment", lambda: False)
    with pytest.raises(BasetenError, match="Baseten deployment is not available"):
        Baseten(exec_async=False, async_batch_size=128)

    # Test when get_deployed_model_id returns None
    monkeypatch.setattr("flow_judge.models.baseten.ensure_model_deployment", lambda: True)
    monkeypatch.setattr("flow_judge.models.baseten.get_deployed_model_id", lambda: None)
    with pytest.raises(BasetenError, match="Unable to retrieve Basten's deployed model id"):
        Baseten(exec_async=False, async_batch_size=128)
