from unittest.mock import mock_open, patch

import pytest
import yaml

from flow_judge.models.adapters.baseten.gpu import (
    _get_gpu_key,
    _has_gpu_key,
    _update_config,
    ensure_gpu,
)


@pytest.fixture
def mock_env_setup(monkeypatch):
    """Mock the environment variables for testing.

    :returns: None
    :rtype: None
    """
    monkeypatch.delenv("BASETEN_GPU", raising=False)


@pytest.mark.parametrize(
    "env_value, expected",
    [("H100", "H100"), ("h100", "h100"), ("A10G", "A10G"), ("a10g", "a10g"), (None, None)],
)
def test_get_gpu_key(mock_env_setup, env_value: str | None, expected: str | None, monkeypatch):
    """Test the _get_gpu_key function.

    :param mock_env_setup: Fixture to mock environment variables.
    :type mock_env_setup: None
    :param env_value: The value to set for the BASETEN_GPU environment variable.
    :type env_value: str | None
    :param expected: The expected output.
    :type expected: str | None
    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :returns: None
    :rtype: None
    """
    if env_value is not None:
        monkeypatch.setenv("BASETEN_GPU", env_value)
    assert _get_gpu_key() == expected


@pytest.mark.parametrize(
    "env_value, expected",
    [("H100", True), ("h100", True), ("A10G", True), ("a10g", True), (None, False)],
)
def test_has_gpu_key(mock_env_setup, env_value: str | None, expected: bool, monkeypatch):
    """Test the _has_gpu_key function.

    :param mock_env_setup: Fixture to mock environment variables.
    :type mock_env_setup: None
    :param env_value: The value to set for the BASETEN_GPU environment variable.
    :type env_value: str | None
    :param expected: The expected output.
    :type expected: bool
    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :returns: None
    :rtype: None
    """
    if env_value is not None:
        monkeypatch.setenv("BASETEN_GPU", env_value)
    assert _has_gpu_key() == expected


@pytest.mark.parametrize(
    "mock_file_contents, mock_env_value, expected",
    [
        (
            {
                "resources": {"accelerator": "test"},
                "repo_id": "test",
                "model_metadata": {"repo_id": "test"},
            },
            "H100",
            True,
        ),
        (
            {
                "resources": {"accelerator": "test"},
                "repo_id": "test",
                "model_metadata": {"repo_id": "test"},
            },
            "A10G",
            True,
        ),
        ({}, "H100", False),
        ({}, "A10G", False),
    ],
)
def test_update_config(monkeypatch, mock_file_contents: dict, mock_env_value: str, expected: bool):
    """Test the _update_config function.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :param mock_file_contents: The contents to be mocked for the config.yaml file.
    :type mock_file_contents: dict
    :param mock_env_value: The value to set for the BASETEN_GPU environment variable.
    :type mock_env_value: str
    :param expected: The expected output.
    :type expected: bool
    :returns: None
    :rtype: None
    """
    monkeypatch.setenv("BASETEN_GPU", mock_env_value)
    mock_open_obj = mock_open(read_data=yaml.dump(mock_file_contents))
    with patch("builtins.open", mock_open_obj):
        assert _update_config() == expected


@pytest.mark.parametrize(
    "mock_env_value, mock_input_value, expected",
    [
        ("H100", b"y\n", True),
        ("H100", b"n\n", True),
        (None, b"y\n", False),
        (None, b"n\n", False),
        (None, b"\n", False),
        (None, KeyboardInterrupt, False),
    ],
)
def test_ensure_gpu(monkeypatch, mock_env_value: str | None, mock_input_value, expected: bool):
    """Test the ensure_gpu function.

    :param monkeypatch: Pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :param mock_env_value: The value to set for the BASETEN_GPU environment variable.
    :type mock_env_value: str | None
    :param mock_input_value: The value to mock for user input.
    :type mock_input_value: bytes | KeyboardInterrupt
    :param expected: The expected output.
    :type expected: bool
    :returns: None
    :rtype: None
    """
    if mock_env_value is not None:
        monkeypatch.setenv("BASETEN_GPU", mock_env_value)
    with patch("builtins.input", side_effect=[mock_input_value]):
        with patch("flow_judge.models.adapters.baseten.gpu._update_config") as mock_update_config:
            mock_update_config.return_value = True
            assert ensure_gpu() == expected
