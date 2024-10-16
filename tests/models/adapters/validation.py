import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from flow_judge.models.adapters.baseten.validation import (
    TIMESTAMP_TOLERANCE_SECONDS,
    validate_baseten_signature,
)

# Mock test data for AsyncPredictResult
mock_async_predict_result_data = {
    "model_config": {},
    "request_id": "mock_request_id",
    "model_id": "mock_model_id",
    "deployment_id": "mock_deployment_id",
    "type": "mock_type",
    "time": datetime.now(timezone.utc),
    "data": {"mock_data": "mock_value"},
    "errors": [{"mock_error": "mock_error_value"}],
}

# Mock webhook secret
mock_webhook_secret = "mock_webhook_secret"


@pytest.fixture(autouse=True)
def mock_os_environ():
    """Mock os.environ for BASETEN_WEBHOOK_SECRET."""
    with patch.dict("os.environ", {"BASETEN_WEBHOOK_SECRET": mock_webhook_secret}):
        yield


class TestValidateBasetenSignature:
    """Test suite for validate_baseten_signature."""

    def test_valid_signature(self):
        """Test for a valid Baseten signature."""
        with patch(
            "flow_judge.models.adapters.baseten.validation.AsyncPredictResult"
        ) as mock_async_predict_result:
            mock_async_predict_result_instance = mock_async_predict_result.return_value
            mock_async_predict_result_instance.model_dump_json.return_value = "mock_model_dump_json"
            mock_async_predict_result_instance.__dict__.update(mock_async_predict_result_data)

            # Mock valid signature
            valid_signature = hmac.digest(
                mock_webhook_secret.encode("utf-8"),
                b"mock_model_dump_json",
                hashlib.sha256,
            ).hex()

            assert (
                validate_baseten_signature(
                    mock_async_predict_result_instance, f"v1={valid_signature}"
                )
                is True
            )

    def test_invalid_signature(self):
        """Test for an invalid Baseten signature."""
        with patch(
            "flow_judge.models.adapters.baseten.validation.AsyncPredictResult"
        ) as mock_async_predict_result:
            mock_async_predict_result_instance = mock_async_predict_result.return_value
            mock_async_predict_result_instance.model_dump_json.return_value = "mock_model_dump_json"
            mock_async_predict_result_instance.__dict__.update(mock_async_predict_result_data)

            # Mock invalid signature
            invalid_signature = "invalid_signature"

            assert (
                validate_baseten_signature(
                    mock_async_predict_result_instance, f"v1={invalid_signature}"
                )
                is False
            )

    def test_stale_timestamp(self):
        """Test for a stale timestamp."""
        stale_timestamp = datetime.now(timezone.utc) - timedelta(
            seconds=TIMESTAMP_TOLERANCE_SECONDS + 1
        )

        with patch(
            "flow_judge.models.adapters.baseten.validation.AsyncPredictResult"
        ) as mock_async_predict_result:
            mock_async_predict_result_instance = mock_async_predict_result.return_value
            mock_async_predict_result_instance.model_dump_json.return_value = "mock_model_dump_json"
            mock_async_predict_result_instance.__dict__.update(mock_async_predict_result_data)
            mock_async_predict_result_instance.time = stale_timestamp

            # Mock valid signature
            valid_signature = hmac.digest(
                mock_webhook_secret.encode("utf-8"),
                b"mock_model_dump_json",
                hashlib.sha256,
            ).hex()

            assert (
                validate_baseten_signature(
                    mock_async_predict_result_instance, f"v1={valid_signature}"
                )
                is False
            )

    def test_missing_webhook_secret(self, monkeypatch):
        """Test for a missing BASETEN_WEBHOOK_SECRET environment variable."""
        monkeypatch.delenv("BASETEN_WEBHOOK_SECRET", raising=False)

        with patch(
            "flow_judge.models.adapters.baseten.validation.AsyncPredictResult"
        ) as mock_async_predict_result:
            mock_async_predict_result_instance = mock_async_predict_result.return_value
            mock_async_predict_result_instance.model_dump_json.return_value = "mock_model_dump_json"
            mock_async_predict_result_instance.__dict__.update(mock_async_predict_result_data)

            # Mock valid signature
            valid_signature = hmac.digest(
                mock_webhook_secret.encode("utf-8"),
                b"mock_model_dump_json",
                hashlib.sha256,
            ).hex()

            assert (
                validate_baseten_signature(
                    mock_async_predict_result_instance, f"v1={valid_signature}"
                )
                is False
            )
