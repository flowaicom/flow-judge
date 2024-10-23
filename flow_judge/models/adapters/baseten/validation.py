import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, JsonValue

logger = logging.getLogger(__name__)

TIMESTAMP_TOLERANCE_SECONDS = 300


class AsyncPredictResult(BaseModel):
    """Baseten completion response format."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    request_id: str
    model_id: str
    deployment_id: str
    type: str
    time: datetime
    data: JsonValue
    errors: list[dict]


def validate_baseten_signature(result, actual_signatures) -> bool:
    """Webhook signature validation from baseten."""
    try:
        webhook_secret = os.environ["BASETEN_WEBHOOK_SECRET"]
    except Exception as e:
        logger.error(
            "Baseten webhook secret is not in the environment."
            "Unable to validate baseten signature for batched requests."
            "Set the BASETEN_WEBHOOK_SECRET env variable to proceed."
            f"{e}"
        )
        return False

    async_predict_result = AsyncPredictResult(**result)

    if (
        datetime.now(timezone.utc) - async_predict_result.time
    ).total_seconds() > TIMESTAMP_TOLERANCE_SECONDS:
        logger.error(
            f"Async predict result was received after {TIMESTAMP_TOLERANCE_SECONDS} seconds"
            "and is considered stale, Baseten signature was not validated."
        )
        return False

    for actual_signature in actual_signatures.replace("v1=", "").split(","):
        expected_signature = hmac.digest(
            webhook_secret.encode("utf-8"),
            async_predict_result.model_dump_json().encode("utf-8"),
            hashlib.sha256,
        ).hex()

        if hmac.compare_digest(expected_signature, actual_signature):
            logger.info("Baseten signature is valid!")
            return True

    logger.error(
        "Baseten signature is not valid. Ensure your webhook secrets are properly configured."
    )
    return False
