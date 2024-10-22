import logging
import os
from typing import TypedDict

import requests
import truss
from truss.api.definitions import ModelDeployment
from truss.remote.baseten.error import ApiError

from flow_judge.models.adapters.baseten.management import sync_set_scale_down

from .api_auth import ensure_baseten_authentication, get_baseten_api_key
from .gpu import ensure_gpu
from .webhook import ensure_baseten_webhook_secret

logger: logging.Logger = logging.getLogger(__name__)


class ModelInfo(TypedDict):
    """Type definition for model information returned by Baseten API.

    :ivar id: The unique identifier of the model.
    :ivar name: The name of the model.
    """

    id: str
    name: str


def _initialize_model() -> bool:
    """Initialize the Flow Judge model.

    :return: True if initialization is successful, False otherwise.
    :rtype: bool
    """
    logger.info("Initializing Flow Judge model")
    if _is_flowjudge_deployed():
        logger.info("Flow Judge already deployed")
        return True

    if not ensure_gpu():
        logger.error("BASETEN_GPU environment variable is required." " Set one of: H100 or A10G")
        return False

    truss_path: str = f"{os.path.dirname(os.path.realpath(os.path.abspath(__file__)))}/deployment"
    logger.debug(f"Truss path: {truss_path}")
    try:
        deployment: ModelDeployment = truss.push(
            truss_path, promote=True, trusted=True, environment=None
        )
        logger.debug("Waiting for deployment to become active")
        deployment.wait_for_active()
        logger.info("Flow Judge Baseten deployment successful")

        model_id = get_deployed_model_id()
        api_key = get_baseten_api_key()
        if model_id and api_key:
            has_updated_scale_down = sync_set_scale_down(
                scale_down_delay=120, api_key=api_key, model_id=model_id
            )
            if has_updated_scale_down:
                logger.info(
                    "Successfully updated Baseten deployed model scale down delay to 2 mins."
                )
            else:
                logger.info(
                    "Unable to update Baseten deployed model scale down delay period."
                    " Continuing with default"
                )

        return True
    except ApiError as e:
        logger.error(
            "Flow Judge Baseten deployment failed. "
            f"Ensure that provided API key is correct and try again. {e.message}"
        )
    except ValueError as e:
        logger.error(
            "Flow Judge Baseten deployment failed. "
            f"Ensure that provided API key is correct and try again. {str(e)}"
        )
    return False


def _get_models() -> list[ModelInfo] | None:
    """Fetch the list of models from Baseten API.

    :return: List of ModelInfo if successful, None otherwise.
    :rtype: Optional[List[ModelInfo]]
    """
    logger.debug("Fetching models from Baseten API")
    api_key: str | None = get_baseten_api_key()
    if not api_key:
        logger.warning("No API key available to fetch models")
        return None

    try:
        response: requests.Response = requests.get(
            "https://api.baseten.co/v1/models",
            headers={"Authorization": f"Api-Key {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        resp: dict[str, list[ModelInfo]] = response.json()
        models: list[ModelInfo] | None = resp.get("models")
        logger.debug(f"Fetched {len(models) if models else 0} models")
        return models
    except requests.RequestException as e:
        logger.error(f"Error fetching models: {e}")
        return None


def _is_flowjudge_deployed() -> bool:
    """Check if Flow Judge is already deployed.

    :return: True if Flow Judge is deployed, False otherwise.
    :rtype: bool
    """
    logger.debug("Checking if Flow Judge is deployed")
    models: list[ModelInfo] | None = _get_models()

    if models is None:
        logger.warning("Unable to determine if Flow Judge is deployed")
        return False

    is_deployed: bool = any("Flow-Judge" in model["name"] for model in models)
    logger.debug(f"Flow Judge deployed: {is_deployed}")
    return is_deployed


def get_deployed_model_id() -> str | None:
    """Get the ID of the deployed Flow Judge model.

    :return: The model ID if found, None otherwise.
    :rtype: Optional[str]
    """
    logger.debug("Getting deployed Flow Judge model ID")
    models: list[ModelInfo] | None = _get_models()
    if not models:
        logger.warning("No models found")
        return None

    for model in models:
        if "Flow-Judge" in model["name"]:
            logger.debug(f"Found Flow Judge model with ID: {model['id']}")
            return model["id"]

    logger.warning("Flow Judge model not found")
    return None


def ensure_model_deployment() -> bool:
    """Ensure Flow Judge model deployment to Baseten.

    :return: True if deployment is successful, False otherwise.
    :rtype: bool
    """
    logger.info("Ensuring Flow Judge model deployment")
    if not ensure_baseten_authentication():
        logger.error("Baseten not authenticated, interrupting model deployment")
        return False

    ensure_baseten_webhook_secret(optional=True)

    return _initialize_model()
