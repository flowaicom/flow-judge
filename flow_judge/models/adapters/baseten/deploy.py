import getpass
import http
import logging
import os
from typing import TypedDict

import requests
import truss
from truss.remote import remote_factory
from truss.remote.baseten.error import ApiError

from .gpu import ensure_gpu

logger: logging.Logger = logging.getLogger(__name__)


class ModelInfo(TypedDict):
    """Type definition for model information returned by Baseten API.

    :ivar id: The unique identifier of the model.
    :ivar name: The name of the model.
    """

    id: str
    name: str


def _is_interactive() -> bool:
    """Check if the current environment is interactive.

    :return: True if the environment is interactive, False otherwise.
    :rtype: bool
    """
    import sys

    return sys.__stdin__.isatty()


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
        deployment: truss.Deployment = truss.push(
            truss_path, promote=True, trusted=True, environment=None
        )
        logger.debug("Waiting for deployment to become active")
        deployment.wait_for_active()
        logger.info("Flow Judge Baseten deployment successful")
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


def _get_baseten_api_key() -> str | None:
    """Retrieve the Baseten API key from environment or config file.

    :return: The Baseten API key if found, None otherwise.
    :rtype: Optional[str]
    """
    logger.debug("Attempting to retrieve Baseten API key")
    api_key: str | None = os.environ.get("BASETEN_API_KEY")
    if api_key:
        logger.debug("API key found in environment variables")
        if len(api_key) != 41:
            logger.warning(
                "Warning: Baseten API key might be incorrect. "
                "The length should be exactly 41 characters."
            )
        return api_key

    logger.debug("API key not found in environment, checking remote config")
    try:
        c: remote_factory.RemoteConfig | None = remote_factory.RemoteFactory.load_remote_config(
            "baseten"
        )
        if c is not None:
            api_key = c.configs.get("api_key")
            if api_key:
                logger.debug("API key found in remote config")
                if len(api_key) != 41:
                    logger.warning(
                        "Warning: Baseten API key might be incorrect. "
                        "The length should be exactly 41 characters."
                    )
                os.environ["BASETEN_API_KEY"] = api_key
                return api_key
    except Exception as e:
        logger.error(f"Error loading remote config: {e}")

    logger.debug("API key not found")
    return None


def _validate_auth_status() -> bool:
    """Validate the authentication status with Baseten.

    :return: True if authentication is successful, False otherwise.
    :rtype: bool
    """
    logger.debug("Validating authentication status")
    api_key: str | None = _get_baseten_api_key()
    if not api_key:
        logger.warning("No API key available for authentication")
        return False

    try:
        response: requests.Response = requests.get(
            "https://api.baseten.co/v1/models",
            headers={"Authorization": f"Api-Key {api_key}"},
            timeout=10,
        )
        is_valid: bool = response.status_code == http.HTTPStatus.OK
        logger.debug(f"Authentication status: {'valid' if is_valid else 'invalid'}")
        return is_valid
    except requests.RequestException as e:
        logger.error(f"Error validating auth status: {e}")
        return False


def _get_models() -> list[ModelInfo] | None:
    """Fetch the list of models from Baseten API.

    :return: List of ModelInfo if successful, None otherwise.
    :rtype: Optional[List[ModelInfo]]
    """
    logger.debug("Fetching models from Baseten API")
    api_key: str | None = _get_baseten_api_key()
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


def _authenticate_truss_with_env_api_key() -> None:
    """Authenticate Truss with the API key from environment variable.

    :return: None
    """
    logger.debug("Authenticating Truss with environment API key")
    truss_authenticated: bool = False
    try:
        c: remote_factory.RemoteConfig | None = remote_factory.RemoteFactory.load_remote_config(
            "baseten"
        )
        truss_authenticated = c is not None and c.configs.get("api_key") is not None
    except Exception as e:
        logger.error(f"Error loading Truss remote config: {e}")

    env_api_key: str | None = os.environ.get("BASETEN_API_KEY")
    if env_api_key and not truss_authenticated:
        logger.debug("Logging in to Truss with environment API key")
        truss.login(env_api_key)
    else:
        logger.debug("Truss already authenticated or no API key available")


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
    _authenticate_truss_with_env_api_key()

    if _validate_auth_status():
        logger.info("Baseten authenticated")
        return _initialize_model()

    logger.warning("Baseten authentication failed")
    print("To run Flow Judge remotely with Baseten, signup and generate API key")
    print(" ➡️ Signup: https://app.baseten.co/signup")
    print(" ➡️ API keys: https://app.baseten.co/settings/api_keys")
    print(" ➡️ Docs: https://docs.baseten.co/quickstart#setup\n")

    if not _is_interactive():
        logger.info("Non-interactive environment detected")
        print("Set the Baseten API key in `BASETEN_API_KEY` environment variable " "and run again:")
        print("```")
        print('os.environ["BASETEN_API_KEY"] = "«your API key»"')
        print("```")
        return False

    logger.info("Prompting for API key in interactive environment")
    while True:
        key: str = getpass.getpass("Baseten API key (hidden): ")
        if not key:
            logger.warning("Empty API key entered")
            print("Input is empty, try again.")
            continue

        if len(key) != 41:
            logger.warning(
                "Warning: Baseten API key might be incorrect. "
                "The length should be exactly 41 characters."
            )

        try:
            logger.debug("Attempting to log in with provided API key")
            truss.login(key)
            if _validate_auth_status():
                logger.info("Login successful")
                return _initialize_model()
            logger.warning("Invalid API key")
            print("Invalid Baseten API key, try again.")
        except Exception as e:
            logger.error(f"Error during login: {e}")
            print("An error occurred during login. Please try again.")

    return False
