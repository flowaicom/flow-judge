import getpass
import http
import logging
import os

import requests
import truss
from truss.remote import remote_factory

from .util import is_interactive

logger: logging.Logger = logging.getLogger(__name__)


def get_baseten_api_key() -> str | None:
    """Retrieve the Baseten API key from environment or config file.

    :return: The Baseten API key if found, None otherwise.
    :rtype: str | None
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


def _validate_auth_status(api_key: str | None = None) -> bool:
    """Validate the authentication status with Baseten.

    :param api_key: Optional API key. If None, it will try to obtain API key from env or .trussrc.
    :return: True if authentication is successful, False otherwise.
    :rtype: bool
    """
    logger.debug("Validating authentication status")

    api_key: str | None = api_key or get_baseten_api_key()
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


def _attempt_truss_auth_with_env_api_key() -> None:
    """Authenticate Truss with the API key from environment variable.

    :return: None
    """
    logger.debug("Authenticating Truss with environment API key")
    env_api_key: str | None = os.environ.get("BASETEN_API_KEY")
    if not env_api_key:
        logger.debug("Baseten API key not available in env var")
        return

    logger.debug("Logging in to Truss with environment API key")
    truss.login(env_api_key)


def _print_noninteractive_prompt() -> None:
    """Print a API key prompt for non-interactive environments."""
    print("Set the Baseten API key in `BASETEN_API_KEY` environment variable " "and run again:")
    print("```")
    print('os.environ["BASETEN_API_KEY"] = "«your API key»"')
    print("```")


def _print_general_prompt() -> None:
    """Print a API key prompt information."""
    print("To run Flow Judge remotely with Baseten, signup and generate API key")
    print(" ➡️ Signup: https://app.baseten.co/signup")
    print(" ➡️ API keys: https://app.baseten.co/settings/api_keys")
    print(" ➡️ Docs: https://docs.baseten.co/quickstart#setup\n")


def _validate_entered_key(key: str) -> bool:
    """Checks if the key entered by the user in interactive environments is valid.

    :param key: The key entered by the user
    :return: True if the key entered by the user is valid, False otherwise.
    :rtype: bool
    """
    try:
        logger.debug("Attempting to log in with provided API key")
        truss.login(key)
    except Exception as e:
        logger.error(f"Error during login: {e}")
        print("An error occurred during login. Please try again.")
        return False

    if not _validate_auth_status(key):
        logger.warning("Invalid Baseten API key")
        print("Invalid Baseten API key, try again.")
        return False

    logger.info("Baseten authentication successful")
    return True


def ensure_baseten_authentication() -> bool:
    """Attempts to obtain and validate the Baseten API key from the user.

    :return: True if authentication is successful, False otherwise.
    """
    # Checks if truss is already authenticated or if the key is provided in the env to authenticate
    _attempt_truss_auth_with_env_api_key()

    # Checks if authentication above succeeded
    if _validate_auth_status():
        logger.info("Baseten authenticated")
        return True

    logger.warning("Baseten authentication failed or not initialized")
    _print_general_prompt()

    # In non-interactive environments we return early because their inputs (env vars)
    # are processed at the very beginning of this function.
    if not is_interactive():
        logger.info("Non-interactive environment detected")
        _print_noninteractive_prompt()
        return False

    logger.info("Prompting for API key in interactive environment")
    while True:
        key: str = getpass.getpass("Baseten API key (hidden): ")
        if not key:
            logger.warning("Empty API key entered")
            print("Input is empty, try again.")
            continue

        print(key)

        if len(key) != 41:
            logger.warning(
                "Warning: Baseten API key might be incorrect. "
                "The length should be exactly 41 characters."
            )

        if _validate_entered_key(key):
            logger.info("Login successful")
            return True

    return False
