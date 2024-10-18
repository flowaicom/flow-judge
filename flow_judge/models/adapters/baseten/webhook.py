import getpass
import logging
import os
import re

from .util import is_interactive

logger: logging.Logger = logging.getLogger(__name__)

_stored_secret_path: str = "~/.config/flow-judge/baseten_webhook_secret"
_stored_skip_file: str = "~/.config/flow-judge/baseten_whsec_skipped"


def _is_valid_secret(secret: str) -> bool:
    """Validate the Baseten webhook secret.

    :param secret: The secret to validate
    :return: True if the secret is valid, False otherwise
    """
    logger.debug("Validating webhook secret")
    return bool(re.match(r"^whsec_[a-zA-Z0-9]{40}$", secret))


def _save_webhook_secret(secret: str) -> None:
    """Save the webhook secret to a file.

    :param secret: The secret to save
    """
    logger.debug(f"Saving webhook secret to {_stored_secret_path}")
    try:
        p: str = os.path.expanduser(_stored_secret_path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(secret)
        logger.info("Webhook secret saved successfully")
    except OSError as e:
        logger.error(f"Failed to save webhook secret: {e}")


def _get_stored_secret() -> str | None:
    """Retrieve the stored webhook secret.

    :return: The stored secret if available, None otherwise
    """
    logger.debug(f"Attempting to retrieve stored webhook secret from {_stored_secret_path}")
    try:
        p: str = os.path.expanduser(_stored_secret_path)
        if not os.path.exists(p):
            logger.info("No stored webhook secret found")
            return None
        with open(p) as f:
            secret: str = f.read().strip()
        logger.info("Stored webhook secret retrieved successfully")
        return secret
    except OSError as e:
        logger.error(f"Failed to retrieve stored webhook secret: {e}")
        return None


def _save_skip_file() -> None:
    """Creates a flag file indicating that user requested to skip secret webhook input."""
    os.makedirs(os.path.expanduser(os.path.dirname(_stored_skip_file)), exist_ok=True)
    open(os.path.expanduser(_stored_skip_file), "a").close()


def _handle_skip() -> bool:
    """Handles a possible user request to skip question about webhook secret.

    Checks for existence of BASETEN_SKIP_WEBHOOK_SECRET env variable, indicating that
    skip file should be saved. Skip file existence is a flag telling us to skip the
    question about webhook secret.

    :return: True if the webhook secret prompt should be skipped, False otherwise
    :rtype: bool
    """
    if os.environ.get("BASETEN_SKIP_WEBHOOK_SECRET"):
        logger.info(
            "User requested to skip the webhook secret prompt, skipping and saving skip file. "
            "Remove ~/.config/flow-judge/baseten_whsec_skipped to restore the prompt."
        )
        _save_skip_file()
        return True

    if os.path.exists(os.path.expanduser(_stored_skip_file)):
        logger.info(
            "Webhook secret not required and user skipped it before, skipping. "
            "Remove ~/.config/flow-judge/baseten_whsec_skipped to restore the prompt."
        )
        return True

    return False


def _handle_env_variable_input() -> bool:
    """Checks if webhook secret was provided in environment variable.

    :return: True if the webhook secret was provided, False otherwise
    :rtype: bool
    """
    env_secret: str | None = os.getenv("BASETEN_WEBHOOK_SECRET")
    if not env_secret:
        return False

    logger.info("Found BASETEN_WEBHOOK_SECRET in environment variables")
    if _is_valid_secret(env_secret):
        logger.info("Environment variable contains a valid webhook secret")
        _save_webhook_secret(env_secret)
    else:
        logger.warning("Probably invalid BASETEN_WEBHOOK_SECRET in environment variable")
    return True


def _handle_stored_secret() -> bool:
    """Checks if webhook secret was previously provided and stored in file.

    :return: True if the webhook secret was provided, False otherwise
    :rtype: bool
    """
    stored_secret: str | None = _get_stored_secret()
    if not stored_secret:
        return False

    logger.info("Found stored webhook secret")
    if _is_valid_secret(stored_secret):
        logger.info("Stored webhook secret is valid")
        os.environ["BASETEN_WEBHOOK_SECRET"] = stored_secret
    else:
        logger.warning("Stored webhook secret is probably invalid")
    return True


def _prompt_interactively(optional: bool) -> str | None:
    """Asks user to enter webhook secret or skip if optional.

    :param optional: True if the input can be skipped by user
    :return: User-provided webhook secret or None if skipped
    :rtype: str | None
    """
    while True:
        secret: str = getpass.getpass(
            "Baseten webhook secret (hidden): "
            if not optional
            else "Baseten webhook secret (hidden; leave empty to skip): "
        )

        if optional and not secret:
            logger.info("User skipped optional webhook secret input")
            return None

        if not secret:
            logger.warning("Empty input received")
            print("Input is empty, please try again.")
            continue

        return secret


def _print_general_prompt(optional: bool) -> None:
    """Prints general information/prompt about webhook secret input requirement.

    :param optional: Whether to print the information that the input is optional
    """
    print(
        "To run Flow Judge remotely with Baseten and enable async execution, "
        "you need to create and configure a webhook secret.\n"
        "➡️ Creating a webhook secret: https://docs.baseten.co/invoke/async-secure\n\n"
        "The webhook secret is used to validate that the webhook responses originated "
        "from Baseten. "
        f"It will be stored in {_stored_secret_path} for later use.\n"
        "For your convenience, Baseten responses are forwarded to you using Flow AI proxy.\n"
        "Explore what this means and the alternatives here: "
        "https://github.com/flowaicom/flow-judge/\n"
    )
    if optional:
        print("\033[1mOptional. This is  only required if you plan async execution.\033[0m")


def _print_noninteractive_prompt(optional: bool) -> None:
    """Prints a prompt for non interactive environments.

    :param optional: Whether to print the information that the input is optional
    """
    logger.info("Non-interactive environment detected")
    print(
        "Set the Baseten webhook secret in the BASETEN_WEBHOOK_SECRET "
        "environment variable and run again:\n"
        'os.environ["BASETEN_WEBHOOK_SECRET"] = "«your webhook secret»"\n'
        "The secret should start with 'whsec_' followed by 40 alphanumeric characters.\n"
    )
    if optional:
        print(
            "If you don't want to see this message in the future set the "
            "BASETEN_SKIP_WEBHOOK_SECRET environment variable to any non-empty value:\n"
            'os.environ["BASETEN_SKIP_WEBHOOK_SECRET"]="true"\n'
        )


def ensure_baseten_webhook_secret(optional: bool = False) -> bool:
    """Ensure a Baseten webhook secret is available.

    This function checks for a webhook secret in the following order:
    1. Environment variable
    2. Stored secret file
    3. User input (in interactive mode)

    :param optional: Whether allow the user to omit the webhook secret input
    :return: True if a secret is available (valid or not), False if no secret could be obtained
    """
    # If input optional, check if user requested to skip the prompt
    if optional and _handle_skip():
        return False

    logger.info("Ensuring Baseten webhook secret")

    # Check environment variable
    if _handle_env_variable_input():
        return True

    # Check stored secret
    if _handle_stored_secret():
        return True

    logger.info("No existing webhook secret found, prompting user")
    _print_general_prompt(optional)

    # In non-interactive environments we return early because their inputs (env vars)
    # are processed at the very beginning of this function.
    if not is_interactive():
        _print_noninteractive_prompt(optional)
        return False

    logger.info("Prompting user for webhook secret interactively")
    secret = _prompt_interactively(optional)

    # Secret not provided which means user asked to skip the prompt
    if secret is None:
        print("(skipped)")
        _save_skip_file()
        return False

    if not _is_valid_secret(secret):
        logger.warning("Invalid webhook secret provided")
        print(
            "Warning: The provided webhook secret is probably invalid. "
            "It should start with 'whsec_' followed by 40 alphanumeric characters. "
            "Proceeding anyway."
        )

    _save_webhook_secret(secret)
    os.environ["BASETEN_WEBHOOK_SECRET"] = secret

    return True
