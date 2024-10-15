import getpass
import logging
import os
import re

from flow_judge.models.adapters.baseten.deploy import _is_interactive

logger: logging.Logger = logging.getLogger(__name__)

_stored_secret_path: str = "~/.config/flow-judge/baseten_webhook_secret"


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


def ensure_baseten_webhook_secret() -> bool:
    """Ensure a Baseten webhook secret is available.

    This function checks for a webhook secret in the following order:
    1. Environment variable
    2. Stored secret file
    3. User input (in interactive mode)

    :return: True if a secret is available (valid or not), False if no secret could be obtained
    """
    logger.info("Ensuring Baseten webhook secret")

    # Check environment variable
    env_secret: str | None = os.getenv("BASETEN_WEBHOOK_SECRET")
    if env_secret:
        logger.info("Found BASETEN_WEBHOOK_SECRET in environment variables")
        if _is_valid_secret(env_secret):
            logger.info("Environment variable contains a valid webhook secret")
            _save_webhook_secret(env_secret)
        else:
            logger.warning("Invalid BASETEN_WEBHOOK_SECRET in environment variable")
        return True

    # Check stored secret
    stored_secret: str | None = _get_stored_secret()
    if stored_secret:
        logger.info("Found stored webhook secret")
        if _is_valid_secret(stored_secret):
            logger.info("Stored webhook secret is valid")
            os.environ["BASETEN_WEBHOOK_SECRET"] = stored_secret
        else:
            logger.warning("Stored webhook secret is invalid")
        return True

    logger.info("No existing webhook secret found, prompting user")
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

    if not _is_interactive():
        logger.info("Non-interactive environment detected")
        print(
            "Set the Baseten webhook secret in the BASETEN_WEBHOOK_SECRET "
            "environment variable and run again:\n"
            'os.environ["BASETEN_WEBHOOK_SECRET"] = "«your webhook secret»"\n'
            "The secret should start with 'whsec_' followed by 40 alphanumeric characters."
        )
        return False

    logger.info("Prompting user for webhook secret")
    while True:
        secret: str = getpass.getpass("Baseten webhook secret (hidden): ")
        if not secret:
            logger.warning("Empty input received")
            print("Input is empty, please try again.")
            continue

        if _is_valid_secret(secret):
            logger.info("Valid webhook secret entered")
            _save_webhook_secret(secret)
            os.environ["BASETEN_WEBHOOK_SECRET"] = secret
            return True
        else:
            logger.warning("Invalid webhook secret entered")
            print(
                "Warning: Invalid webhook secret. It should start with 'whsec_' "
                "followed by 40 alphanumeric characters. Proceeding anyway."
            )
            _save_webhook_secret(secret)
            os.environ["BASETEN_WEBHOOK_SECRET"] = secret
            return True

    return False
