import logging
import os
import getpass
from typing import Optional

from flow_judge.models.adapters.baseten.deploy import _is_interactive

logger = logging.getLogger(__name__)

_stored_secret_path = "~/.config/flow-judge/baseten_webhook_secret"


def _save_webhook_secret(secret):
    p = os.path.expanduser(_stored_secret_path)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p))

    with open(p, "w+") as f:
        f.write(secret)


def _get_stored_secret() -> Optional[str]:
    if not os.path.exists(os.path.expanduser(_stored_secret_path)):
        return None

    with open(os.path.expanduser(_stored_secret_path)) as f:
        return f.read()


def ensure_baseten_webhook_secret():
    # In case user already provided API key in env variable
    if os.getenv("BASETEN_WEBHOOK_SECRET", "") != "":
        _save_webhook_secret(os.getenv("BASETEN_WEBHOOK_SECRET"))
        return True

    stored_secret = _get_stored_secret()
    if stored_secret is not None:
        os.environ["BASETEN_WEBHOOK_SECRET"] = stored_secret
        return True

    print(
        "To run Flow Judge remotely with Baseten \033[1mand enable async execution\033[0m,"
        "you need to create and configure a webhook secret\n"
        " ➡️ Creating a webhook secret: https://docs.baseten.co/invoke/async-secure\n\n"

        "The webhook secret is used to validate that the webhook responses originated from Baseten. "
        f"It never leaves your computer. It will be stored in plain text in {_stored_secret_path} for later use.\n"
        "For your convenience, Baseten responses are forwarded to your device using Flow AI proxy server.\n"
        "Explore what this means and the alternatives here: https://github.com/flowaicom/flow-judge/\n"
    )

    # Not interactive environment (eg. notebook)
    if not _is_interactive():
        print("Set the Baseten webhook secret in `BASETEN_WEBHOOK_SECRET` environment variable and run again:")
        print("```")
        print("os.environ[\"BASETEN_WEBHOOK_SECRET\"] = \"«your webhook secret»\"")
        print("```")
        return False

    # Interactive
    while True:
        secret = getpass.getpass("Baseten webhook secret (hidden): ")
        if secret is None:
            print("Input is empty, try again.")
            continue

        if len(secret) > 46 or not secret.startswith("whsec_"):
            logger.warning(
                "Warning: Baseten webhook secret might be incorrect. "
                "The length should not exceed 46 characters and it should start with `whsec_`."
            )

        _save_webhook_secret(secret)

        return True
