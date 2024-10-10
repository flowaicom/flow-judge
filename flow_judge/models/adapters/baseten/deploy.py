import http
import logging
import os
from typing import Optional, List, Dict

import requests
import truss
import getpass

from truss.remote import remote_factory
from truss.remote.baseten.error import ApiError

logger = logging.getLogger(__name__)


def _is_interactive() -> bool:
    import sys
    return sys.__stdin__.isatty()


def _initialize_model() -> bool:
    if _is_flowjudge_deployed():
        logger.info("Flow Judge already deployed")
        return True

    truss_path = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
    try:
        truss.push(truss_path, promote=True, publish=False, trusted=True)
        logger.info("Flow Judge Baseten deployment successful")
        return True
    except ApiError:
        pass
    except ValueError:
        pass

    logger.error("Flow Judge Baseten deployment failed. Ensure that provided API key is correct and try again.")


def _get_baseten_api_key() -> Optional[str]:
    api_key = os.environ.get("BASETEN_API_KEY")
    if api_key is None or api_key == "":
        # Try trussrc file
        try:
            c = remote_factory.RemoteFactory.load_remote_config("baseten")
        except Exception:
            return None

        if c is not None:
            api_key = c.configs["api_key"]
            os.environ["BASETEN_API_KEY"] = api_key

    return api_key


def _validate_auth_status():
    api_key = _get_baseten_api_key()
    if api_key is None:
        return False

    response = requests.request(
        "GET",
        "https://api.baseten.co/v1/models",
        headers={"Authorization": f"Api-Key {api_key}"},
        json={}
    )

    if response.status_code != http.HTTPStatus.OK:
        return False

    return True


def _get_models() -> Optional[List[Dict]]:
    api_key = _get_baseten_api_key()

    response = requests.request(
        "GET",
        "https://api.baseten.co/v1/models",
        headers={"Authorization": f"Api-Key {api_key}"},
        json={}
    )

    if response.status_code != http.HTTPStatus.OK:
        return None

    resp = response.json()
    if resp is None or type(resp) is not dict or "models" not in resp:
        return None

    return resp["models"]


def _is_flowjudge_deployed() -> bool:
    models = _get_models()

    if models is None:
        return False

    return any("Flow-Judge" in model["name"] for model in models)


def _authenticate_truss_with_env_api_key():
    truss_authenticated = False
    try:
        c = remote_factory.RemoteFactory.load_remote_config("baseten")
        truss_authenticated = c.configs["api_key"] is not None
    except Exception:
        pass

    env_api_key = os.environ.get("BASETEN_API_KEY")
    if env_api_key is not None and env_api_key != "" and not truss_authenticated:
        truss.login(env_api_key)


def get_deployed_model_id() -> Optional[str]:
    models = _get_models()
    if models is None:
        return None

    for model in models:
        if "Flow-Judge" in model["name"]:
            return model["id"]

    return None


def ensure_model_deployment():
    # In case user already provided API key in env variable
    _authenticate_truss_with_env_api_key()

    if _validate_auth_status():
        logger.info("Baseten authenticated")
        return _initialize_model()

    print("To run Flow Judge remotely with Baseten, signup and generate API key")
    print(" ➡️ Signup: https://app.baseten.co/signup")
    print(" ➡️ API keys: https://app.baseten.co/settings/api_keys")
    print(" ➡️ Docs: https://docs.baseten.co/quickstart#setup\n")

    # Not interactive environment (eg. notebook)
    if not _is_interactive():
        print("Set the Baseten API key in `BASETEN_API_KEY` environment variable and run again:")
        print("```")
        print("os.environ[\"BASETEN_API_KEY\"] = \"«your API key»\"")
        print("```")
        return False

    # Interactive
    while True:
        key = getpass.getpass("Baseten API key (hidden): ")
        if key is None or len(key) < 5:
            print("Baseten API key seems to be incorrect. The key must be at least 5 characters long")
            continue

        truss.login(key)
        if not _validate_auth_status():
            print("Invalid Baseten API key, try again.")
            continue

        return _initialize_model()
