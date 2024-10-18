import logging
import os
from enum import Enum
from typing import Any

import yaml

from .util import is_interactive

logger = logging.getLogger(__name__)


class ModelByGPU(Enum):
    """Select the appropriate model based on GPU."""

    H100_40GB = "flowaicom/Flow-Judge-v0.1-FP8"
    A10G = "flowaicom/Flow-Judge-v0.1-AWQ"


def _get_gpu_key() -> str | None:
    """Fetches the BASETEN_GPU environment variable.

    :returns: The value of the variable: one of H100, A10G
    :rtype: str | None
    """
    gpu: str | None = os.environ.get("BASETEN_GPU")

    if gpu:
        if gpu.lower() not in ["h100", "a10g"]:
            raise ValueError("BASETEN_GPU option is incorrect." "Possible options: H100, A10G")

    gpu = "H100_40GB" if gpu.lower() == "h100" else gpu

    return gpu


def _has_gpu_key() -> bool:
    """Verifies that the BASETEN_GPU environment variable is present.

    :returns: True if found, False otherwise.
    :rtype: bool
    """
    gpu: str | None = _get_gpu_key()

    return True if gpu else False


def _update_config() -> bool:
    """Update the config.yaml file with the GPU & flow-judge model id.

    :returns: True if successfully updated, False if not.
    :rtype: bool
    """
    gpu: str = _get_gpu_key()

    if not gpu:
        return False

    config_path: str = os.path.join(
        os.path.dirname(os.path.realpath(os.path.abspath(__file__))), "deployment", "config.yaml"
    )

    try:
        with open(config_path) as file:
            data: dict[str, Any] = yaml.safe_load(file)

        data["resources"]["accelerator"] = ModelByGPU[gpu.upper()].name
        data["repo_id"] = ModelByGPU[gpu.upper()].value
        data["model_metadata"]["repo_id"] = ModelByGPU[gpu.upper()].value

        with open(config_path, "w") as file:
            yaml.safe_dump(data, file, default_flow_style=False)

        return True

    except FileNotFoundError:
        logger.error(f"Baseten config.yaml file not found on path {config_path}")
        return False
    except yaml.YAMLError as e:
        logger.error(f"Error: Failed to parse the Baseten config file. {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred with Baseten config file update: {e}")
        return False


def ensure_gpu() -> bool:
    """Enable GPU selection for FlowJudge model deployment.

    :return: True if successfully updated, False otherwise
    :rtype: bool
    """
    if _has_gpu_key():
        return _update_config()

    if is_interactive():
        print("What GPU on Baseten should we deploy the FlowJudge model to?")
        print(" ➡️ H100")
        print(" ➡️ A10G: default")
        print("Would you like to switch your deployment to H100?")
        print("y/n?\n")

    else:
        logger.info("Non-interactive environment detected")
        print("What GPU on Baseten should we deploy the FlowJudge model to?")
        print(" ➡️ H100")
        print(" ➡️ A10G")
        print("Please set the environment variable to the appropriate value:")
        print("```")
        print('os.environ["BASETEN_GPU"] = "<<H100_or_A10G>>"')
        print("```")
        return False

    logger.info("Prompting for GPU in interactive environment")
    while True:
        upgrade: str = input()
        if not upgrade:
            logger.warning("Empty option entered")
            print("Input is empty, try again.")
            continue

        if upgrade.lower() not in ["yes", "y", "n", "no"]:
            logger.warning("Incorrect option selected")
            print("Incorrect option, select one from y/n")
            continue

        if upgrade.lower() in ["yes", "y"]:
            os.environ["BASETEN_GPU"] = ModelByGPU.H100_40GB.name
            return _update_config()

        if upgrade.lower() in ["n", "no"]:
            os.environ["BASETEN_GPU"] = ModelByGPU.A10G.name
            return _update_config()
