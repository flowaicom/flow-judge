import json

import aiohttp
import requests
import structlog

logger = structlog.get_logger(__name__)


def _get_management_base_url(model_id: str) -> str:
    """Get the base URL for the Management API.

    :param model_id: The ID of the deployed model.
    :returns: The URL for the management endpoints
    """
    return f"https://api.baseten.co/v1/models/{model_id}/deployments/production"


def sync_set_scale_down(scale_down_delay: int, api_key: str, model_id: str) -> bool:
    """Syncronous update to the cooldown period for the deployed model.

    :param scale_down_delay: The cooldown period in seconds.
    :param api_key: The Baseten API key.
    :param model_id: The ID of the deployed model on Baseten.
    :returns bool: True if successful, False otherwise.
    """
    try:
        resp = requests.patch(
            f"{_get_management_base_url(model_id)}/autoscaling_settings",
            headers={"Authorization": f"Api-Key {api_key}"},
            json={"scale_down_delay": scale_down_delay},
        )

        re = resp.json()
        if "status" in re:
            return True if re["status"] in ["ACCEPTED", "UNCHANGED", "QUEUED"] else False
    except Exception as e:
        logger.error("Unexpected error occurred with Baseten scale down delay request" f" {e}")
        return False


async def set_scale_down_delay(scale_down_delay: int, api_key: str, model_id: str) -> bool:
    """Dynamically updates the cooldown period for the deployed model.

    :param scale_down_delay: The cooldown period in seconds.
    :param api_key: The Baseten API key.
    :param model_id: The ID of the deployed model on Baseten.
    :returns bool: True if successful, False otherwise.
    """
    url = f"{_get_management_base_url(model_id)}/autoscaling_settings"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                url=url,
                headers={"Authorization": f"Api-Key {api_key}"},
                json={"scale_down_delay": scale_down_delay},
            ) as response:
                if response.status != 200:
                    logger.warning(
                        "Unable to update Baseten scale down delay attribute."
                        f"Request failed with status code {response.status}"
                    )
                    return False

                resp = await response.json()
                if "status" in resp:
                    return True if resp["status"] in ["ACCEPTED", "UNCHANGED", "QUEUED"] else False
    except aiohttp.ClientError as e:
        logger.warning("Network error with Baseten scale_down_delay" f" {e}")
        return False
    except Exception as e:
        logger.warning("Unexpected error occurred with Baseten scale down delay request" f" {e}")
        return False


async def wake_deployment(model_id: str, api_key: str) -> bool:
    """Activates the Baseten model.

    :param model_id: The ID of the deployed model.
    :returns: True if success, False if failed.
    :rtype: bool
    """
    url = f"https://model-{model_id}.api.baseten.co/production/wake"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=url, headers={"Authorization": f"Api-Key {api_key}"}, json={}
            ) as response:
                if response.status != 202:
                    logger.warning(
                        "Unable to activate Baseten model."
                        f"Request failed with status code {response.status}"
                    )
                    return False

                return True
    except aiohttp.ClientError as e:
        logger.warning("Network error with Baseten model activation." f" {e}")
        return False
    except Exception as e:
        logger.error("Unexpected error occurred with Baseten model activation." f" {e}")
        return False


async def get_production_deployment_status(model_id: str, api_key: str) -> str | None:
    """Get model production deployment_id by it's model_id.

    :param model_id: The ID of the deployed model.
    :returns: The deployment_id of the production model.
    :rtype: str
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                _get_management_base_url(model_id),
                headers={"Authorization": f"Api-Key {api_key}"},
            ) as response:
                if response.status != 200:
                    logger.warning(
                        "Unable to get model deployment details"
                        f"Request failed with status {response.status}"
                    )
                    return None

                re = await response.json()
                return re["status"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("Unable to parse response for Model deployment info request." f" {e}")
        return None
    except aiohttp.ClientError as e:
        logger.warning("Network error with Baseten model deployment information." f" {e}")
        return None
    except Exception as e:
        logger.error(
            "Unexpected error occurred with Baseten model deployment info request." f" {e}"
        )
        return None
