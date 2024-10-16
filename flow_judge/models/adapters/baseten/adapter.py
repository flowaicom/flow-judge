import asyncio
import json
import logging
import os
from typing import Any

import aiohttp
from openai import OpenAI, OpenAIError

from ..base import AsyncBaseAPIAdapter, BaseAPIAdapter
from ..baseten.webhook import ensure_baseten_webhook_secret
from .validation import validate_baseten_signature

logger = logging.getLogger(__name__)


class BasetenAPIAdapter(BaseAPIAdapter):
    """API utility class to execute sync requests from Baseten remote model hosting."""

    def __init__(self, baseten_model_id: str):
        """Initialize the BasetenAPIAdapter.

        :param baseten_model_id: The model_id designated for your deployment
            - Can be found on your Baseten dashboard: https://app.baseten.co/models
        :raises ValueError: if BASETEN_API_KEY environment variable is missing.
        """
        self.baseten_model_id = baseten_model_id

        try:
            self.baseten_api_key = os.environ["BASETEN_API_KEY"]
        except KeyError as e:
            raise ValueError("BASETEN_API_KEY is not provided in the environment.") from e

        base_url = "https://bridge.baseten.co/v1/direct"
        self.client = OpenAI(api_key=self.baseten_api_key, base_url=base_url)

        super().__init__(base_url)

    def _make_request(self, request_messages: dict[str, Any]) -> dict:
        try:
            completion = self.client.chat.completions.create(
                messages=request_messages,
                model="flowaicom/Flow-Judge-v0.1-AWQ",
                extra_body={"baseten": {"model_id": self.baseten_model_id}},
            )
            return completion

        except OpenAIError as e:
            logger.warning(f"Model request failed: {e}")
        except Exception as e:
            logger.warning(f"An unexpected error occurred: {e}")

    def _fetch_response(self, request_messages: dict[str, Any]) -> str:
        completion = self._make_request(request_messages)

        try:
            message = completion.choices[0].message.content.strip()
            return message
        except Exception as e:
            logger.warning(f"Failed to parse model response: {e}")
            logger.warning("Returning default value")
            return ""

    def _fetch_batched_response(self, request_messages: list[dict[str, Any]]) -> list[str]:
        outputs = []
        for message in request_messages:
            completion = self._make_request(message)
        try:
            outputs.append(completion.choices[0].message.content.strip())
        except Exception as e:
            logger.warning(f"Failed to parse model response: {e}")
            logger.warning("Returning default value")
            outputs.append("")
        return outputs


class AsyncBasetenAPIAdapter(AsyncBaseAPIAdapter):
    """Async webhook requests for the Baseten remote model."""

    def __init__(self, baseten_model_id: str, webhook_proxy_url: str, batch_size: int):
        """Initialize the async Baseten adapter.

        :param baseten_model_id: The model_id designated for your deployment.
            - Can be found on your Baseten dashboard: https://app.baseten.co/models
        :param webhook_proxy_url: The webhook url for Baseten's response.
        :param batch_size: The batch size of concurrent requests. (default 128)
            - For optimization set size to concurrency_target * number_of_replicas
            - See: https://docs.baseten.co/performance/concurrency
        :raises ValueError: if environment variables are missing.
        """
        super().__init__(f"https://model-{baseten_model_id}.api.baseten.co/production")
        self.baseten_model_id = baseten_model_id
        self.webhook_proxy_url = webhook_proxy_url
        self.batch_size = batch_size

        if not ensure_baseten_webhook_secret():
            raise ValueError("BASETEN_WEBHOOK_SECRET is not provided in the environment.")

        try:
            self.baseten_api_key = os.environ["BASETEN_API_KEY"]
        except KeyError as e:
            raise ValueError("BASETEN_API_KEY is not provided in the environment.") from e

    async def _make_request(self, request_messages: list[dict[str, Any]]) -> str:
        model_input = {"messages": request_messages}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=self.base_url.rstrip("/") + "/async_predict",
                headers={"Authorization": f"Api-Key {self.baseten_api_key}"},
                json={
                    "webhook_endpoint": self.webhook_proxy_url.rstrip("/") + "/webhook",
                    "model_input": model_input,
                },
            ) as response:
                resp_json = await response.json()
                try:
                    return resp_json["request_id"]
                except KeyError as e:
                    if "error" in resp_json:
                        error_msg = resp_json["error"]
                        logger.error(f"Baseten request failed. {error_msg}")

                    logger.error(f"Unable to parse Baseten response: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Unknown error occured with Beseten request." f"{e}")
                    return None

    async def _get_stream_token(self, request_id: str) -> str | None:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.webhook_proxy_url}/token", json={"request_id": request_id}
            ) as response:
                try:
                    resp = await response.json()
                    return resp["token"]
                except KeyError as e:
                    logger.error(f"Unable to parse stream token response: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Cannot retrieve stream token." f"{e}")
                    return None

    async def _fetch_stream(self, request_id: str) -> str:
        if request_id is None:
            return ""
        async with aiohttp.ClientSession() as session:
            token = await self._get_stream_token(request_id)
            if token is None:
                logger.error(
                    "Unable to retrieve the stream token. "
                    f"Cannot connect to the proxy for request_id={request_id}."
                )
                return ""

            async with session.get(
                f"{self.webhook_proxy_url}/listen/{request_id}",
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                message = ""
                signature = ""

                try:
                    async for chunk in response.content.iter_any():
                        decoded_chunk = chunk.decode()

                        if decoded_chunk == "data: keep-alive\n\n":
                            continue
                        if decoded_chunk == "data: server-gone\n\n":
                            break

                        data_and_eot = decoded_chunk.split("\n\n")
                        data_chunk = data_and_eot[0]
                        signature = data_and_eot[1].split("data: signature=")[1]

                        resp_str = (
                            data_chunk.split("data: ")[1]
                            if data_chunk.startswith("data: ")
                            else data_chunk
                        )

                        try:
                            resp = json.loads(resp_str)
                            message = resp["data"]["choices"][0]["message"]["content"].strip()
                        except json.JSONDecodeError as e:
                            logging.warning(
                                f"Failed to parse JSON for request_id {request_id}: {e}"
                            )
                            continue

                        message = (
                            ""
                            if message != "" and not validate_baseten_signature(resp, signature)
                            else message
                        )

                        if len(data_and_eot) > 2 and "data: eot" in data_and_eot[2]:
                            break

                    return message

                except (TimeoutError, asyncio.CancelledError):
                    logger.error(f"Request timed out for request_id: {request_id}")
                    return message
                except Exception as e:
                    logger.error(f"Unknown exception occurred for request_id: {request_id}" f"{e}")
                    return message

    async def _async_fetch_response(self, request_messages: dict[str, Any]) -> str:
        request_id = await self._make_request(request_messages)
        return await asyncio.wait_for(
            self._fetch_stream(request_id), timeout=120
        )  # 2 minutes timeout

    async def _async_fetch_batched_response(
        self, request_messages: list[dict[str, Any]]
    ) -> list[str]:
        batches = [
            request_messages[i : i + self.batch_size]
            for i in range(0, len(request_messages), self.batch_size)
        ]
        results = []
        for batch in batches:
            request_ids = await asyncio.gather(*[self._make_request(message) for message in batch])

            tasks = [self._fetch_stream(request_id) for request_id in request_ids]

            batch_results = await asyncio.gather(
                *[asyncio.wait_for(task, timeout=120) for task in tasks], return_exceptions=True
            )

            results.extend(
                [result if not isinstance(result, Exception) else "" for result in batch_results]
            )
        return results
