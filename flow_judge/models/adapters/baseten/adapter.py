import os
import json
import requests
import asyncio
import aiohttp
import logging

from typing import Any, Dict, List

from ..base import BaseAPIAdapter


logger = logging.getLogger(__name__)

class BasetenAPIAdapter(BaseAPIAdapter):
    """API utility class to execute sync requests from Baseten remote model hosting."""
    def __init__(self, baseten_model_id: str):
        super().__init__(f"https://model-{baseten_model_id}.api.baseten.co/production")

        self.baseten_model_id = baseten_model_id
        try:
            self.baseten_api_key = os.environ["BASETEN_API_KEY"]
        except KeyError:
            raise ValueError("BASETEN_API_KEY is not provided in the environment.")

    def _make_request(self, request_messages: Dict[str, Any]) -> Dict:
        try:
            resp = requests.post(
                url=self.base_url + "/predict",
                headers={"Authorization": f"Api-Key {self.baseten_api_key}"},
                json=request_messages
            )

            return resp.json()
        except Exception as e:
            logger.warning(f"An unexpected error occurred: {e}")

    def _fetch_response(self, request_messages: Dict[str, Any]) -> str:
        request_body = {"messages": request_messages}
        parsed_resp = self._make_request(request_body)

        try:
            message = parsed_resp["choices"][0]["message"]["content"].strip()
            return message
        except Exception as e:
            logger.warning(f"Failed to parse model response: {e}")
            logger.warning(f"Returning default value {parsed_resp}")
            return ""
        
    def _fetch_batched_response(self, request_messages: list[Dict[str, Any]]) -> list[str]:
        outputs = []
        for message in request_messages:
            request_body = {"messages": message}
            parsed_resp = self._make_request(request_body)
            try:
                message = parsed_resp["choices"][0]["message"]["content"].strip()
                outputs.append(message)
            except Exception as e:
                logger.warning(f"Failed to parse model response: {e}")
                logger.warning("Returning default value")
                outputs.append("")
        return outputs

class AsyncBasetenAPIAdapter(BaseAPIAdapter):
    """Async webhook requests for the Baseten remote model"""
    def __init__(self, baseten_model_id: str, webhook_proxy_url: str):
        super().__init__(f"https://model-{baseten_model_id}.api.baseten.co/production")
        self.baseten_model_id = baseten_model_id
        self.webhook_proxy_url = webhook_proxy_url
        try:
            self.baseten_api_key = os.environ["BASETEN_API_KEY"]
        except KeyError:
            raise ValueError("BASETEN_API_KEY is not provided in the environment.")

    async def _make_request(self, request_messages: Dict[str, Any]) -> str:
        model_input = {"messages": request_messages}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=self.base_url + "/async_predict",
                headers={"Authorization": f"Api-Key {self.baseten_api_key}"},
                json={
                    "webhook_endpoint": self.webhook_proxy_url + "/webhook",
                    "model_input": model_input
                }
            ) as response:
                resp_json = await response.json()
                return resp_json["request_id"]

    async def _fetch_stream(self, request_id: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.webhook_proxy_url}/listen/{request_id}") as response:
                message = ""
                async for chunk in response.content.iter_any():
                    decoded_chunk = chunk.decode()
                    if decoded_chunk == "data: keep-alive\n\n":
                        continue
                    if decoded_chunk == "data: server-gone\n\n":
                        break
                    data_and_eot = decoded_chunk.split("\n\n")
                    data_chunk = data_and_eot[0]
                    resp_str = data_chunk.split("data: ")[1] if data_chunk.startswith("data: ") else data_chunk
                    try:
                        resp = json.loads(resp_str)
                        message = resp["data"]["choices"][0]["message"]["content"].strip()
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse JSON for request_id {request_id}: {e}")
                        continue
                    if len(data_and_eot) > 1 and "\n\ndata: eot" in data_and_eot[1]:
                        break
                return message

    async def _async_fetch_response(self, request_messages: Dict[str, Any]) -> str:
        request_id = await self._make_request(request_messages)
        return await asyncio.wait_for(self._fetch_stream(request_id), timeout=120)  # 2 minutes timeout

    async def _async_fetch_batched_response(self, request_messages: List[Dict[str, Any]]) -> List[str]:
        request_ids = await asyncio.gather(*[self._make_request(message) for message in request_messages])
        tasks = [self._fetch_stream(request_id) for request_id in request_ids]
        results = await asyncio.gather(*[asyncio.wait_for(task, timeout=120) for task in tasks], return_exceptions=True)
        return [result if not isinstance(result, Exception) else "" for result in results]