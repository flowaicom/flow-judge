import asyncio
import json
import logging
import os
import time
from typing import Any

import aiohttp
import structlog
from openai import OpenAI, OpenAIError
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from flow_judge.models.adapters.base import AsyncBaseAPIAdapter, BaseAPIAdapter
from flow_judge.models.adapters.baseten.data_io import BatchResult, Message
from flow_judge.models.adapters.baseten.errors import (
    BasetenAPIError,
    BasetenRateLimitError,
    BasetenRequestError,
    BasetenResponseError,
    FlowJudgeError,
)
from flow_judge.models.adapters.baseten.management import (
    get_production_deployment_status,
    set_scale_down_delay,
    wake_deployment,
)
from flow_judge.models.adapters.baseten.token_bucket import TokenBucket
from flow_judge.models.adapters.baseten.validation import validate_baseten_signature

logger = structlog.get_logger(__name__)


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
    """Asynchronous API adapter for Baseten model interactions.

    This adapter provides methods for making asynchronous requests to Baseten models,
    handling retries with exponential backoff, and managing rate limits.

    Attributes:
        model_id (str): The ID of the Baseten model.
        webhook_proxy_url (str): URL of the webhook proxy.
        batch_size (int): Size of batches for processing.
        max_retries (int): Maximum number of retry attempts.
        request_timeout (float): Timeout for individual requests in seconds.
        rate_limit (int): Maximum number of requests per second.
        retry_min_wait (float): Minimum wait time between retries in seconds.
        retry_max_wait (float): Maximum wait time between retries in seconds.

    Note:
        This adapter implements rate limiting and retry mechanisms. However, be aware
        that excessive retries or high concurrency may still lead to rate limit errors
        or increased costs. Monitor usage closely and adjust parameters as needed.
    """

    def __init__(
        self,
        model_id: str,
        webhook_proxy_url: str,
        batch_size: int = 128,
        max_retries: int = 1,
        request_timeout: float = 120.0,
        rate_limit: int = 20,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 60.0,
    ):
        """Initialize the AsyncBasetenAPIAdapter.

        Args:
            model_id (str): The ID of the Baseten model.
            webhook_proxy_url (str): URL of the webhook proxy.
            batch_size (int): Size of batches for processing.
            max_retries (int): Maximum number of retry attempts. Defaults to 3.
            request_timeout (float): Timeout for individual requests in seconds. Defaults to 120.0.
            rate_limit (int): Maximum number of requests per minute. Defaults to 20.
            retry_min_wait (float): Minimum wait time between retries in seconds. Defaults to 1.0.
            retry_max_wait (float): Maximum wait time between retries in seconds. Defaults to 60.0.

        Raises:
            ValueError: If any of the input parameters are invalid.

        Note:
            The retry_min_wait and retry_max_wait parameters control the exponential backoff
            strategy for retries. Adjust these values based on your specific use case and
            the characteristics of the Baseten API.
        """
        super().__init__(f"https://model-{model_id}.api.baseten.co/production")
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")
        if not webhook_proxy_url or not isinstance(webhook_proxy_url, str):
            raise ValueError("webhook_proxy_url must be a non-empty string")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be a positive number")
        if rate_limit <= 0:
            raise ValueError("rate_limit must be positive")
        if retry_min_wait < 0 or retry_max_wait < 0:
            raise ValueError("retry wait times must be non-negative")
        if retry_min_wait > retry_max_wait:
            raise ValueError("retry_min_wait must not exceed retry_max_wait")

        self.model_id = model_id
        self.webhook_proxy_url = webhook_proxy_url.rstrip()
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.rate_limit = rate_limit
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.rate_limiter = TokenBucket(
            tokens=rate_limit, fill_rate=rate_limit / 60, capacity=rate_limit
        )
        self.semaphore = asyncio.Semaphore(self.rate_limit)

        try:
            self.baseten_api_key = os.environ["BASETEN_API_KEY"]
        except KeyError as e:
            raise ValueError("BASETEN_API_KEY is not provided in the environment.") from e

    async def _check_webhook_health(self) -> bool:
        """Make an async health request to the webhook before executing generation tasks.

        Returns:
                bool: Health status, True if healthy, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.webhook_proxy_url}/health") as response:
                    if response.status != 200:
                        raise BasetenRequestError(
                            "Proxy seems in a unhealthy state." " Aborting Baseten requests."
                        )
                    return True

        except aiohttp.ClientError as e:
            raise BasetenRequestError(f"Network error while fetching token: {str(e)}") from e
        except ConnectionError as e:
            raise BasetenRequestError(
                "Unable to connect to the webhook proxy." " Make sure the correct URL is given."
            ) from e

    async def _make_request(self, request_messages: list[dict[str, Any]]) -> str:
        """Make an asynchronous request to the Baseten model.

        Args:
            request_messages (List[Dict[str, Any]]): List of request messages to process.

        Returns:
            str: The request ID for the Baseten model execution.

        Raises:
            BasetenRateLimitError: If the rate limit is exceeded.
            BasetenRequestError: If the request fails due to a network or server error.
            BasetenResponseError: If the response from Baseten is invalid.

        Note:
            This method implements rate limiting. If the rate limit is exceeded,
            it will raise a BasetenRateLimitError instead of waiting.
        """
        if not self.rate_limiter.consume():
            raise BasetenRateLimitError("Rate limit exceeded")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/async_predict",
                    headers={"Authorization": f"Api-Key {self.baseten_api_key}"},
                    json={
                        "webhook_endpoint": f"{self.webhook_proxy_url}/webhook",
                        "model_input": {"messages": request_messages},
                    },
                ) as response:
                    if response.status == 429:
                        raise BasetenRateLimitError("Rate limit exceeded")
                    elif response.status >= 400:
                        raise BasetenRequestError(f"Request failed with status {response.status}")
                    resp_json = await response.json()

                    if "request_id" not in resp_json:
                        raise BasetenResponseError("Invalid response from Baseten")

                    return resp_json["request_id"]

        except aiohttp.ClientError as e:
            raise BasetenRequestError(f"Network error: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise BasetenResponseError(f"Invalid JSON response: {str(e)}") from e

    async def _get_stream_token(self, request_id: str) -> str | None:
        """Retrieve the stream token for a given request ID.

        Args:
            request_id (str): The ID of the request to fetch the token for.

        Returns:
            Optional[str]: The stream token if successful, None otherwise.

        Raises:
            BasetenRequestError: If there's an error fetching the token.
            BasetenResponseError: If the response doesn't contain a valid token.

        Note:
            This method is used internally to authenticate stream requests.
            It should not be called directly by users of the adapter.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.webhook_proxy_url}/token", json={"request_id": request_id}
                ) as response:
                    if response.status != 200:
                        raise BasetenRequestError(f"Failed to fetch token: HTTP {response.status}")
                    data = await response.json()
                    if "token" not in data:
                        raise BasetenResponseError("Response does not contain a token")
                    return data["token"]
        except aiohttp.ClientError as e:
            raise BasetenRequestError(f"Network error while fetching token: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise BasetenResponseError(f"Invalid JSON response for token: {str(e)}") from e

    async def _fetch_stream(self, request_id: str) -> str:
        """Fetch and process the stream from the webhook proxy.

        Args:
            request_id (str): The ID of the request to fetch.

        Returns:
            str: The processed message from the stream.

        Raises:
            ValueError: If the request_id is None or if an empty response is received.
            BasetenResponseError: If there's an error parsing the stream data.

        Note:
            This method does not implement retry logic. Retries should be handled
            by the calling method if necessary.
        """
        if request_id is None:
            raise ValueError("request_id cannot be None")

        async with aiohttp.ClientSession() as session:
            token = await self._get_stream_token(request_id)
            if token is None:
                raise ValueError(f"Unable to retrieve stream token for request_id={request_id}")

            async with session.get(
                f"{self.webhook_proxy_url}/listen/{request_id}",
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                message = ""
                signature = ""
                async for chunk in response.content.iter_any():
                    decoded_chunk = chunk.decode()
                    if decoded_chunk == "data: keep-alive\n\n":
                        continue
                    if decoded_chunk == "data: server-gone\n\n":
                        break
                    try:
                        split_chunks = decoded_chunk.split("data: ")

                        resp = json.loads(split_chunks[1])

                        message = resp["data"]["choices"][0]["message"]["content"].strip()
                        signature = split_chunks[2].replace("\n\n", "").split("signature=")[1]

                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.warning(f"Failed to parse chunk: {e}")
                        raise BasetenResponseError(f"Invalid JSON response: {str(e)}") from e

                    if "data: eot" in decoded_chunk:
                        break

                if not message:
                    raise ValueError("Empty response from stream")

                # Validate the webhook signature
                if not validate_baseten_signature(resp, signature):
                    raise ValueError("Invalid Baseten signature")

                return message

    async def _process_request_with_retry(self, message: Message) -> Message | FlowJudgeError:
        """Process a single request message with retry and exponential backoff.

        Args:
            message (Message): The request message to process.

        Returns:
            Union[Message, FlowJudgeError]: The processed output or an error object.

        Note:
            This method implements a retry mechanism with exponential backoff.
            It will attempt to process the request up to self.max_retries times
            before giving up and returning a FlowJudgeError.
        """

        @retry(
            retry=retry_if_exception_type(BasetenAPIError),
            stop=stop_after_attempt(self.max_retries),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            wait=wait_exponential(min=self.retry_min_wait, max=self.retry_max_wait),
            reraise=True,
        )
        async def _attempt_request():
            async with self.semaphore:
                message_body = {"role": "user", "content": message["prompt"]}

                request_id = await self._make_request([message_body])

                logger.debug(f"Requested baseten, request_id: {request_id}")

                message["id"] = request_id
                logger.debug(f"Request message updated with request_id: {message}")

                return await self._fetch_stream(request_id)

        try:
            response = await _attempt_request()
            message["response"] = response
            return message
        except RetryError as e:
            return FlowJudgeError(
                error_type="RetryError",
                error_message=str(e),
                request_id=message["id"],
                retry_count=self.max_retries,
            )
        except (BasetenAPIError, ValueError) as e:
            return FlowJudgeError(
                error_type=type(e).__name__,
                error_message=str(e),
                request_id=message["id"],
            )

    async def _is_model_awake(self) -> bool:
        """Wake the deployed model.

        :returns: True if successfully awake or pinging has a soft failure.
        False if the model status is indicating 'failure'.

        Note: Baseten does not wait for the model to wake up.
        Hence the additional pinging for the model status.
        We use the model information API from the Management endpoints
        to get information of the model status and keep pinging every 10secs
        when status is in a 'transitionary' phase.
        """
        # Initial check to see if the model is already active.
        status = await get_production_deployment_status(self.model_id, self.baseten_api_key)
        if status == "ACTIVE":
            return

        has_triggered_correctly = await wake_deployment(self.model_id, self.baseten_api_key)
        if not has_triggered_correctly:
            raise BasetenAPIError("Trigger to wake the deployed model failed.")

        timeout_seconds = 300

        # Wait for the initial trigger to switch deployment status
        await asyncio.sleep(3)

        async def has_model_activated(start_time: float | int):
            status = await get_production_deployment_status(self.model_id, self.baseten_api_key)
            if status is None:
                logger.warning("Unable to detect if model is awake. " "Continuing anyway.")
                return True
            if status in ["BUILDING", "DEPLOYING", "LOADING_MODEL", "WAKING_UP", "UPDATING"]:
                logger.info("The deployed model is waking up.")

                if time.time() - start_time >= timeout_seconds:
                    raise BasetenAPIError("Model took too long to wake up. Stopping execution.")

                await asyncio.sleep(10)
                return await has_model_activated(start_time)

            if status in ["BUILD_FAILED", "BUILD_STOPPED", "FAILED", "UNHEALTHY"]:
                raise BasetenAPIError(
                    "Model seems to be in an unhealthy state. Stopping execution."
                )

            if status in ["ACTIVE"]:
                logger.info("The deployed model is active.")
                return True

        if not await has_model_activated(time.time()):
            raise BasetenAPIError("Unable to wake up the model.")

    async def _initialize_state_for_request(self, scale_down_delay: int) -> None:
        """Pre-steps for a single/batched request.

        :param scale_down_delay: The delay in seconds to scale down the model.

        Note:
            Activates the model by sendng a "wake-up" request and waits for
            the model to wake-up. Updates the scale-down delay value, defaults to
            120secs for a single request, and 30secs for batched requests.

        Raises:
            BasetenAPIError for when we are unable to activate the model.
        """
        await self._is_model_awake()

        # Update scale down delay to 30secs for batched requests.
        is_scaled_down = await set_scale_down_delay(
            scale_down_delay=30, api_key=self.baseten_api_key, model_id=self.model_id
        )
        if not is_scaled_down:
            logger.warning("Unable to reduce scale down delay. Continuing with default.")

    async def _async_fetch_response(self, prompt: str) -> Message | FlowJudgeError:
        """Single async request to Baseten.

        Args:
            prompt: Prompt string for the request.

        Returns:
            A message dictionary or an error.
            (Message | FlowJudgeError)
        """
        # Attempt to initialize the model state.
        try:
            await self._check_webhook_health()
            await self._initialize_state_for_request(scale_down_delay=120)
        except BasetenAPIError as e:
            return FlowJudgeError(
                error_type=type(e).__name__, error_message=str(e), request_id=None
            )

        result = await self._process_request_with_retry(
            Message(prompt=prompt, index=1, id=None, response=None)
        )

        if isinstance(result, FlowJudgeError):
            return result

        return result["response"]

    async def _async_fetch_batched_response(self, prompts: list[str]) -> BatchResult:
        """Process a batch of evaluation inputs asynchronously.

        Args:
            prompts (List[str]): A list of prompts to process.

        Returns:
            BatchResult: An object containing successful outputs and errors.

        Note:
            This method processes each input in the batch concurrently, up to
            the rate limit. It aggregates results and errors into a BatchResult.
        """
        indexed_prompts = [
            Message(index=i + 1, prompt=prompt, id=None, response="")
            for i, prompt in enumerate(prompts)
        ]

        all_results = []

        # Attempt to initialize the model state.
        try:
            await self._initialize_state_for_request(scale_down_delay=30)
        except BasetenAPIError as e:
            return BatchResult(
                successful_outputs=[],
                errors=[
                    FlowJudgeError(
                        error_type=type(e).__name__, error_message=str(e), request_id=None
                    )
                ],
                success_rate=0,
                total_requests=0,
            )

        for i in range(0, len(indexed_prompts), self.batch_size):
            try:
                await self._check_webhook_health()
            except BasetenAPIError as e:
                all_results.append(
                    FlowJudgeError(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        request_id=None,
                    )
                )
                break

            batch = indexed_prompts[i : i + self.batch_size]
            logger.debug(f"Batch {i}: {batch}")
            tasks = [self._process_request_with_retry(request_message) for request_message in batch]

            results = await asyncio.gather(*tasks)
            all_results.extend(results)

        successful_outputs = []
        errors = []

        for result in all_results:
            if isinstance(result, dict) and all(
                key in list(Message.__annotations__.keys()) for key, _ in result.items()
            ):
                successful_outputs.append(result)
            elif isinstance(result, FlowJudgeError):
                errors.append(result)

        successful_outputs = sorted(successful_outputs, key=lambda o: o["index"])
        total_requests = len(successful_outputs) + len(errors)
        success_rate = len(successful_outputs) / total_requests if total_requests > 0 else 0

        return BatchResult(
            successful_outputs=successful_outputs,
            errors=errors,
            total_requests=total_requests,
            success_rate=success_rate,
        )
