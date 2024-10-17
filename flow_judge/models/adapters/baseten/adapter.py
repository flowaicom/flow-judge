import os
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypedDict

import aiohttp
# import bleach
import structlog
from pydantic import BaseModel, Field, field_validator
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from flow_judge.eval_data_types import EvalInput, EvalOutput
from flow_judge.models.adapters.base import AsyncBaseAPIAdapter
# from flow_judge.metrics import CustomMetric, Metric

logger = structlog.get_logger(__name__)


class RequestMessage(TypedDict):
    """Represents a single request message for the Baseten API.

    Note:
        This class uses TypedDict for strict type checking. Ensure all fields
        are provided when instantiating. The 'id' field is crucial for tracking
        and error reporting throughout the evaluation process.

    Warning:
        Do not include sensitive information in the 'content' field, as it may
        be logged or stored for debugging purposes.
    """

    id: str
    index: int
    content: str

class ResponseMessage(TypedDict):
    """Represents a single response message from the Webhook proxy.

    Note:
        This class uses TypedDict for strict type checking. Ensure all fields
        are provided when instantiating. The 'id' field is crucial for tracking
        and error reporting throughout the evaluation process.

    Warning:
        Do not include sensitive information in the 'content' field, as it may
        be logged or stored for debugging purposes.
    """

    index: int
    request_id: str
    response: str


class FlowJudgeError(BaseModel):
    """Represents an error encountered during the Flow Judge evaluation process.

    This class encapsulates detailed error information, including the type of error,
    the specific message, the request ID that caused the error, and other metadata.

    Attributes:
        error_type (str): The type of error encountered (e.g., "TimeoutError").
        error_message (str): A detailed description of the error.
        request_id (str): The ID of the request that caused the error.
        timestamp (datetime): The time when the error occurred.
        retry_count (int): The number of retry attempts made before the error was raised.
        raw_response (Optional[str]): The raw response from Baseten or proxy, if available.

    Note:
        This class is used for both logging and error handling. Ensure that sensitive
        information is not included in the error_message or raw_response fields.
    """

    error_type: str = Field(..., description="Type of the error encountered")
    error_message: str = Field(..., description="Detailed error message")
    request_id: str = Field(..., description="ID of the request that caused the error")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Time when the error occurred"
    )
    retry_count: int = Field(default=0, description="Number of retry attempts made")
    raw_response: str | None = Field(
        None, description="Raw response from Baseten or proxy, if available"
    )

    @field_validator("error_type", "error_message", "request_id")
    @classmethod
    def check_non_empty_string(cls, v):
        """Placeholder."""
        if not v.strip():
            raise ValueError("Field must not be empty or just whitespace")
        return v


class BatchResult(BaseModel):
    """Represents the result of a batch evaluation process.

    This class contains both successful outputs and errors encountered during the
    evaluation process, as well as metadata about the batch operation.

    Attributes:
        successful_outputs (List[EvalOutput]): List of successful evaluation outputs.
        errors (List[FlowJudgeError]): List of errors encountered during evaluation.
        total_requests (int): Total number of requests processed in the batch.
        success_rate (float): Rate of successful evaluations (0.0 to 1.0).

    Note:
        The success_rate is calculated as (len(successful_outputs) / total_requests).
        Be cautious when interpreting results with a low success rate, as it may
        indicate systemic issues with the evaluation process or input data.
    """

    successful_outputs: list[EvalOutput] = Field(
        default_factory=list, description="List of successful evaluation outputs"
    )
    errors: list[FlowJudgeError] = Field(
        default_factory=list, description="List of errors encountered during evaluation"
    )
    total_requests: int = Field(..., description="Total number of requests processed")
    success_rate: float = Field(..., description="Rate of successful evaluations")

    @field_validator("total_requests")
    @classmethod
    def check_positive_total_requests(cls, v):
        """Placeholder."""
        if v <= 0:
            raise ValueError("total_requests must be positive")
        return v

    @field_validator("success_rate")
    @classmethod
    def check_success_rate_range(cls, v):
        """Placeholder."""
        if not 0 <= v <= 1:
            raise ValueError("success_rate must be between 0 and 1")
        return v


class BasetenAPIError(Exception):
    """Base exception for Baseten API errors."""

    pass


class BasetenRequestError(BasetenAPIError):
    """Exception for request-related errors."""

    pass


class BasetenResponseError(BasetenAPIError):
    """Exception for response-related errors."""

    pass


class BasetenRateLimitError(BasetenAPIError):
    """Exception for rate limit errors."""

    pass


@dataclass
class TokenBucket:
    """Implements a token bucket algorithm for rate limiting.

    This class manages a token bucket with a specified capacity and fill rate,
    allowing for controlled consumption of tokens over time.

    Attributes:
        tokens (float): Current number of tokens in the bucket.
        fill_rate (float): Rate at which tokens are added to the bucket (tokens per second).
        capacity (float): Maximum number of tokens the bucket can hold.
        last_update (float): Timestamp of the last token update.

    Note:
        This implementation is not thread-safe. If used in a multi-threaded environment,
        external synchronization mechanisms should be applied.
    """

    tokens: float
    fill_rate: float
    capacity: float
    last_update: float = field(default_factory=time.time)

    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from the bucket.

        Args:
            tokens (int): Number of tokens to consume. Defaults to 1.

        Returns:
            bool: True if tokens were successfully consumed, False otherwise.

        Note:
            This method updates the token count based on the time elapsed since
            the last update, then attempts to consume the requested number of tokens.
        """
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + self.fill_rate * (now - self.last_update))
        self.last_update = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


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
        batch_size: int,
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
                        # FIXME: signature is not used yet
                        signature = split_chunks[2].replace("\n\n","")
                        # signature = resp.get("signature", "")
                        logger.debug(f"signature: {signature} for request {request_id}")

                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.warning(f"Failed to parse chunk: {e}")
                        continue
                    if "data: eot" in decoded_chunk:
                        break

                if not message:
                    raise ValueError("Empty response from stream")

                # Validate the signature here if needed
                # if not validate_baseten_signature(message, signature):
                #     raise ValueError("Invalid Baseten signature")

                return message

    async def _process_request_with_retry(
        self, message: RequestMessage
    ) -> EvalOutput | FlowJudgeError:
        """Process a single request message with retry and exponential backoff.

        Args:
            message (RequestMessage): The request message to process.

        Returns:
            Union[EvalOutput, FlowJudgeError]: The processed output or an error object.

        Note:
            This method implements a retry mechanism with exponential backoff.
            It will attempt to process the request up to self.max_retries times
            before giving up and returning a FlowJudgeError.
        """
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(min=self.retry_min_wait, max=self.retry_max_wait),
            reraise=True,
        )
        async def _attempt_request():
            async with self.semaphore:
                message_body = {"role": "user", "content": message["content"]}
                request_id = await self._make_request([message_body])

                logger.debug(f"Requested baseten, request_id: {request_id}")

                message["id"] = request_id
                logger.debug(f"Request message updated with request_id: {message}")

                return await self._fetch_stream(request_id)

        try:
            response = await _attempt_request()
            return ResponseMessage(
                request_id=message["id"],
                response=response,
                index=message["index"]
                )
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
        
    # TODO: Implement - runtime raises error for abstract method
    async def _async_fetch_response(self, message: str) -> str:
        pass

    async def _async_fetch_batched_response(self, prompts: list[str]) -> BatchResult:
        """Process a batch of evaluation inputs asynchronously.

        Args:
            batch (List[EvalInput]): A list of evaluation inputs to process.

        Returns:
            BatchResult: An object containing successful outputs and errors.

        Note:
            This method processes each input in the batch concurrently, up to
            the rate limit. It aggregates results and errors into a BatchResult.
        """
        indexed_prompts = [RequestMessage(index=i+1, content=prompt, id=None) for i, prompt in enumerate(prompts)]
        all_results = []
        for i in range(0, len(indexed_prompts), self.batch_size):
            batch = indexed_prompts[i : i + self.batch_size]
            logger.debug(f"Batch {i}: {batch}")
            tasks = [
                self._process_request_with_retry(request_message)
                for request_message in batch
            ]
            results = await asyncio.gather(*tasks)
            all_results.append(results)

        return all_results

        successful_outputs = []
        errors = []
        for result in results:
            if isinstance(result, EvalOutput):
                successful_outputs.append(result)
            elif isinstance(result, FlowJudgeError):
                errors.append(result)

        total_requests = len(batch)
        success_rate = len(successful_outputs) / total_requests if total_requests > 0 else 0

        return BatchResult(
            successful_outputs=successful_outputs,
            errors=errors,
            total_requests=total_requests,
            success_rate=success_rate,
        )


# class BasetenFlowJudge(BaseFlowJudge):
#     """Flow Judge implementation using the Baseten API.

#     This class extends BaseFlowJudge to provide evaluation functionality
#     using the Baseten API for model inference.

#     Attributes:
#         adapter (AsyncBasetenAPIAdapter): The adapter for making API calls to Baseten.

#     Note:
#         This implementation assumes that the Baseten model is configured to
#         handle the specific evaluation task. Ensure that the model input and
#         output formats match the expectations of this class.
#     """

#     def __init__(
#         self,
#         model_id: str,
#         webhook_proxy_url: str,
#         batch_size: int,
#         max_retries: int = 3,
#         request_timeout: float = 30.0,
#         rate_limit: int = 20,
#         retry_min_wait: float = 1.0,
#         retry_max_wait: float = 60.0,
#     ):
#         """Initialize the BasetenFlowJudge.

#         Args:
#             model_id (str): The ID of the Baseten model to use for evaluation.
#             webhook_proxy_url (str): The URL of the webhook proxy for async requests.
#             batch_size (int): The number of inputs to process in each batch.
#             max_retries (int): Maximum number of retry attempts for failed requests.
#             request_timeout (float): Timeout for individual requests in seconds.
#             rate_limit (int): Maximum number of requests per minute.
#             retry_min_wait (float): Minimum wait time between retries in seconds.
#             retry_max_wait (float): Maximum wait time between retries in seconds.

#         Raises:
#             ValueError: If any of the input parameters are invalid.

#         Note:
#             The retry and rate limiting parameters should be tuned based on
#             the specific characteristics of your Baseten model and use case.
#         """
#         super().__init__()
#         self.adapter = AsyncBasetenAPIAdapter(
#             model_id,
#             webhook_proxy_url,
#             batch_size,
#             max_retries,
#             request_timeout,
#             rate_limit,
#             retry_min_wait,
#             retry_max_wait,
#         )

#     async def evaluate(self, eval_inputs: list[EvalInput]) -> list[EvalOutput]:
#         """Evaluate a list of inputs using the Baseten model.

#         Args:
#             eval_inputs (List[EvalInput]): A list of evaluation inputs to process.

#         Returns:
#             List[EvalOutput]: A list of evaluation outputs.

#         Raises:
#             RuntimeError: If there are any errors during the evaluation process.

#         Note:
#             This method processes inputs in batches and aggregates the results.
#             Any errors encountered during processing will be logged and raised
#             as a RuntimeError at the end of the evaluation.
#         """
#         all_outputs = []
#         all_errors = []

#         for i in range(0, len(eval_inputs), self.adapter.batch_size):
#             batch = eval_inputs[i : i + self.adapter.batch_size]
#             batch_result = await self.adapter.process_batch(batch)
#             all_outputs.extend(batch_result.successful_outputs)
#             all_errors.extend(batch_result.errors)

#         if all_errors:
#             error_messages = [
#                 f"Error for input {error.request_id}: {error.error_type} - {error.error_message}"
#                 for error in all_errors
#             ]
#             raise RuntimeError(
#                 f"Encountered {len(all_errors)} errors during evaluation:\n"
#                 + "\n".join(error_messages)
#             )

#         return all_outputs
