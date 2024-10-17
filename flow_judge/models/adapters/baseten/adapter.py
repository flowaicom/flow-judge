import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypedDict

import aiohttp
import bleach
import structlog
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from flow_judge.eval_data_types import EvalInput, EvalOutput
from flow_judge.flow_judge import AsyncBaseAPIAdapter, BaseFlowJudge
from flow_judge.metrics import CustomMetric, Metric

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
    content: str


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


class SanitizedRequestMessage(BaseModel):
    """A sanitized and validated version of the RequestMessage.

    This class ensures that the input message conforms to expected formats and
    is free from potentially harmful content.

    Attributes:
        id (str): A unique identifier for the request (1-100 characters).
        content (str): The sanitized content of the request (1-10000 characters).

    Note:
        The content field is sanitized using the bleach library to remove potentially
        harmful HTML. Be aware that this may alter the original content.
    """

    id: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=10000)

    @field_validator("content")
    @classmethod
    def sanitize_content(cls, v):
        """Sanitize the content field to remove potentially harmful HTML."""
        return bleach.clean(v)


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
        max_retries: int = 3,
        request_timeout: float = 30.0,
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
            request_timeout (float): Timeout for individual requests in seconds. Defaults to 30.0.
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
        super().__init__(model_id, webhook_proxy_url, batch_size)
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

        super().__init__(f"https://model-{model_id}.api.baseten.co/production")
        self.model_id = model_id
        self.webhook_proxy_url = webhook_proxy_url
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
                        resp = json.loads(decoded_chunk.split("data: ")[1])
                        message = resp["data"]["choices"][0]["message"]["content"].strip()
                        # FIXME: signature is not used yet
                        signature = resp.get("signature", "")
                        logger.debug(f"signature: {signature}")
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry_error_callback=lambda retry_state: retry_state.outcome.result(),
    )
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
            It will attempt to process the request up to 3 times before giving up.
            The wait time between retries increases exponentially, starting at 1 second
            and capped at 60 seconds. These values are fixed in the decorator and do not
            use the instance variables for technical reasons. If different retry behavior
            is needed, consider implementing a custom retry decorator.
        """
        async with self.semaphore:
            try:
                response = await asyncio.wait_for(
                    self._make_request([message]), timeout=self.request_timeout
                )
                parsed_response = await self._fetch_stream(response)
                return EvalOutput.parse(parsed_response)
            except asyncio.TimeoutError:
                return FlowJudgeError(
                    error_type="TimeoutError",
                    error_message=f"Request timed out after {self.request_timeout} seconds",
                    request_id=message["id"],
                )
            except Exception as e:
                return FlowJudgeError(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    request_id=message["id"],
                )

    async def _async_fetch_batched_response(
        self, request_messages: list[RequestMessage]
    ) -> BatchResult:
        """Fetch batched responses asynchronously with retry mechanism.

        Args:
            request_messages (List[RequestMessage]): List of request messages to process.

        Returns:
            BatchResult: The result of the batch evaluation,
            including successful outputs and errors.

        Note:
            This method processes all requests concurrently. While this improves throughput,
            it may lead to increased resource usage. Monitor system resources when processing
            large batches. The method also respects rate limits, which may cause some requests
            to be delayed if the rate limit is reached.
        """
        results = BatchResult(
            successful_outputs=[],
            errors=[],
            total_requests=len(request_messages),
            success_rate=0.0,
        )

        # Process all requests with retry
        all_results = await asyncio.gather(
            *[self._process_request_with_retry(msg) for msg in request_messages]
        )

        for result in all_results:
            if isinstance(result, EvalOutput):
                results.successful_outputs.append(result)
            elif isinstance(result, FlowJudgeError):
                results.errors.append(result)

        results.success_rate = len(results.successful_outputs) / results.total_requests
        return results


class AsyncFlowJudge(BaseFlowJudge):
    """Asynchronous Flow Judge for batch evaluations.

    This class provides methods for asynchronously evaluating batches of inputs
    using a specified metric and Baseten model. It handles concurrent processing,
    error management, and result aggregation.

    Attributes:
        metric (Union[Metric, CustomMetric]): The metric used for evaluation.
        model (AsyncBasetenAPIAdapter): The async Baseten API adapter.
        output_dir (str): Directory to save output.

    Note:
        This class is designed for high-throughput, asynchronous processing. Be aware
        of potential resource constraints when evaluating large batches. Consider
        implementing additional safeguards (e.g., batch size limits) for production use.
    """

    def __init__(
        self,
        metric: Metric | CustomMetric,
        model: AsyncBasetenAPIAdapter,
        output_dir: str | None = "output/",
    ):
        """Initialize the AsyncFlowJudge.

        Args:
            metric (Union[Metric, CustomMetric]): The metric to use for evaluation.
            model (AsyncBasetenAPIAdapter): The async Baseten API adapter.
            output_dir (Optional[str]): Directory to save output. Defaults to "output/".

        Raises:
            ValueError: If the model is not an instance of AsyncBasetenAPIAdapter.

        Note:
            Ensure that the provided metric is compatible with the model's output format.
            Incompatible metrics may lead to parsing errors or incorrect evaluations.
        """
        if not isinstance(model, AsyncBasetenAPIAdapter):
            raise ValueError("model must be an instance of AsyncBasetenAPIAdapter")
        super().__init__(metric, model, output_dir)
        self.model: AsyncBasetenAPIAdapter = model

    async def async_batch_evaluate(
        self,
        eval_inputs: list[EvalInput],
        use_tqdm: bool = True,
        save_results: bool = True,
        fail_on_parse_error: bool = False,
        batch_timeout: float = 300.0,
    ) -> BatchResult:
        """Asynchronously evaluate a batch of inputs with a timeout.

        Args:
            eval_inputs (List[EvalInput]): List of evaluation inputs.
            use_tqdm (bool): Whether to use tqdm for progress. Defaults to True.
            save_results (bool): Whether to save results to disk. Defaults to True.
            fail_on_parse_error (bool): Whether to raise an exception on parse errors.
                Defaults to False.
            batch_timeout (float): Timeout for the entire batch in seconds. Defaults to 300.0.

        Returns:
            BatchResult: The result of the batch evaluation.

        Raises:
            ValueError: If evaluation fails and fail_on_parse_error is True.
            asyncio.TimeoutError: If the batch evaluation exceeds the specified timeout.

        Note:
            This method processes the entire batch asynchronously. While this improves
            throughput, it may lead to high memory usage for large batches. Monitor
            system resources and consider processing in smaller sub-batches if necessary.
            The batch_timeout applies to the entire batch, not individual requests.
        """
        self._validate_inputs(eval_inputs)
        prompts = [self._format_prompt(eval_input) for eval_input in eval_inputs]
        request_messages = [
            RequestMessage(id=str(i), content=prompt) for i, prompt in enumerate(prompts)
        ]

        try:
            batch_result = await asyncio.wait_for(
                self.model._async_fetch_batched_response(request_messages), timeout=batch_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Batch evaluation timed out after {batch_timeout} seconds")
            return BatchResult(
                successful_outputs=[],
                errors=[
                    FlowJudgeError(
                        error_type="BatchTimeoutError",
                        error_message=f"Batch evaluation timed out after {batch_timeout} seconds",
                        request_id="batch",
                    )
                ],
                total_requests=len(eval_inputs),
                success_rate=0.0,
            )

        if save_results:
            await asyncio.to_thread(
                self._save_results, eval_inputs, batch_result.successful_outputs
            )

        if batch_result.errors:
            logger.warning(
                f"Number of errors: {len(batch_result.errors)} \
                    out of {batch_result.total_requests}"
            )
            for error in batch_result.errors:
                logger.error(
                    f"Error in request {error.request_id}: \
                        {error.error_type} - {error.error_message}"
                )

        if fail_on_parse_error and batch_result.errors:
            raise ValueError(f"Evaluation failed for {len(batch_result.errors)} requests")

        return batch_result
