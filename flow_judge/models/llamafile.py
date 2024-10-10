import asyncio
import atexit
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
import warnings
import weakref
from typing import Any

import requests
from tqdm import tqdm

from flow_judge.models.common import (
    AsyncBaseFlowJudgeModel,
    BaseFlowJudgeModel,
    GenerationParams,
    ModelConfig,
    ModelType,
)

try:
    from openai import AsyncOpenAI, OpenAI

    LLAMAFILE_AVAILABLE = True
except ImportError:
    LLAMAFILE_AVAILABLE = False

LLAMAFILE_URL = (
    "https://huggingface.co/flowaicom/Flow-Judge-v0.1-Llamafile/resolve/main/"
    "flow-judge.llamafile"
)

logger = logging.getLogger(__name__)


class LlamafileConfig(ModelConfig):
    """Configuration class for Llamafile models.

    Extends the base ModelConfig with Llamafile specific parameters.
    """

    _DEFAULT_MODEL_ID = "flowaicom/Flow-Judge-v0.1-Llamafile"

    def __init__(
        self,
        generation_params: GenerationParams,
        model_filename: str = "flow-judge.llamafile",
        cache_dir: str = os.path.expanduser("~/.cache/flow-judge"),
        port: int = 8085,
        disable_kv_offload: bool = False,
        quantized_kv: bool = True,
        flash_attn: bool = True,
        context_size: int = 8192,
        gpu_layers: int = 34,
        thread_count: int | None = None,
        batch_size: int = 32,
        max_concurrent_requests: int = 1,
        llamafile_server_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize LlamafileConfig with model details and Llamafile specific parameters."""
        model_id = kwargs.pop("_model_id", None) or self._DEFAULT_MODEL_ID
        super().__init__(
            model_id=model_id,
            model_type=ModelType.LLAMAFILE,
            generation_params=generation_params,
            **kwargs,
        )
        self.model_filename = model_filename
        self.cache_dir = cache_dir
        self.port = port
        self.disable_kv_offload = disable_kv_offload
        self.quantized_kv = quantized_kv
        self.flash_attn = flash_attn
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.thread_count = thread_count
        self.batch_size = batch_size
        self.max_concurrent_requests = max_concurrent_requests
        self.llamafile_server_kwargs = llamafile_server_kwargs or {}


def cleanup_llamafile(process_ref):
    """Clean up the Llamafile process.

    :param process_ref: Weak reference to the Llamafile process.
    """
    process = process_ref()
    if process:
        pgid = None
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except OSError:
            # Handle the case where we can't get the process group ID
            try:
                process.terminate()
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass


class Llamafile(BaseFlowJudgeModel, AsyncBaseFlowJudgeModel):
    """Combined FlowJudge model class for Llamafile supporting both sync and async operations."""

    _instances = set()
    _next_port = 8085
    _DEFAULT_MODEL_ID = "flowaicom/Flow-Judge-v0.1-Llamafile"
    _MODEL_TYPE = "llamafile"

    @classmethod
    def _get_next_port(cls):
        port = cls._next_port
        cls._next_port += 1
        return port

    def __init__(
        self,
        generation_params: dict[str, Any] | None = None,
        cache_dir: str = os.path.expanduser("~/.cache/flow-judge"),
        port: int | None = None,
        disable_kv_offload: bool = False,
        quantized_kv: bool = True,
        flash_attn: bool = True,
        context_size: int = 8192,
        gpu_layers: int = 34,
        thread_count: int | None = None,
        batch_size: int = 32,
        max_concurrent_requests: int = 1,
        llamafile_server_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize the FlowJudge Llamafile model.

        :param model_id: Identifier for the model. If None, uses the default model.
        :param generation_params: Dictionary of parameters for text generation. Can include:
            - temperature: float (default: 0.1)
            - top_p: float (default: 0.95)
            - max_new_tokens: int (default: 1000)
            - do_sample: bool (default: True)
        :param cache_dir: Directory to cache the model files.
        :param port: Port to run the Llamafile server on. If None, assigns automatically.
        :param disable_kv_offload: Whether to disable KV offloading.
        :param quantized_kv: Whether to enable Quantized KV.
        :param flash_attn: Whether to enable Flash Attention.
        :param context_size: Size of the context window.
        :param gpu_layers: Number of GPU layers to use.
        :param thread_count: Number of threads to use. If None, uses the number of CPU cores.
        :param batch_size: Batch size for processing.
        :param max_concurrent_requests: Maximum number of concurrent requests.
        :param llamafile_server_kwargs: Additional keyword arguments for the Llamafile server.
        :param kwargs: Additional keyword arguments.

        Note:
        Continuous batching is enabled by default.

        For more information about these arguments and their effects, please refer to the
        Llamafile documentation:
        https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md

        :raises LlamafileError: If required Llamafile packages are not installed or
            initialization fails.
        """
        if not LLAMAFILE_AVAILABLE:
            raise LlamafileError(
                status_code=1,
                message="The required Llamafile packages are not installed. "
                "Please install them by adding 'llamafile' to your extras:\n"
                "pip install flow-judge[llamafile]",
            )

        # Allow internal override of model_id for debugging/development
        model_id = kwargs.pop("_model_id", None) or self._DEFAULT_MODEL_ID

        if model_id != self._DEFAULT_MODEL_ID:
            warnings.warn(
                f"The model '{model_id}' is not officially supported. "
                f"This library is designed for the '{self._DEFAULT_MODEL_ID}' model. "
                "Using other models may lead to unexpected behavior, and we do not handle "
                "GitHub issues for unsupported models. Proceed with caution.",
                UserWarning,
                stacklevel=2,
            )

        generation_params = GenerationParams(**(generation_params or {}))

        if port is None:
            port = self._get_next_port()

        if thread_count is None:
            thread_count = os.cpu_count()

        config = LlamafileConfig(
            generation_params=generation_params,
            model_filename="flow-judge.llamafile",
            cache_dir=cache_dir,
            port=port,
            disable_kv_offload=disable_kv_offload,
            quantized_kv=quantized_kv,
            flash_attn=flash_attn,
            context_size=context_size,
            gpu_layers=gpu_layers,
            thread_count=thread_count,
            batch_size=batch_size,
            max_concurrent_requests=max_concurrent_requests,
            llamafile_server_kwargs=llamafile_server_kwargs,
            **kwargs,
        )
        self.config = config

        # Call both parent class initializers
        BaseFlowJudgeModel.__init__(
            self, model_id, self._MODEL_TYPE, config.generation_params, **kwargs
        )
        AsyncBaseFlowJudgeModel.__init__(
            self, model_id, self._MODEL_TYPE, config.generation_params, **kwargs
        )

        try:
            self.generation_params = config.generation_params
            self.cache_dir = config.cache_dir
            self.model_repo = config.model_id.split("/")[0]
            self.model_filename = config.model_filename
            self.port = config.port
            self.llamafile_process = None

            self.sync_client = kwargs.get("sync_client") or OpenAI(
                base_url=f"http://127.0.0.1:{self.port}/v1", api_key="not-needed"
            )
            self.async_client = kwargs.get("async_client") or AsyncOpenAI(
                base_url=f"http://127.0.0.1:{self.port}/v1", api_key="not-needed"
            )

            self.timeout = kwargs.get("timeout", 30)
            self._server_running = False
            self._context_depth = 0

            self.disable_kv_offload = config.disable_kv_offload
            self.quantized_kv = config.quantized_kv
            self.flash_attn = config.flash_attn
            self.llamafile_server_kwargs = config.llamafile_server_kwargs

            self.metadata = {
                "model_id": model_id,
                "model_type": "llamafile",
            }

        except Exception as e:
            raise LlamafileError(
                status_code=2,
                message=f"An error occurred while initializing the Llamafile model: {str(e)}\n"
                "Please make sure you have installed all required dependencies"
                " by adding 'llamafile' to your extras:\n"
                "pip install flow-judge[llamafile]",
            ) from e

        self._instances.add(weakref.ref(self, self._finalizer))
        atexit.register(self.cleanup)

    @classmethod
    def _finalizer(cls, ref):
        cls._instances.discard(ref)

    def is_server_running(self):
        """Check if the Llamafile server is running.

        :return: True if the server is running, False otherwise.
        """
        try:
            self.sync_client.models.list()
            return True
        except Exception:
            return False

    def download_llamafile(self):
        """Download the Llamafile model.

        :return: Path to the downloaded Llamafile.
        :raises DownloadError: If the download fails.
        """
        local_dir = self.cache_dir
        os.makedirs(local_dir, exist_ok=True)
        llamafile_path = os.path.abspath(os.path.join(local_dir, self.model_filename))

        if not os.path.exists(llamafile_path):
            logging.info(f"Downloading llamafile to {llamafile_path}")
            try:
                response = requests.get(LLAMAFILE_URL, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                block_size = 8192

                with (
                    open(llamafile_path, "wb") as file,
                    tqdm(
                        desc="Downloading",
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as progress_bar,
                ):
                    for data in response.iter_content(block_size):
                        size = file.write(data)
                        progress_bar.update(size)

            except requests.exceptions.RequestException as e:
                logging.error(f"Error downloading llamafile: {str(e)}")
                if os.path.exists(llamafile_path):
                    os.remove(llamafile_path)
                raise DownloadError(f"Failed to download llamafile: {str(e)}") from e

            except Exception as e:
                logging.error(f"Unexpected error during download: {str(e)}")
                if os.path.exists(llamafile_path):
                    os.remove(llamafile_path)
                raise DownloadError(f"Unexpected error during download: {str(e)}") from e

        try:
            os.chmod(llamafile_path, 0o755)
            logging.debug(f"Set executable permissions for {llamafile_path}")
        except OSError as e:
            logging.error(f"Failed to set executable permissions: {str(e)}")
            raise DownloadError(f"Failed to set executable permissions: {str(e)}") from e

        if not os.path.exists(llamafile_path):
            raise DownloadError("Llamafile not found after download attempt")

        return llamafile_path

    def start_llamafile_server(self):
        """Start the Llamafile server.

        :raises FileNotFoundError: If the Llamafile is not found.
        :raises PermissionError: If the Llamafile is not executable.
        :raises RuntimeError: If the server fails to start within the timeout period.
        """
        logging.info("Starting llamafile server...")
        llamafile_path = self.download_llamafile()
        logging.info(f"Llamafile path: {llamafile_path}")

        if not os.path.exists(llamafile_path):
            logging.error(f"Llamafile not found at {llamafile_path}")
            raise FileNotFoundError(f"Llamafile not found at {llamafile_path}")

        if not os.access(llamafile_path, os.X_OK):
            logging.error(f"Llamafile at {llamafile_path} is not executable")
            raise PermissionError(f"Llamafile at {llamafile_path} is not executable")

        command = self._build_llamafile_command(llamafile_path)
        logging.info(f"Starting llamafile server with command: {command}")

        try:
            self.llamafile_process = self._start_llamafile_process(command)
            self._wait_for_server_start()
        except Exception as e:
            logging.exception(f"Error starting llamafile server: {str(e)}")
            if self.llamafile_process:
                logging.info("Terminating llamafile process due to startup error")
                self.llamafile_process.terminate()
            raise

    def _build_llamafile_command(self, llamafile_path):
        command = (
            f"sh -c '{llamafile_path} --server --host 127.0.0.1 --port {self.port} "
            f"-c {self.config.context_size} "
            f"-ngl {self.config.gpu_layers} "
            f"--temp {self.generation_params.temperature} "
            f"-n {self.generation_params.max_new_tokens} "
            f"--threads {self.config.thread_count} "
            f"--nobrowser -b {self.config.batch_size} "
            f"--parallel {self.config.max_concurrent_requests} "
            f"--cont-batching"
        )

        if self.config.disable_kv_offload:
            command += " -nkvo"
            logging.info("KV offloading disabled")

        if self.config.quantized_kv:
            command += " -ctk q4_0 -ctv q4_0"
            logging.info("Quantized KV enabled")

        if self.config.flash_attn:
            command += " -fa"
            logging.info("Flash Attention enabled")

        if self.config.quantized_kv and not self.config.flash_attn:
            raise LlamafileError(
                "Quantized KV is enabled but Flash Attention is disabled."
                " This configuration won't function."
            )

        # Add any additional server arguments
        for key, value in self.llamafile_server_kwargs.items():
            command += f" --{key} {value}"
            logging.info(f"Additional server argument added: --{key} {value}")

        command += "'"
        return command

    def _start_llamafile_process(self, command):
        def log_output(pipe, log_func):
            for line in iter(pipe.readline, ""):
                log_func(line.strip())

        process = subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            text=True,
            preexec_fn=os.setsid,
        )
        logging.info(f"Subprocess started with PID: {process.pid}")

        # Register cleanup function for this specific process
        atexit.register(cleanup_llamafile, weakref.ref(process))

        # Start threads to log stdout and stderr in real-time
        threading.Thread(
            target=log_output, args=(process.stdout, logging.info), daemon=True
        ).start()
        threading.Thread(
            target=log_output, args=(process.stderr, logging.info), daemon=True
        ).start()

        return process

    def _wait_for_server_start(self):
        # Wait for the server to start or timeout after 60 seconds
        start_time = time.time()
        while time.time() - start_time < 60:
            if self.is_server_running():
                logging.info("Llamafile server started successfully")
                return  # Exit the method, keeping the server running
            time.sleep(1)
            logging.debug(
                f"Waiting for server to start... " f"(Elapsed: {time.time() - start_time:.2f}s)"
            )

            # Check if the process has terminated
            if self.llamafile_process.poll() is not None:
                stdout, stderr = self.llamafile_process.communicate()
                logging.error(
                    "Llamafile process terminated unexpectedly. "
                    f"Exit code: {self.llamafile_process.returncode}"
                )
                logging.error(f"Stdout: {stdout}")
                logging.error(f"Stderr: {stderr}")
                raise RuntimeError(
                    "Llamafile process terminated unexpectedly. "
                    f"Exit code: {self.llamafile_process.returncode}"
                )

        # If we've reached here, the server didn't start in time
        logging.error("Llamafile server failed to start within 60 seconds.")
        self.llamafile_process.terminate()
        raise RuntimeError("Llamafile server failed to start within 60 seconds.")

    def stop_llamafile_server(self):
        """Stop the Llamafile server if it's running."""
        if self.llamafile_process:
            cleanup_llamafile(weakref.ref(self.llamafile_process))
            self.llamafile_process = None

    def cleanup(self):
        """Clean up resources by stopping the Llamafile server."""
        self.stop_llamafile_server()

    def __del__(self):
        """Destructor to ensure cleanup when the object is garbage collected."""
        try:
            self.cleanup()
        except (ProcessLookupError, OSError) as e:
            # These exceptions might occur if the process is already terminated
            logging.warning(f"Error during cleanup in __del__: {str(e)}")
        except Exception as e:
            # Log unexpected exceptions, but don't raise them as __del__ should not raise exceptions
            logging.error(f"Unexpected error during cleanup in __del__: {str(e)}")

    def _generate(self, prompt: str) -> str:
        self._ensure_server_running()
        response = self.sync_client.chat.completions.create(
            model="flow-judge",
            messages=[{"role": "user", "content": prompt.strip()}],
            **self._get_generation_params(),
        )
        return response.choices[0].message.content.strip()

    async def _async_generate(self, prompt: str) -> str:
        self._ensure_server_running()
        response = await self.async_client.chat.completions.create(
            model="flow-judge",
            messages=[{"role": "user", "content": prompt.strip()}],
            **self._get_generation_params(),
        )
        return response.choices[0].message.content.strip()

    def _get_generation_params(self):
        return {
            "max_tokens": self.generation_params.max_new_tokens,
            "top_p": self.generation_params.top_p,
            "temperature": self.generation_params.temperature,
            "stop": getattr(self.generation_params, "stop", ["<|endoftext|>"]),
        }

    def _batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        self._ensure_server_running()
        return [self._generate(prompt) for prompt in prompts]

    async def _async_batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        self._ensure_server_running()
        max_concurrency = self.generation_params.get("max_concurrent_requests", 4)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def generate_with_semaphore(prompt):
            async with semaphore:
                return await self._async_generate(prompt)

        return await asyncio.gather(*[generate_with_semaphore(prompt) for prompt in prompts])

    def _ensure_server_running(self):
        if not self._server_running:
            self.start_llamafile_server()
            start_time = time.time()
            while not self.is_server_running():
                if time.time() - start_time > self.timeout:
                    raise TimeoutError("Server failed to start within the specified timeout.")
                time.sleep(0.1)
            self._server_running = True

    def __enter__(self):
        """Context manager entry method."""
        self._context_depth += 1
        self._ensure_server_running()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self._context_depth -= 1
        if self._context_depth == 0:
            self.stop_llamafile_server()
            self._server_running = False

    @classmethod
    def cleanup_all(cls):
        """Class method to clean up all instances of Llamafile."""
        for instance_ref in list(cls._instances):
            instance = instance_ref()
            if instance is not None:
                instance.cleanup()


# Register cleanup_all at the module level
atexit.register(Llamafile.cleanup_all)


class LlamafileError(Exception):
    """Custom exception for Llamafile-related errors."""

    def __init__(self, status_code: int, message: str):
        """Initialize a LlamafileError with a status code and message."""
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class DownloadError(Exception):
    """Exception raised when there's an error downloading the Llamafile."""

    pass
