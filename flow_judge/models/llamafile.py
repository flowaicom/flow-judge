import asyncio
import logging
import os
import shlex
import subprocess
import threading
import time
from typing import Any, Dict, List

import requests
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

from flow_judge.models.common import AsyncBaseFlowJudgeModel, BaseFlowJudgeModel, ModelConfig, ModelType


LLAMAFILE_URL = (
    "https://huggingface.co/sariola/flow-judge-llamafile/resolve/main/flow-judge.llamafile"
)


class DownloadError(Exception):
    pass

class LlamafileConfig(ModelConfig):
    def __init__(
        self,
        model_id: str,
        generation_params: Dict[str, Any],
        model_filename: str = "flow-judge.llamafile",
        cache_dir: str = os.path.expanduser("~/.cache/flow-judge"),
        port: int = 8085,
        disable_kv_offload: bool = False,
        llamafile_kvargs: str = "",
        **kwargs: Any
    ):
        super().__init__(model_id, ModelType.LLAMAFILE, generation_params, **kwargs)
        self.model_filename = model_filename
        self.cache_dir = cache_dir
        self.port = port
        self.disable_kv_offload = disable_kv_offload
        self.llamafile_kvargs = llamafile_kvargs

class Llamafile(BaseFlowJudgeModel, AsyncBaseFlowJudgeModel):
    def __init__(
        self,
        model: str = None,
        generation_params: dict[str, Any] = None,
        cache_dir: str = os.path.expanduser("~/.cache/flow-judge"),
        port: int = 8085,
        disable_kv_offload: bool = False,
        llamafile_kvargs: str = "",
        **kwargs: Any
    ):
        default_model_id = "sariola/flow-judge-llamafile"
        model = model or default_model_id

        default_generation_params = {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_tokens": 2000,
            "context_size": 8192,
            "gpu_layers": 34,
            "thread_count": os.cpu_count() or 1,
            "batch_size": 32, # here batch doesn't mean parallel requests, it's just the batch size for the llamafile server
            "max_concurrent_requests": 1,
            "stop": ["<|endoftext|>"]
        }
        generation_params = generation_params or default_generation_params

        config = LlamafileConfig(
            model_id=model,
            generation_params=generation_params,
            model_filename="flow-judge.llamafile",
            cache_dir=cache_dir,
            port=port,
            disable_kv_offload=disable_kv_offload,
            llamafile_kvargs=llamafile_kvargs,
            **kwargs
        )

        super().__init__(model, "llamafile", generation_params, **kwargs)

        self.generation_params = generation_params
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
        self.llamafile_kvargs = config.llamafile_kvargs

        self.metadata = {
            "model_id": model,
            "model_type": "llamafile",
        }

    def is_server_running(self):
        try:
            self.sync_client.models.list()
            return True
        except Exception:
            return False

    def download_llamafile(self):
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
                raise DownloadError(f"Failed to download llamafile: {str(e)}")

            except Exception as e:
                logging.error(f"Unexpected error during download: {str(e)}")
                if os.path.exists(llamafile_path):
                    os.remove(llamafile_path)
                raise DownloadError(f"Unexpected error during download: {str(e)}")

        try:
            os.chmod(llamafile_path, 0o755)
            logging.debug(f"Set executable permissions for {llamafile_path}")
        except OSError as e:
            logging.error(f"Failed to set executable permissions: {str(e)}")
            raise DownloadError(f"Failed to set executable permissions: {str(e)}")

        if not os.path.exists(llamafile_path):
            raise DownloadError("Llamafile not found after download attempt")

        return llamafile_path

    def start_llamafile_server(self):
        logging.info("Starting llamafile server...")
        llamafile_path = self.download_llamafile()
        logging.info(f"Llamafile path: {llamafile_path}")

        if not os.path.exists(llamafile_path):
            logging.error(f"Llamafile not found at {llamafile_path}")
            raise FileNotFoundError(f"Llamafile not found at {llamafile_path}")

        # Check if the file is executable
        if not os.access(llamafile_path, os.X_OK):
            logging.error(f"Llamafile at {llamafile_path} is not executable")
            raise PermissionError(f"Llamafile at {llamafile_path} is not executable")

        command = f"sh -c '{llamafile_path} --server --host 127.0.0.1 --port {self.port} -c {self.generation_params.get('context_length', 8192)} -ngl {self.generation_params.get('gpu_layers', 34)} --temp {self.generation_params.get('temperature', 0.1)} -n {self.generation_params.get('max_tokens', 1000)} --threads {self.generation_params.get('thread_count', os.cpu_count() or 1)} --nobrowser -b {self.generation_params.get('batch_size', 1)} --parallel {self.generation_params.get('max_concurrent_requests', 1)} --cont-batching'"

        if self.generation_params.get("disable_kv_offload", False):
            command += " -nkvo"
            logging.info("KV offloading disabled")

        extra_args = self.generation_params.get("llamafile_kvargs", "")
        if extra_args:
            command += f" {extra_args}"
            logging.info(f"Additional arguments added: {extra_args}")

        logging.info(f"Starting llamafile server with command: {command}")

        def log_output(pipe, log_func):
            for line in iter(pipe.readline, ""):
                log_func(line.strip())

        try:
            self.llamafile_process = subprocess.Popen(
                shlex.split(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                text=True,
            )
            logging.info(f"Subprocess started with PID: {self.llamafile_process.pid}")

            # Start threads to log stdout and stderr in real-time
            threading.Thread(
                target=log_output, args=(self.llamafile_process.stdout, logging.info), daemon=True
            ).start()
            threading.Thread(
                target=log_output, args=(self.llamafile_process.stderr, logging.info), daemon=True
            ).start()

            # Wait for the server to start or timeout after 60 seconds
            start_time = time.time()
            while time.time() - start_time < 60:
                if self.is_server_running():
                    logging.info("Llamafile server started successfully")
                    return  # Exit the method, keeping the server running
                time.sleep(1)
                logging.debug(
                    f"Waiting for server to start... (Elapsed: {time.time() - start_time:.2f}s)"
                )

                # Check if the process has terminated
                if self.llamafile_process.poll() is not None:
                    stdout, stderr = self.llamafile_process.communicate()
                    logging.error(
                        f"Llamafile process terminated unexpectedly. Exit code: {self.llamafile_process.returncode}"
                    )
                    logging.error(f"Stdout: {stdout}")
                    logging.error(f"Stderr: {stderr}")
                    raise RuntimeError(
                        f"Llamafile process terminated unexpectedly. Exit code: {self.llamafile_process.returncode}"
                    )

            # If we've reached here, the server didn't start in time
            logging.error(f"Llamafile server failed to start within 60 seconds.")
            self.llamafile_process.terminate()
            raise RuntimeError("Llamafile server failed to start within 60 seconds.")

        except Exception as e:
            logging.exception(f"Error starting llamafile server: {str(e)}")
            if self.llamafile_process:
                logging.info("Terminating llamafile process due to startup error")
                self.llamafile_process.terminate()
            raise

    def stop_llamafile_server(self):
        if self.llamafile_process:
            self.llamafile_process.terminate()
            time.sleep(1)
            self.llamafile_process.kill()
            self.llamafile_process = None

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
            "max_tokens": self.generation_params.get("max_tokens", 1000),
            "top_p": self.generation_params.get("top_p", 0.95),
            "temperature": self.generation_params.get("temperature", 0.1),
            "stop": self.generation_params.get("stop", ["<|endoftext|>"]),
        }

    def _batch_generate(self, prompts: List[str], use_tqdm: bool = True, **kwargs: Any) -> List[str]:
        self._ensure_server_running()
        return [self._generate(prompt) for prompt in prompts]

    async def _async_batch_generate(
        self, prompts: List[str], use_tqdm: bool = True, **kwargs: Any
    ) -> List[str]:
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
        self._context_depth += 1
        self._ensure_server_running()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context_depth -= 1
        if self._context_depth == 0:
            self.stop_llamafile_server()
            self._server_running = False

    def __del__(self):
        try:
            if hasattr(self, "_server_running") and self._server_running:
                self.stop_llamafile_server()
        except:
            pass  # Ignore any errors during deletion


class LlamafileError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class DownloadError(Exception):
    pass
