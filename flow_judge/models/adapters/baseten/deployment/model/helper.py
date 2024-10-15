import asyncio
import logging
import os
import threading

import httpx

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_CHECK_INTERVAL = 5  # seconds


def log_subprocess_output(process):
    """Process logs for the vLLM subprocess."""
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            logger.info(f"vLLM subprocess stdout: {output.strip()}")
    rc = process.poll()
    if rc != 0:
        for error_output in process.stderr.readlines():
            logger.error(f"vLLM subprocess stderr: {error_output.strip()}")


async def monitor_vllm_server_health(vllm_server_url, health_check_interval):
    """Health check for the vLLM server."""
    assert vllm_server_url is not None, "vllm_server_url must not be None"
    try:
        async with httpx.AsyncClient() as client:
            while True:
                response = await client.get(f"{vllm_server_url}/health")
                if response.status_code != 200:
                    raise RuntimeError("vLLM is unhealthy")
                await asyncio.sleep(health_check_interval)
    except Exception as e:
        logging.error(
            f"vLLM has gone into an unhealthy state due to error: {e}, restarting service now..."
        )
        os._exit(1)


async def monitor_vllm_engine_health(vllm_engine, health_check_interval):
    """Health check for the vLLM engine."""
    assert vllm_engine is not None, "vllm_engine must not be None"
    try:
        while True:
            await vllm_engine.check_health()
            await asyncio.sleep(health_check_interval)
    except Exception as e:
        logging.error(
            f"vLLM has gone into an unhealthy state due to error: {e}, restarting service now..."
        )
        os._exit(1)


def run_background_vllm_health_check(
    use_openai_compatible_server=False,
    health_check_interval=DEFAULT_HEALTH_CHECK_INTERVAL,
    vllm_engine=None,
    vllm_server_url=None,
):
    """Background process for vLLM health checks."""
    logger.info("Starting background health check loop")
    loop = asyncio.new_event_loop()
    if use_openai_compatible_server:
        loop.create_task(monitor_vllm_server_health(vllm_server_url, health_check_interval))
    else:
        loop.create_task(monitor_vllm_engine_health(vllm_engine, health_check_interval))
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
