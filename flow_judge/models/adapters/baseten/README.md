# Baseten remote execution

### Running Remotely on Baseten

Running Flow Judge on Baseten allows you to offload processing tasks from your local machine.
Using remote execution allows for running generations in parallel, multiple requests at a time.
It can significantly improve throughput and reduce overall wait times, which might be useful for larger workloads.

### Execution modes

- **Sync Mode**:
    - For smaller jobs, sync mode provides immediate feedback and is perfect for quick iterations and exploratory tasks

- **Async (Batch) Mode**:
    - Allows to submit multiple requests simultaneously, ideal for larger workloads. It allows for parallel processing of multiple requests, making it suitable for large-scale
      evaluations.

## Setup

### Installation

To use Baseten integration, ensure you install the optional `baseten` dependency:

```bash
pip install -e .[dev,baseten]
```

### Creating Baseten account

Follow the instruction on the signup page: https://app.baseten.co/signup

### Baseten API key

To use Baseten, you need to create an API key in your account.
Navigate to your account settings to generate and manage your API keys:
https://app.baseten.co/settings/account/api_keys

### Baseten webhook secret (optional)

For async (batch) execution, it is required to create webhook secret in your Baseten account.
Follow the official Baseten instructions:
https://docs.baseten.co/invoke/async-secure#creating-webhook-secrets

### Baseten GPU

The Flow Judge model can be deployed with A10G or H100 40GB on Baseten's infrastructure.
You have an option to set the `BASETEN_GPU` environment variable using either `A10G` or `H100` as the value in a notebook environment.
If in an interactive environment (CLI) you will be asked if you would like to switch to H100.
The FlowJudge models are then selected based on the architecture and GPU selection:

A10G -> Flow-Judge-v0.1-AWQ

H100 -> Flow-Judge-v0.1-FP8

## Sync execution

When running Flow Judge eval, use `Baseten` class as a model
(see [Quickstart](https://github.com/flowaicom/flow-judge?tab=readme-ov-file#quick-start)):

```python
model=Baseten()
```

That's it! :)

During the first run you will be asked to provide the Baseten API key and the GPU.
The key will be stored on your computer in `~/.trussrc` (see [truss](https://docs.baseten.co/truss-reference/overview)).
It is used to validate if the model is already deployed and deploy it if needed.

As part of the execution process we deploy [Flow Judge model](https://huggingface.co/flowaicom/Flow-Judge-v0.1-AWQ) to
your Baseten account and promote it to published
deployment.

## Async Execution

### Description

In async mode, Baseten sends model outputs to a specified webhook address that needs to be publicly accessible on the
internet.

We offer a free proxy as an effortless solution that can forward responses directly to the Flow Judge,
without exposing endpoint from your computer to the internet.
This is the default behavior when running in batch mode.

Alternatively, you can use tools like `ngrok` or `localtunnel` to expose the [same proxy](https://github.com/flowaicom/webhook-proxy) running locally on your device
next to the Flow Judge.

### Flow AI proxy (hosted)

For the hosted proxy there's no additional setup needed. You only need to configure the model instance to work in the
async mode:

```python
model=Baseten(exec_async=True, webhook_proxy_url="https://proxy.flow-ai.dev")
```

Similarly to the synchronous execution, during the first run you will be asked for the API key to your Baseten account and the GPU,
unless provided earlier. The key is used to validate and deploy the model to your Baseten account.

Additionally, when using asynchronous execution, we verify the signature of the received webhooks payloads, as
recommended by Baseten (see [official documentation](https://docs.baseten.co/invoke/async-secure)).
Because of that, during first run you will be asked to provide the webhook secret. It will be stored
in `~/.config/flow-judge/baseten_webhook_secret` and never leaves your device.

### Using proxy locally

Currently Flow Judge does not provide a standalone endpoint to expose to the Internet. Instead, you can run an instance
of our proxy on your machine and expose it's endpoint using eg. `ngrok`.

1. Download pre-built binary from the proxy releases page: https://github.com/flowaicom/webhook-proxy/releases
   or build it according to the instructions provided in the repository.
2. Run the proxy:
    ```shell
    ./proxy -addr=0.0.0.0:8000
    ```
   For more options and detailed instructions see the documentation provided in the proxy repository.
3. Expose the running proxy to the internet with eg. `ngrok`:
    ```shell
    ngrok localhost:8000
    ```
   The output of this command will provide you with the public URL.
4. Use the public URL when setting up the model instance in your Flow Judge implementation:
    ```python
    model=Baseten(exec_async=True, webhook_proxy_url="https://«ngrok url»")
    ```

#### Using Docker

In addition to the pre-built binaries, we also provide proxy Docker images.

Run the proxy with Docker:

```shell
docker pull ghcr.io/flowaicom/webhook-proxy:latest
docker run --name=flowai-proxy -d -p 8000:8000 ghcr.io/flowaicom/webhook-proxy:latest
```

Then continue with the process from point 3. from the list above.
