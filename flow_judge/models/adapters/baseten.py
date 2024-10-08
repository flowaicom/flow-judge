import os
import json
import time
import requests

from typing import Any, Dict

from .base import BaseAPIAdapter

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
        resp = requests.post(
            url=self.base_url + "/predict",
            headers={"Authorization": f"Api-Key {self.baseten_api_key}"},
            json=request_messages
        )

        return resp.json()

    def fetch_response(self, request_messages: Dict[str, Any]) -> str:
        request_body = {"messages": request_messages}
        parsed_resp = self._make_request(request_body)

        try:
            message = parsed_resp["choices"][0]["message"]["content"].strip()
            return message
        except Exception as e:
            # TODO: Log warning here (?)
            return ""
        
    def fetch_batched_response(self, request_messages: list[Dict[str, Any]]) -> list[str]:
        outputs = []
        for message in request_messages:
            request_body = {"messages": message}
            parsed_resp = self._make_request(request_body)
            outputs.append(parsed_resp["choices"][0]["message"]["content"].strip())
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

    def _make_request(self, request_messages: Dict[str, Any]) -> str:
        model_input = {"messages": request_messages}
        resp = requests.post(
            url=self.base_url + "/async_predict",
            headers={"Authorization": f"Api-Key {self.baseten_api_key}"},
            json={
                "webhook_endpoint": self.webhook_proxy_url + "/webhook",
                "model_input": model_input
            }
        )
        return resp.json()["request_id"]

    def fetch_response(self, request_messages: Dict[str, Any]) -> str:
        request_id = self._make_request(request_messages)
        stream = requests.get(f"{self.webhook_proxy_url}/listen/{request_id}", stream=True)
        message = ""
        for chunk in stream.iter_content(chunk_size=None):
            if not chunk:
                break
            elif chunk.decode() == "data: keep-alive\n\n":
                continue
            else:
                decoded_chunk = chunk.decode()
                data_and_eot = decoded_chunk.split("\n\n")
                data_chunk = data_and_eot[0]
                resp_str = data_chunk.split("data: ")[1] if data_chunk.startswith("data: ") else data_chunk

                try:
                    resp = json.loads(resp_str)
                    message = resp["data"]["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    # raise? warn?
                    continue

                if len(data_and_eot) > 1 and "\n\ndata: eot" in data_and_eot[1]:
                    stream.close()

        return message

    def fetch_batched_response(self, request_messages: list[Dict[str, Any]]) -> list[str]:
        request_ids = [self._make_request([message]) for message in request_messages]
        all_messages = []
        # Define retry parameters
        max_retries = 3
        retry_delay = 2  # seconds

        # Process the streams in batches of 4
        batch_size = 4
        for i in range(0, len(request_ids), batch_size):
            batch_ids = request_ids[i:i + batch_size]

            # Open a stream for each request ID in the current batch
            open_streams = []
            for request_id in batch_ids:
                for attempt in range(max_retries):
                    try:
                        stream = requests.get(f"{self.webhook_proxy_url}/listen/{request_id}", stream=True)
                        open_streams.append(stream)
                        break  # Exit retry loop on success
                    except requests.RequestException as e:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)  # Wait before retrying
                        else:
                            print(f"Failed to open stream for request_id {request_id} after {max_retries} attempts: {e}")
                            open_streams.append(None)  # Append None to maintain index consistency

            # Default to "" for eval parser to add failed feedback response (-1)
            batch_messages = ["" for _ in range(len(batch_ids))]

            # Handle streams concurrently
            while any(open_streams):
                for index, stream in enumerate(open_streams):
                    if stream is None:
                        continue  # Skip streams that failed to open

                    try:
                        for chunk in stream.iter_content(chunk_size=None):
                            if not chunk:
                                raise StopIteration()
                            elif chunk.decode() == "data: keep-alive\n\n":
                                continue
                            else:
                                decoded_chunk = chunk.decode()
                                data_and_eot = decoded_chunk.split("\n\n")
                                data_chunk = data_and_eot[0]
                                resp_str = data_chunk.split("data: ")[1] if data_chunk.startswith("data: ") else data_chunk
                                
                                try:
                                    resp = json.loads(resp_str)
                                    batch_messages[index] = resp["data"]["choices"][0]["message"]["content"].strip()
                                except Exception as e:
                                    # raise? warn?
                                    continue

                                if len(data_and_eot) > 1 and "\n\ndata: eot" in data_and_eot[1]:
                                    stream.close()
                                    open_streams[index] = None

                    except (StopIteration, requests.RequestException) as e:
                        open_streams[index] = None
                        stream.close()

                # Remove closed or failed streams
                open_streams = [s for s in open_streams if s is not None]

            all_messages.extend(batch_messages)

        return all_messages