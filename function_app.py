import azure.functions as func
import logging
import json
import os
from openai import AzureOpenAI
from azure.eventhub import EventHubProducerClient, EventData
from datetime import datetime
import httpx
from typing import Any, Dict
import time
from azurefunctions.extensions.http.fastapi import Request, StreamingResponse
from fastapi.responses import JSONResponse

# Allow anonymous requests
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

class HeaderCaptureClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_headers = None

    def send(self, request, *args, **kwargs):
        response = super().send(request, *args, **kwargs)
        self.last_headers = response.headers
        return response

def create_openai_client():
    http_client = HeaderCaptureClient()
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_BASE_URL"],
        http_client=http_client,
    ), http_client

# Deployment name from environment variable
deployment_name = os.environ.get("AZURE_DEPLOYMENT_NAME", "external-dekrahr")
route_path = f"openai/deployments/{deployment_name}/chat/completions"

@app.route(route=route_path, methods=[func.HttpMethod.POST])
async def aoaifn(req: Request) -> StreamingResponse:
    logging.info('Processing OpenAI proxy request')
    
    try:
        request_body = await req.json()
        logging.info(f"Request body: {json.dumps(request_body)}")

        api_version = req.query_params.get("api-version")
        if not api_version:
            return JSONResponse({"error": "api-version is required"}, status_code=400)

        client, http_client = create_openai_client()
        messages = request_body.get("messages", [])
        stream = request_body.get("stream", False)
        extra_args = {k: v for k, v in request_body.items() if k not in ["messages", "stream"]}

        start_time = time.time()

        if stream:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                stream=True,
                **extra_args
            )
            return await process_openai_stream(response, messages, http_client, start_time)
        else:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                stream=False,
                **extra_args
            )
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            headers = http_client.last_headers
            return process_openai_sync(response, messages, headers, latency_ms)

    except Exception as e:
        logging.error(f"Error in proxy function: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

def log_to_eventhub(log_data: dict):
    try:
        if "AZURE_EVENTHUB_CONN_STR" not in os.environ:
            logging.info("Event Hub connection string not configured, skipping event hub logging")
            return

        producer = EventHubProducerClient.from_connection_string(
            conn_str=os.environ["AZURE_EVENTHUB_CONN_STR"]
        )

        log_data["timestamp"] = datetime.utcnow().isoformat()
        event_data = EventData(json.dumps(log_data))

        with producer:
            batch = producer.create_batch()
            batch.add(event_data)
            producer.send_batch(batch)
            
    except Exception as e:
        logging.error(f"Failed to log to Event Hub: {str(e)}")

def process_openai_sync(response, messages, headers, latency_ms):
    try:
        content = response.choices[0].message.content if response.choices else ""

        if getattr(response, "usage", None):
            log_data = {
                "type": "completion",
                "content": content,
                "usage": response.usage.model_dump(),
                "model": response.model,
                "prompt": messages,
                "region": headers.get("x-ms-region", "unknown"),
                "latency_ms": latency_ms
            }
            log_to_eventhub(log_data)

        return JSONResponse(
            content=response.model_dump(),
            headers={'x-ms-region': headers.get("x-ms-region", "unknown")}
        )

    except Exception as e:
        logging.error(f"Error processing sync response: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def process_openai_stream(response, messages, http_client, start_time):
    headers = http_client.last_headers
    content_buffer = []
    usage_data = None
    model_name = None
    first_chunk_time = None

    async def generate():
        try:
            for chunk in response:
                chunk_dict = chunk.model_dump()
                current_time = time.time()
                nonlocal first_chunk_time
                if first_chunk_time is None:
                    first_chunk_time = current_time

                if chunk.choices and chunk.choices[0].delta.content:
                    content_buffer.append(chunk.choices[0].delta.content)
                
                if hasattr(chunk, 'model'):
                    nonlocal model_name
                    model_name = chunk.model
                if hasattr(chunk, 'usage') and chunk.usage:
                    nonlocal usage_data
                    usage_data = chunk.usage.model_dump()

                yield f"data: {json.dumps(chunk_dict)}\n\n"

        except Exception as e:
            logging.error(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            last_chunk_time = time.time()
            time_to_first_chunk = int((first_chunk_time - start_time) * 1000) if first_chunk_time else None
            streaming_duration = int((last_chunk_time - first_chunk_time) * 1000) if first_chunk_time else None
            latency_ms = int((last_chunk_time - start_time) * 1000)

            try:
                if content_buffer:
                    log_data = {
                        "type": "stream_completion",
                        "content": "".join(content_buffer),
                        "model": model_name or "unknown",
                        "usage": usage_data,
                        "prompt": messages,
                        "region": headers.get("x-ms-region", "unknown"),
                        "latency_ms": latency_ms,
                        "time_to_first_chunk_ms": time_to_first_chunk,
                        "streaming_duration_ms": streaming_duration,
                    }
                    logging.info(f"Logging streaming completion to EventHub: {json.dumps(log_data)}")
                    log_to_eventhub(log_data)
            except Exception as e:
                logging.error(f"Failed to log to EventHub: {str(e)}")

            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
            'x-ms-region': headers.get("x-ms-region", "unknown")
        }
    )

# Catch-all route to debug 404s
@app.route(route="{*any}", methods=[func.HttpMethod.GET, func.HttpMethod.POST])
async def catch_all(req: Request):
    return JSONResponse({
        "message": "Catch-all route hit",
        "path_received": str(req.url)
    })
