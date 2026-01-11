import os
import io
import time
import psutil
import base64
import uvicorn
import pandas as pd
from typing import List
from pydantic import BaseModel
from main import LogClassifier
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, Response
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Initialize the FastAPI app with metadata
app = FastAPI(
    title="Vigilis Log Classifier",
    description="API and UI for classifying log messages using Regex, BERT, and LLM.",
    version="1.0.0",
)

# Mount static files directory to serve CSS and other static assets
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize Jinja2 templates for serving HTML pages
templates = Jinja2Templates(directory="app/templates")

# Initialize the classifier globally to reuse loaded models across requests.
# This prevents reloading heavy models (BERT) on every request.
classifier: LogClassifier = LogClassifier()

# --- Prometheus Metrics Configuration ---
registry: CollectorRegistry = CollectorRegistry()

REQUEST_COUNT: Counter = Counter(
    "app_request_count",
    "Total number of requests to the app",
    ["method", "endpoint"],
    registry=registry,
)

REQUEST_LATENCY: Histogram = Histogram(
    "app_request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"],
    registry=registry,
)

PREDICTION_COUNT: Counter = Counter(
    "model_prediction_count",
    "Count of predictions for each class",
    ["prediction"],
    registry=registry,
)

INPUT_LENGTH: Histogram = Histogram(
    "app_input_length_chars",
    "Length of input text in characters",
    buckets=[0, 50, 100, 200, 500, 1000, 2000, 5000, float("inf")],
    registry=registry,
)

ERROR_COUNT: Counter = Counter(
    "app_error_count",
    "Total number of errors",
    ["type"],
    registry=registry,
)

MEMORY_USAGE: Gauge = Gauge(
    "app_memory_usage_bytes",
    "Memory usage of the application in bytes",
    registry=registry,
)


def update_system_metrics() -> None:
    """
    Updates the system metrics gauge (memory usage) with the current process RSS.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    MEMORY_USAGE.set(memory_info.rss)


# --- Pydantic Models ---


class LogRequest(BaseModel):
    """
    Request schema for classifying a single log message.
    """

    source: str
    log_message: str


class LogResponse(BaseModel):
    """
    Response schema for a classification result.
    """

    source: str
    log_message: str
    label: str


class BatchLogRequest(BaseModel):
    """
    Request schema for classifying multiple log messages in a batch.
    """

    logs: List[LogRequest]


class BatchLogResponse(BaseModel):
    """
    Response schema for batch classification results.
    """

    results: List[LogResponse]


# --- Web UI Endpoints ---


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> Response:
    """
    Renders the home page with the log classification form.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        Response: The rendered HTML template for the home page.
    """
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time: float = time.time()

    try:
        update_system_metrics()
        response: Response = templates.TemplateResponse(
            request=request, name="index.html", context={"result": None}
        )
        REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
        return response
    except Exception as e:
        ERROR_COUNT.labels(type=type(e).__name__).inc()
        raise e


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> Response:
    """
    Handles CSV file upload, processes logs, and returns a page with results table and download link.

    Args:
        request (Request): The incoming HTTP request.
        file (UploadFile): The uploaded CSV file containing 'source' and 'log_message'.

    Returns:
        Response: The HTML page with results.
    """
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    t0: float = time.time()

    try:
        # Read the uploaded file into a DataFrame
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if "source" not in df.columns or "log_message" not in df.columns:
            # Render error in template instead of raising text-only 400
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "error": "CSV must contain 'source' and 'log_message' columns.",
                },
            )

        # Convert to list of dicts for batch processing
        logs_data = df[["source", "log_message"]].to_dict("records")
        labels = classifier.batch_classify(logs_data)

        # Add labels to the DataFrame
        df["label"] = labels

        results = df.to_dict(orient="records")

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        csv_data = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")

        t1: float = time.time()

        # Update metrics
        for label in labels:
            PREDICTION_COUNT.labels(prediction=str(label)).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(t1 - t0)
        update_system_metrics()

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "results": results,
                "csv_data": csv_data,
                "filename": "labelled_logs.csv",
                "uploaded_filename": file.filename,
            },
        )

    except Exception as e:
        ERROR_COUNT.labels(type=type(e).__name__).inc()
        print(f"Error processing file: {e}")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"error": f"Error processing file: {str(e)}"},
        )


@app.get("/metrics")
def metrics() -> Response:
    """
    Exposes Prometheus metrics for scraping.

    Returns:
        Response: The current metrics registry in Prometheus text format.
    """
    update_system_metrics()
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


# --- API Endpoints ---


@app.post("/classify", response_model=LogResponse)
def classify_log_api(request: LogRequest) -> LogResponse:
    """
    API endpoint to classify a single log message.

    Args:
        request (LogRequest): The JSON body containing source and log_message.

    Returns:
        LogResponse: The classification result.

    Raises:
        HTTPException: If an error occurs during classification.
    """
    REQUEST_COUNT.labels(method="POST", endpoint="/classify").inc()
    t0: float = time.time()
    try:
        label: str = classifier.classify_message(request.source, request.log_message)

        PREDICTION_COUNT.labels(prediction=str(label)).inc()
        REQUEST_LATENCY.labels(endpoint="/classify").observe(time.time() - t0)

        return LogResponse(
            source=request.source, log_message=request.log_message, label=label
        )
    except Exception as e:
        ERROR_COUNT.labels(type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", response_model=BatchLogResponse)
def classify_batch_logs_api(request: BatchLogRequest) -> BatchLogResponse:
    """
    API endpoint to classify a batch of log messages.

    Args:
        request (BatchLogRequest): The JSON body containing a list of logs.

    Returns:
        BatchLogResponse: A list of classification results.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    REQUEST_COUNT.labels(method="POST", endpoint="/classify/batch").inc()
    t0: float = time.time()
    try:
        # Convert Pydantic models to dicts for the classifier
        logs_data: List[dict] = [log.model_dump() for log in request.logs]
        labels: List[str] = classifier.batch_classify(logs_data)

        results: List[LogResponse] = []
        for log, label in zip(request.logs, labels):
            PREDICTION_COUNT.labels(prediction=str(label)).inc()
            results.append(
                LogResponse(source=log.source, log_message=log.log_message, label=label)
            )

        REQUEST_LATENCY.labels(endpoint="/classify/batch").observe(time.time() - t0)
        return BatchLogResponse(results=results)

    except Exception as e:
        ERROR_COUNT.labels(type=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting FastAPI app...", flush=True)
    uvicorn.run("app.app:app", host="0.0.0.0", port=8000, reload=True)
