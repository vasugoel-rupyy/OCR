import ipaddress
import uvicorn
import shutil
import tempfile
import os
import logging
import uuid
import time
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Header, Depends
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from ..utils import setup_logging
from .models import OCRRequest, OCRResponse
from .metrics import REQUEST_COUNT, REQUEST_LATENCY, TASK_ENQUEUED
from ..core.tasks import process_document_task
from ..core.celery_app import app as celery_app

setup_logging()
logger = logging.getLogger("ocr_pipeline.api")

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="OCR Pipeline API",
    description="Production-ready OCR Pipeline with Background Workers",
    version="1.2.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

def is_private_url(url: str) -> bool:
    """Check if a URL points to a private/internal IP address (SSRF protection)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return True
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
    except ValueError:
        # Not an IP, could be a hostname like 'localhost' or 'internal-service'
        private_hostnames = ["localhost", "127.0.0.1", "::1", "metadata.google.internal", "169.254.169.254"]
        return any(h in hostname.lower() for h in private_hostnames)

# No shared directory needed as URLs are passed directly to workers

# Middleware for Request ID tracing
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Record metrics
    endpoint = request.url.path
    REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(process_time)
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/metrics")
async def metrics():
    return JSONResponse(content=generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

@app.post("/ocr/process_url")
@limiter.limit("60/minute")  # Per-IP limit
@limiter.limit("200/minute", key_func=lambda *args: "global")  # Global limit
async def process_url(request: Request, payload: OCRRequest):
    """
    Enqueue an image URL (e.g. S3 link) for OCR processing directly.
    The worker will download the file locally to transient storage.
    """
    request_id = request.state.request_id
    logger.info(f"[{request_id}] URL processing request: {payload.image_url}")
    
    # SSRF Protection for Webhook URL
    if payload.webhook_url and is_private_url(payload.webhook_url):
        logger.warning(f"[{request_id}] Rejected private webhook URL: {payload.webhook_url}")
        raise HTTPException(status_code=400, detail="Private or internal webhook URLs are not allowed")

    try:
        # Determine queue based on document type
        queue = "llm" if payload.document_type == "disbursement_order" else "ocr"
        TASK_ENQUEUED.labels(queue=queue).inc()
        
        # Enqueue task with URL instead of local path
        task = process_document_task.apply_async(
            args=[payload.image_url, payload.document_type, payload.webhook_url, request_id],
            queue=queue
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "task_id": task.id,
                "request_id": request_id,
                "message": "URL enqueued for background processing"
            }
        )
    except Exception as e:
        logger.error(f"[{request_id}] Failed to enqueue URL task: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to enqueue file: {str(e)}")

@app.post("/ocr/process_file")
@limiter.limit("60/minute")
async def process_file(
    request: Request,
    file: UploadFile = File(...),
    document_type: str = Form("auto"),
    webhook_url: Optional[str] = Form(None)
):
    """
    Upload a file directly for OCR processing.
    The file is saved to shared storage (/llm-calls) for workers to access.
    """
    request_id = request.state.request_id
    logger.info(f"[{request_id}] File processing request: {file.filename} (Type: {document_type})")
    
    # SSRF Protection for Webhook URL
    if webhook_url and is_private_url(webhook_url):
        logger.warning(f"[{request_id}] Rejected private webhook URL: {webhook_url}")
        raise HTTPException(status_code=400, detail="Private or internal webhook URLs are not allowed")

    # Validate file size
    try:
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"[{request_id}] Rejected file too large: {file_size} bytes")
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB")
    except Exception as size_err:
        logger.error(f"[{request_id}] Could not determine file size: {size_err}")

    # Generate a safe local path in shared storage
    # Use UUID to prevent collisions
    safe_filename = f"{request_id}_{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join("/llm-calls", safe_filename)
    
    try:
        # Save uploaded file to shared storage
        os.makedirs("/llm-calls", exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"[{request_id}] File saved to shared storage: {file_path}")

        # Determine queue based on document type
        queue = "llm" if document_type == "disbursement_order" else "ocr"
        TASK_ENQUEUED.labels(queue=queue).inc()
        
        # Enqueue task with local file path
        task = process_document_task.apply_async(
            args=[file_path, document_type, webhook_url, request_id],
            queue=queue
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "task_id": task.id,
                "request_id": request_id,
                "message": "File uploaded and enqueued for processing"
            }
        )
    except Exception as e:
        logger.error(f"[{request_id}] Failed to process upload: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Internal server error while processing upload: {str(e)}")

@app.get("/ocr/status/{task_id}")
async def get_status(task_id: str):
    """
    Check the status and get results of a background task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    if task_result.state == "PENDING":
        return {"task_id": task_id, "status": "PENDING", "message": "Task is waiting in queue"}
    elif task_result.state == "STARTED":
        return {"task_id": task_id, "status": "STARTED", "message": "Task is being processed"}
    elif task_result.state == "SUCCESS":
        return task_result.result
    elif task_result.state == "FAILURE":
        return {
            "task_id": task_id,
            "status": "FAILURE",
            "error": str(task_result.info)
        }
    else:
        return {"task_id": task_id, "status": task_result.state}

def main():
    # Production settings: reload=False, multiple workers usually managed by Gunicorn/Uvicorn
    uvicorn.run("ocr_pipeline.api.server:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
