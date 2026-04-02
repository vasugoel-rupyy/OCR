import uvicorn
import shutil
import tempfile
import os
import logging
import uuid
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Header
from fastapi.responses import JSONResponse
from celery.result import AsyncResult

from ..utils import setup_logging
from .models import OCRRequest, OCRResponse
from ..core.tasks import process_document_task
from ..core.celery_app import app as celery_app

# Setup logging
setup_logging()
logger = logging.getLogger("ocr_pipeline.api")

app = FastAPI(
    title="OCR Pipeline API",
    description="Production-ready OCR Pipeline with Background Workers",
    version="1.2.0"
)

# No shared directory needed as URLs are passed directly to workers

# Middleware for Request ID tracing
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/ocr/process_url")
async def process_url(request: OCRRequest, req: Request):
    """
    Enqueue an image URL (e.g. S3 link) for OCR processing directly.
    The worker will download the file locally to transient storage.
    """
    request_id = req.state.request_id
    logger.info(f"[{request_id}] URL processing request: {request.image_url}")
    
    try:
        # Determine queue based on document type
        queue = "llm" if request.document_type == "disbursement_order" else "ocr"
        
        # Enqueue task with URL instead of local path
        task = process_document_task.apply_async(
            args=[request.image_url, request.document_type, request.webhook_url, request_id],
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
