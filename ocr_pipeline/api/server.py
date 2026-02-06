"""FastAPI server for OCR Pipeline."""

import uvicorn
from fastapi import FastAPI, HTTPException
import requests
import shutil
import tempfile
import os
import logging
from typing import Dict, Any

from ..core.pipeline import OCRPipeline
from .models import OCRRequest, OCRResponse

from fastapi.concurrency import run_in_threadpool

# Config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr_pipeline.api")

app = FastAPI(
    title="OCR Pipeline API",
    description="Extract data from Indian Identity Documents (Aadhaar, PAN, Vehicle RC)",
    version="1.0.0"
)

# Global pipeline instance
pipeline: OCRPipeline | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on server startup."""
    global pipeline
    logger.info("Initializing OCR Pipeline...")
    # Initialize pipeline once on startup
    pipeline = OCRPipeline()
    logger.info("OCR Pipeline initialized.")


def _process_and_respond(image_url: str, doc_type: str) -> OCRResponse:
    """Helper to process an image and return response data.
    
    This function is synchronous and should be run in a threadpool
    to avoid blocking the main event loop.
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    tmp_path = None
    try:
        # 1. Download image
        logger.info(f"Fetching image from: {image_url}")
        response = requests.get(image_url, stream=True, timeout=15)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        suffix = ".jpg" 
        if "png" in content_type: 
            suffix = ".png"
        elif "jpeg" in content_type: 
            suffix = ".jpg"
        elif "webp" in content_type: 
            suffix = ".webp"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(response.raw, tmp_file)
            tmp_path = tmp_file.name
        
        logger.info(f"Image saved to {tmp_path} (Using doc_type: {doc_type})")

        # 2. Process
        logger.info(f"Running processing pipeline with type: {doc_type}")
        result = pipeline.process_document(tmp_path, document_type=doc_type)
        
        # 3. Extract Reason
        reason = "-"
        if hasattr(result, 'decision_result') and result.decision_result:
            if result.decision_result.reasons:
                reason = result.decision_result.reasons[0]
            
        # 4. Response
        response_data = OCRResponse(
            status="success",
            document_type=result.document_type,
            decision=result.decision,
            confidence_score=round(result.confidence.final_score, 3),
            reason=reason,
            extracted_fields=result.extracted_fields,
            processing_time=round(result.processing_time, 2)
        )
        
        logger.info(f"Processing complete: {result.decision} (Score: {response_data.confidence_score})")
        return response_data
        
    except requests.RequestException as e:
        logger.error(f"Network error fetching image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}")


@app.post("/ocr/process_url", response_model=OCRResponse)
async def process_url(request: OCRRequest):
    """
    Process an image from a URL.
    document_type in body overrides default 'auto'.
    """
    return await run_in_threadpool(_process_and_respond, request.image_url, request.document_type)


@app.post("/ocr/process_url/{doc_type}", response_model=OCRResponse)
async def process_url_with_type(doc_type: str, request: OCRRequest):
    """
    Process an image from a URL with a predefined document type in the path.
    Path parameter doc_type overrides anything in the body.
    """
    return await run_in_threadpool(_process_and_respond, request.image_url, doc_type)


def main():
    """Main entry point for running the server."""
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
