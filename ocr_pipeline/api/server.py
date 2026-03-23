"""FastAPI server for OCR Pipeline."""
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import requests
import httpx
from .ollama_service import OllamaExtractor
import shutil
import tempfile
import os
import logging
import traceback
from typing import Dict, Any, Optional

from ..core.pipeline import OCRPipeline
from ..utils import setup_logging
from .models import OCRRequest, OCRResponse

# Try to import PDF utilities
try:
    from ..utils import is_pdf, pdf_to_image_file, is_pdf_supported
    PDF_AVAILABLE = is_pdf_supported()
except ImportError:
    PDF_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("PDF support not available. Install pdf2image to enable PDF processing.")

# Config
setup_logging()
logger = logging.getLogger("ocr_pipeline.api")

app = FastAPI(
    title="OCR Pipeline API",
    description="Extract data from Indian Identity Documents (Aadhaar, PAN, Vehicle RC, Disbursement Order)",
    version="1.1.0"
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


async def _run_pipeline_core(file_path: str, doc_type: str) -> OCRResponse:
    """Core logic to run the pipeline on a local file and build response."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    # Normalize document type
    doc_type = doc_type.lower().strip()
    if doc_type in ['aadhar', 'adhara', 'adhar']:
        doc_type = 'aadhaar'
    elif doc_type in ['rc', 'vehicle', 'car_rc']:
        doc_type = 'vehicle_rc'
    elif doc_type in ['do', 'disbursement']:
        doc_type = 'disbursement_order'

    try:
        # 3. Process with OCR pipeline
        logger.info(f"Running processing pipeline with type: {doc_type}")
        
        # Offload the blocking pipeline call to a thread pool
        from fastapi.concurrency import run_in_threadpool
        result = await run_in_threadpool(pipeline.process_document, file_path, document_type=doc_type)
        
        # 4. Extract reason
        reason = "-"
        if hasattr(result, 'decision_result') and result.decision_result:
            if result.decision_result.reasons:
                reason = result.decision_result.reasons[0]
        
        # 5. Log processing details
        logger.info(f"Processing complete: {result.decision} (Score: {result.confidence.final_score:.3f})")
        
        # 6. Build response
        return OCRResponse(
            status="success",
            document_type=result.document_type,
            decision=result.decision,
            confidence_score=round(result.confidence.final_score, 3),
            reason=reason,
            extracted_fields=result.extracted_fields,
            processing_time=round(result.processing_time, 2),
            extraction_method=result.ocr_stats.get('method', 'ocr'),
            template_confidence=result.ocr_stats.get('template_score'),
            fallback_confidence=result.ocr_stats.get('fallback_score'),
            structured_document=result.structured_document,
            raw_ocr_text=result.full_text
        )

    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        error_trace = traceback.format_exc()
        logger.error(f"{error_msg}\n{error_trace}")
        return OCRResponse(
            status="error",
            document_type=doc_type,
            decision="error",
            confidence_score=0.0,
            reason=error_msg,
            extracted_fields={},
            processing_time=0.0,
            extraction_method="error",
            error_details=error_trace
        )


async def _process_and_webhook(file_path: str, doc_type: str, webhook_url: str):
    """Background task to process document and send result to webhook."""
    try:
        # Run OCR pipeline, which natively extracts fields using Ollama Mistral if it's a disbursement order
        result = await _run_pipeline_core(file_path, doc_type)
                
        # Send to webhook
        logger.info(f"Sending async pipeline result to webhook: {webhook_url}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(webhook_url, json=result.model_dump(mode='json') if hasattr(result, 'model_dump') else result.dict())
            
    except Exception as e:
        logger.error(f"Error in background webhook task: {str(e)}")
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass


async def _handle_pdf_conversion(tmp_path: str) -> str:
    """Handles PDF conversion to image if necessary. Returns the path to the image to process."""
    if PDF_AVAILABLE and is_pdf(tmp_path):
        logger.info("PDF detected, converting first page to image...")
        try:
            return pdf_to_image_file(tmp_path, dpi=300, page=1)
        except Exception as pdf_error:
            error_msg = f"Failed to convert PDF to image: {str(pdf_error)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
    return tmp_path


@app.post("/ocr/process_url")
async def process_url(request: OCRRequest, background_tasks: BackgroundTasks):
    """Process an image from a URL."""
    tmp_path = None
    try:
        logger.info(f"Fetching file from: {request.image_url}")
        response = requests.get(request.image_url, stream=True, timeout=15)
        response.raise_for_status()
        
        url_path = request.image_url.split('?')[0]
        suffix = os.path.splitext(url_path)[1] if '.' in os.path.basename(url_path) else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(response.raw, tmp_file)
            tmp_path = tmp_file.name
        
        final_path = tmp_path
        
        if request.webhook_url:
            background_tasks.add_task(_process_and_webhook, final_path, request.document_type, request.webhook_url)
            return JSONResponse(status_code=202, content={"status": "processing", "message": "Background job started"})
            
        try:
            return await _run_pipeline_core(final_path, request.document_type)
        finally:
            if os.path.exists(final_path):
                os.remove(final_path)
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


@app.post("/ocr/process_file")
async def process_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_type: str = Form("auto"),
    webhook_url: Optional[str] = Form(None)
):
    """
    Upload a file directly.
    Works perfectly in Swagger (docs page) using the 'Choose File' button.
    """
    tmp_path = None
    try:
        # Save uploaded file to temp
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
            
        final_path = tmp_path
        
        if webhook_url:
            background_tasks.add_task(_process_and_webhook, final_path, document_type, webhook_url)
            return JSONResponse(status_code=202, content={"status": "processing", "message": "Background job started"})
            
        try:
            return await _run_pipeline_core(final_path, document_type)
        finally:
            if os.path.exists(final_path):
                os.remove(final_path)
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


@app.post("/ocr/process_file/{doc_type}")
async def process_file_with_type(
    doc_type: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    webhook_url: Optional[str] = Form(None)
):
    """Upload a file with a predefined document type in the URL path."""
    return await process_file(background_tasks=background_tasks, file=file, document_type=doc_type, webhook_url=webhook_url)


def main():
    """Main entry point for running the server."""
    # Set reload=True for development to pick up mounted volume changes
    # Note: Application must be passed as an import string for reload to work
    uvicorn.run("ocr_pipeline.api.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
