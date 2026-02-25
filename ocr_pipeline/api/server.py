"""FastAPI server for OCR Pipeline."""

import uvicorn
from fastapi import FastAPI, HTTPException
import requests
import shutil
import tempfile
import os
import logging
import traceback
from typing import Dict, Any

from ..core.pipeline import OCRPipeline
from ..utils import setup_logging
from .models import OCRRequest, OCRResponse

# Try to import PDF utilities
try:
    from ..utils_pdf import is_pdf, pdf_to_image_file, is_pdf_supported
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


async def _process_and_respond(image_url: str, doc_type: str) -> OCRResponse:
    """Helper to process an image/PDF and return response data."""
    # Normalize document type
    doc_type = doc_type.lower().strip()
    if doc_type in ['aadhar', 'adhara', 'adhar']:
        doc_type = 'aadhaar'
    elif doc_type in ['rc', 'vehicle', 'car_rc']:
        doc_type = 'vehicle_rc'
        
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    tmp_path = None
    tmp_image_path = None
    
    try:
        # 1. Download file
        logger.info(f"Fetching file from: {image_url}")
        response = requests.get(image_url, stream=True, timeout=15)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        # Determine file type and suffix
        is_pdf_file = False
        suffix = ".jpg"
        
        if "pdf" in content_type:
            is_pdf_file = True
            suffix = ".pdf"
        elif "png" in content_type:
            suffix = ".png"
        elif "jpeg" in content_type or "jpg" in content_type:
            suffix = ".jpg"
        elif "webp" in content_type:
            suffix = ".webp"
        
        # Download file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(response.raw, tmp_file)
            tmp_path = tmp_file.name
        
        logger.info(f"File saved to {tmp_path} (Type: {content_type})")
        
        # 2. Handle PDF conversion if needed
        final_image_path = tmp_path
        
        # Double check if it's actually a PDF by checking the file
        if is_pdf_file or (PDF_AVAILABLE and is_pdf(tmp_path)):
            if not PDF_AVAILABLE:
                error_msg = "PDF file detected but pdf2image is not installed. Please install pdf2image and poppler-utils."
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            
            logger.info("PDF detected, converting first page to image...")
            try:
                # Convert first page to image
                tmp_image_path = pdf_to_image_file(tmp_path, dpi=300, page=1)
                final_image_path = tmp_image_path
                logger.info(f"PDF converted to image: {tmp_image_path}")
            except Exception as pdf_error:
                error_msg = f"Failed to convert PDF to image: {str(pdf_error)}"
                error_trace = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_trace}")
                raise HTTPException(status_code=400, detail=error_msg)
        
        # 3. Process with OCR pipeline
        logger.info(f"Running processing pipeline with type: {doc_type}")
        
        # Offload the blocking pipeline call to a thread pool to avoid blocking the event loop
        # This keeps the request synchronous to the user but helps with concurrency
        from fastapi.concurrency import run_in_threadpool
        result = await run_in_threadpool(pipeline.process_document, final_image_path, document_type=doc_type)
        
        # 4. Extract reason
        reason = "-"
        if hasattr(result, 'decision_result') and result.decision_result:
            if result.decision_result.reasons:
                reason = result.decision_result.reasons[0]
        
        # 5. Log the full payload (as requested)
        # Move raw OCR text and intermediate details to logs instead of API response
        logger.info(f"Processing complete: {result.decision} (Score: {result.confidence.final_score:.3f})")
        logger.info(f"FULL EXTRACTED TEXT:\n{result.full_text}")
        logger.info(f"EXTRACTED FIELDS: {result.extracted_fields}")
        
        # 6. Build response with lean payload
        response_data = OCRResponse(
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
            raw_ocr_text=None,  # Moved to logs
            error_details=None
        )
        
        return response_data
        
    except requests.RequestException as e:
        error_msg = f"Network error fetching file: {str(e)}"
        error_trace = traceback.format_exc()
        logger.error(f"{error_msg}\n{error_trace}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        error_trace = traceback.format_exc()
        logger.error(f"{error_msg}\n{error_trace}")
        
        # Return error response with details
        return OCRResponse(
            status="error",
            document_type=doc_type,
            decision="error",
            confidence_score=0.0,
            reason=error_msg,
            extracted_fields={},
            processing_time=0.0,
            extraction_method="error",
            raw_ocr_text=None,
            error_details=error_trace
        )
    
    finally:
        # Clean up temporary files
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug(f"Removed temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {tmp_path}: {e}")
        
        if tmp_image_path and os.path.exists(tmp_image_path) and tmp_image_path != tmp_path:
            try:
                os.remove(tmp_image_path)
                logger.debug(f"Removed temporary image file: {tmp_image_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp image file {tmp_image_path}: {e}")


@app.post("/ocr/process_url", response_model=OCRResponse)
async def process_url(request: OCRRequest):
    """
    Process an image from a URL.
    document_type in body overrides default 'auto'.
    """
    return await _process_and_respond(request.image_url, request.document_type)


@app.post("/ocr/process_url/{doc_type}", response_model=OCRResponse)
async def process_url_with_type(doc_type: str, request: OCRRequest):
    """
    Process an image from a URL with a predefined document type in the path.
    Path parameter doc_type overrides anything in the body.
    """
    return await _process_and_respond(request.image_url, doc_type)


def main():
    """Main entry point for running the server."""
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
