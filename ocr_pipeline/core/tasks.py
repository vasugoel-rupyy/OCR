import os
import shutil
import tempfile
import requests
import logging
from typing import Dict, Any, Optional
from .celery_app import app
from .pipeline import OCRPipeline
from .persistence import persistence
from ..api.models import OCRResponse

logger = logging.getLogger("ocr_pipeline.tasks")

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = OCRPipeline()
    return _pipeline

@app.task(bind=True, max_retries=3)
def process_document_task(self, input_path: str, document_type: str, webhook_url: Optional[str] = None, request_id: Optional[str] = None):
    """
    Background task to process a document.
    input_path can be a local file path or a URL (e.g. S3 link).
    """
    logger.info(f"Starting task {self.request.id} for {input_path} (Type: {document_type}, Request ID: {request_id})")
    
    persistence.save_result(
        task_id=self.request.id,
        request_id=request_id,
        status="STARTED",
        document_type=document_type,
        image_url=input_path
    )
    
    file_path = input_path
    is_url = input_path.startswith(('http://', 'https://'))
    tmp_file_path = None
    is_retrying = False
    
    try:
        if is_url:
            logger.info(f"Downloading file from URL: {input_path}")
            response = requests.get(input_path, stream=True, timeout=30)
            response.raise_for_status()
            
            url_base = input_path.split('?')[0]
            suffix = os.path.splitext(url_base)[1] if '.' in os.path.basename(url_base) else ".tmp"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                shutil.copyfileobj(response.raw, tmp_file)
                tmp_file_path = tmp_file.name
                file_path = tmp_file_path
                logger.info(f"File downloaded to transient path: {file_path}")

        pipeline = get_pipeline()
        
        pipeline_result = pipeline.process_document(file_path, document_type=document_type)
        
        reason = "-"
        if hasattr(pipeline_result, 'decision_result') and pipeline_result.decision_result:
            if pipeline_result.decision_result.reasons:
                reason = pipeline_result.decision_result.reasons[0]
                
        result = OCRResponse(
            status="success",
            document_type=pipeline_result.document_type,
            decision=pipeline_result.decision,
            confidence_score=round(pipeline_result.confidence.final_score, 3),
            reason=reason,
            extracted_fields=pipeline_result.extracted_fields,
            processing_time=round(pipeline_result.processing_time, 2),
            extraction_method=pipeline_result.ocr_stats.get('method', 'ocr'),
            template_confidence=pipeline_result.ocr_stats.get('template_score'),
            fallback_confidence=pipeline_result.ocr_stats.get('fallback_score'),
            structured_document=pipeline_result.structured_document,
            raw_ocr_text=None  
        ).model_dump(mode='json')
        
        result["task_id"] = self.request.id
        result["request_id"] = request_id
        
        persistence.save_result(
            task_id=self.request.id,
            request_id=request_id,
            status="SUCCESS",
            document_type=pipeline_result.document_type,
            image_url=input_path,
            result=result
        )

        if webhook_url:
            send_webhook(webhook_url, result)
            
        return result

    except Exception as e:
        logger.error(f"Task {self.request.id} failed: {str(e)}", exc_info=True)
        
        error_result = {
            "status": "error",
            "task_id": self.request.id,
            "request_id": request_id,
            "error": str(e)
        }
        
        persistence.save_result(
            task_id=self.request.id,
            request_id=request_id,
            status="FAILURE",
            document_type=document_type,
            image_url=input_path,
            error=str(e)
        )

        if webhook_url:
            send_webhook(webhook_url, error_result)
            
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            is_retrying = True
            raise self.retry(exc=e, countdown=5)
            
        return error_result
        
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
                logger.info(f"Cleaned up transient file: {tmp_file_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup {tmp_file_path}: {cleanup_error}")
        
        if not is_retrying and input_path.startswith("/llm-calls") and os.path.exists(input_path):
            try:
                os.remove(input_path)
                logger.info(f"Cleaned up shared storage file: {input_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup shared storage file {input_path}: {cleanup_error}")

def send_webhook(url: str, payload: Dict[str, Any]):
    """
    Send processing results to the provided webhook URL.
    """
    logger.info(f"Sending results to webhook: {url}")
    try:
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Secret": os.getenv("WEBHOOK_SECRET", "dev-secret")
        }
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Webhook delivery failed for {url}: {e}")
