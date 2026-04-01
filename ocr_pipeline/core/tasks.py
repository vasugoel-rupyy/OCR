import os
import requests
import logging
from typing import Dict, Any, Optional
from .celery_app import app
from .pipeline import OCRPipeline
from ..api.models import OCRResponse

logger = logging.getLogger("ocr_pipeline.tasks")

# Initialize pipeline lazily per worker process
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = OCRPipeline()
    return _pipeline

@app.task(bind=True, max_retries=3)
def process_document_task(self, file_path: str, document_type: str, webhook_url: Optional[str] = None, request_id: Optional[str] = None):
    """
    Background task to process a document.
    """
    logger.info(f"Starting task {self.request.id} for {file_path} (Type: {document_type}, Request ID: {request_id})")
    
    pipeline = get_pipeline()
    result = None
    
    try:
        # Run the pipeline
        pipeline_result = pipeline.process_document(file_path, document_type=document_type)
        
        # Prepare response model for consistent output
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
            raw_ocr_text=None  # Masked/Removed in production as per audit
        ).model_dump(mode='json')
        
        # Add metadata
        result["task_id"] = self.request.id
        result["request_id"] = request_id
        
        # Webhook callback
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
        
        if webhook_url:
            send_webhook(webhook_url, error_result)
            
        # Retry logic for transient failures (e.g. Ollama connection)
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            raise self.retry(exc=e, countdown=5)
            
        return error_result
        
    finally:
        # Cleanup file after processing
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup {file_path}: {cleanup_error}")

def send_webhook(url: str, payload: Dict[str, Any]):
    """
    Send processing results to the provided webhook URL.
    """
    logger.info(f"Sending results to webhook: {url}")
    try:
        # Simple API Key authentication (to be improved if needed)
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Secret": os.getenv("WEBHOOK_SECRET", "dev-secret")
        }
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Webhook delivery failed for {url}: {e}")
