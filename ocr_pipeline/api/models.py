"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import Dict, Any, Optional


class OCRRequest(BaseModel):
    """Request model for OCR processing."""
    image_url: str
    document_type: Optional[str] = 'auto'


class OCRResponse(BaseModel):
    """Response model for OCR processing results."""
    status: str
    document_type: str
    decision: str
    confidence_score: float
    reason: str
    extracted_fields: Dict[str, Any]
    processing_time: float
    extraction_method: Optional[str] = "fallback_ocr"
    template_confidence: Optional[float] = None
    fallback_confidence: Optional[float] = None
