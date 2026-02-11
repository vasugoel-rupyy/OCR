"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import Dict, Any, Optional, Union
from ..documents.aadhaar import AadhaarDocument
from ..documents.pan import PanDocument
from ..documents.vehicle_rc import RcDocument
from ..documents.base import BaseDocument


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
    structured_document: Optional[Union[AadhaarDocument, PanDocument, RcDocument, BaseDocument]] = None
    raw_ocr_text: Optional[str] = None
    error_details: Optional[str] = None
