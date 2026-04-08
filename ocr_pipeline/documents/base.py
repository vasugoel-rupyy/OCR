"""Base document processor classes and Pydantic models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np
from ..ocr.models import OCRResult


class Decision(str, Enum):
    """Decision for document validation."""
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class FieldValue(BaseModel):
    """Represents a field value with its confidence score."""
    value: Optional[str] = None
    confidence: float = 0.0
    
    class Config:
        frozen = True


class BaseDocument(BaseModel):
    """Base model for all document types."""
    template_used: Optional[str] = None
    template_confidence: float = 0.0
    overall_confidence: float = 0.0
    decision: Decision = Decision.REVIEW
    

    raw_extraction: Optional[Dict[str, Any]] = Field(default=None, exclude=True)


class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    def __init__(self, config: Dict):
        """Initialize document processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    @abstractmethod
    def extract_fields(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Extract structured fields from OCR result.
        
        Args:
            ocr_result: OCR result object
            
        Returns:
            Dictionary of extracted fields
        """
        pass
    
    @abstractmethod
    def validate_fields(self, fields: Dict[str, Any]) -> Dict:
        """Validate extracted fields.
        
        Args:
            fields: Extracted fields
            
        Returns:
            Validation results
        """
        pass
    
    @abstractmethod
    def validate_layout(self, ocr_result: OCRResult, image_shape: tuple) -> Dict:
        """Validate document layout.
        
        Args:
            ocr_result: OCR result
            image_shape: (height, width) of image
            
        Returns:
            Layout validation results
        """
        pass
    
    @abstractmethod
    def check_consistency(self, fields: Dict[str, Any]) -> Dict:
        """Check field consistency.
        
        Args:
            fields: Extracted fields
            
        Returns:
            Consistency check results
        """
        pass
    
    @abstractmethod
    def get_document_type(self) -> str:
        """Get document type identifier.
        
        Returns:
            Document type string
        """
        pass
