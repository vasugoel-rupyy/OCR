# Second script to load
"""API module for OCR Pipeline."""

from .server import app
from .models import OCRRequest, OCRResponse

__all__ = ["app", "OCRRequest", "OCRResponse"]
