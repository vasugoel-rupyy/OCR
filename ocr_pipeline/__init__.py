"""OCR Pipeline - Production-ready OCR system for Indian identity documents."""
# First script to load

import sys
import os

# Python 3.13+ compatibility: imghdr.py in root provides the shim
# PaddleOCR will import it automatically from the package root
if sys.version_info >= (3, 13):
    try:
        import imghdr
    except ImportError:
        # The root imghdr.py should be available via package root
        # If not found, PaddleOCR will fail on import
        pass

__version__ = "1.0.0"

from .core.pipeline import OCRPipeline, PipelineResult
from .utils import load_config, setup_logging

__all__ = ["OCRPipeline", "PipelineResult", "load_config", "setup_logging"]
