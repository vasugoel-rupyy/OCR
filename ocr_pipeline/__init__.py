"""OCR Pipeline - Production-ready OCR system for Indian identity documents."""

import sys
import os

if sys.version_info >= (3, 13):
    try:
        import imghdr
    except ImportError:
        pass

__version__ = "1.0.0"

from .core.pipeline import OCRPipeline, PipelineResult
from .utils import load_config, setup_logging

__all__ = ["OCRPipeline", "PipelineResult", "load_config", "setup_logging"]
