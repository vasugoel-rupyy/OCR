# Project Structure

This document describes the refactored directory structure of the OCR Pipeline project.

## Directory Layout

```
OCR/
├── ocr_pipeline/              # Main package
│   ├── __init__.py            # Package initialization (includes imghdr compatibility)
│   ├── api/                   # API layer
│   │   ├── __init__.py
│   │   ├── models.py          # Pydantic request/response models
│   │   └── server.py          # FastAPI application
│   ├── builders/              # Document Construction (NEW)
│   │   ├── __init__.py
│   │   └── document_builder.py # Maps raw extraction to typed Pydantic models
│   ├── compat/                # Compatibility shims
│   │   ├── __init__.py
│   │   └── imghdr.py          # Python 3.13+ compatibility for paddleocr
│   ├── core/                  # Core pipeline logic
│   │   ├── __init__.py
│   │   ├── pipeline.py        # Main OCRPipeline orchestrator
│   │   └── classification.py  # Document type classification
│   ├── documents/             # Document-specific extractors & Models
│   │   ├── __init__.py
│   │   ├── base.py            # Base document models
│   │   ├── aadhaar.py         # Aadhaar extraction & Pydantic model
│   │   ├── pan.py            # PAN extraction & Pydantic model
│   │   └── vehicle_rc.py      # RC extraction & Pydantic model
│   ├── ocr/                   # OCR engine wrapper
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── models.py
│   ├── preprocessing/         # Image preprocessing
│   │   ├── __init__.py
│   │   ├── corrections.py
│   │   ├── id_enhancer.py
│   │   └── pipeline.py
│   ├── quality/               # Image quality assessment
│   │   ├── __init__.py
│   │   └── image_quality.py
│   ├── scoring/               # Confidence scoring and decision
│   │   ├── __init__.py
│   │   ├── confidence.py
│   │   └── decision.py
│   ├── segmentation/          # Document segmentation
│   │   ├── __init__.py
│   │   ├── document_detector.py
│   │   ├── region.py
│   │   ├── segmentation_pipeline.py
│   │   └── text_clustering.py
│   ├── templates/             # Template Matching (Hybrid Path)
│   │   ├── __init__.py
│   │   ├── definitions.py     # Template definitions (Aadhaar/PAN/RC)
│   │   ├── extractor.py       # Template-based field extraction
│   │   ├── library.py         # Template registry
│   │   ├── matcher.py         # Template matching logic
│   │   └── pipeline.py        # Template pipeline orchestrator
│   ├── validation/            # Validation modules
│   │   ├── __init__.py
│   │   ├── anchors.py
│   │   ├── business_rules.py
│   │   ├── distribution.py
│   │   ├── key_value.py
│   │   ├── normalization.py
│   │   └── spatial_validator.py
│   └── utils.py               # Utility functions
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── images/               # Test images
│   ├── test_quality.py
│   ├── test_segmentation.py
│   └── test_validation.py
├── config.yaml                # Configuration file
├── conftest.py                # Pytest configuration
├── run.py                     # Entry point script
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## Key Architecture Components

### 1. Document Builders (`ocr_pipeline/builders/`)
This module is responsible for converting raw, untyped dictionary data from OCR extraction into strictly typed Pydantic models. It handles:
- Field-level confidence assignment.
- Data type conversion.
- Final document assembly (AadhaarDocument, PanDocument, etc.).

### 2. Template Matching (`ocr_pipeline/templates/`)
A generic template matching system that allows for:
- Fast-path extraction for known high-quality document layouts.
- Alignment based on key anchor points.
- Fallback to full OCR if templates don't match.
*(Note: Can be enabled/disabled via pipeline configuration)*

### 3. Core Logic Organization
- **Orchestrator**: `ocr_pipeline/core/pipeline.py` manages the flow between Quality Gate -> Template Matcher -> Fallback OCR -> Document Builder.
- **Classification**: `ocr_pipeline/core/classification.py` handles automatic document type detection.

### 4. Compatibility Handling
- **`ocr_pipeline/compat/`**: Provides strict compatibility shims for Python 3.13+, ensuring dependencies like PaddleOCR continue to function even when standard library modules like `imghdr` are deprecated.

## Import Patterns

```python
from ocr_pipeline import OCRPipeline
from ocr_pipeline.api.models import OCRRequest, OCRResponse
from ocr_pipeline.builders.document_builder import DocumentBuilder
```

## Running the Application

### Development
```bash
python run.py
```

### Production (after installation)
```bash
ocr-pipeline-api
```

### API Documentation
Once running, access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
