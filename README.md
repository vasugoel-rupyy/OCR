# OCR Pipeline

A high-performance, self-hosted OCR system designed to extract structured data from Indian Identity Documents (Aadhaar, PAN, Vehicle RC). It utilizes a multi-layered approach involving image preprocessing, document detection, dual-pass OCR, and a 10-component validation system.

## Features

### 🧠 Core Orchestration
- **Dual-Pass OCR**: Intelligently combines standard and enhanced preprocessing passes to maximize extraction accuracy.
- **Multilingual Support**: Supports English and Hindi/Devanagari text extraction with specialized numeral normalization.
- **Document Classification**: Automatically identifies document types based on keyword frequency and spatial patterns.

### 📉 10-Component Scoring Model
The system explains its decisions via a weighted confidence score [0-1] based on:
1.  **Image Quality**: Blur, brightness, and contrast checks.
2.  **OCR Confidence**: Character-level metadata from PaddleOCR.
3.  **Regex Pattern Match**: Verification against strict document formats.
4.  **Fuzzy Matching**: Anchor word detection (e.g., "Father's Name").
5.  **Layout Analysis**: Verification of physical field locations.
6.  **Key-Value Pair Proximity**: Spatial relationship between keys and values.
7.  **Cross-Field Consistency**: Logical checks between related fields.
8.  **Schema Compliance**: Ensuring all mandatory fields are found.
9.  **Token Distribution**: Analysis of numeric vs. alphabetic ratios.
10. **Spatial Compactness**: Prevents cross-region mixing of fields.

### 🖼️ Image Intelligence
- **Quality Gate**: Rejects blurry or poorly lit images before heavy processing.
- **Auto-Deskewing**: Hough line transform to correct rotated documents.
- **ID Enhancer**: Specialized filters to sharpen small fonts and improve contrast.

## Requirements

- Python 3.8+ (Python 3.13+ supported with compatibility shims)
- See `requirements.txt` for full dependency list

## Quick Start

### Installation

1. **Clone the repository** (if applicable):
```bash
git clone <https://github.com/vasugoel-rupyy/OCR>
cd OCR
```

2. **Create and activate virtual environment** (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the API Server

The system provides a FastAPI-based server for processing images via URL:

```bash
python run.py
```

The server will be available at `http://localhost:8000`.

**API Documentation:**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

**Endpoints:**
- `POST /ocr/process_url`: Standard endpoint. Automatically detects type unless `document_type` is provided in the JSON body.
- `POST /ocr/process_url/{doc_type}`: Explicitly sets the document type (e.g., `/ocr/process_url/aadhaar`).

**Example Request:**
```bash
curl -X POST "http://localhost:8000/ocr/process_url" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/document.jpg",
    "document_type": "auto"
  }'
```

### Command Line Usage

#### 1. Auto-Detect Document Type (Recommended)
The pipeline automatically classifies the document type (Aadhaar, PAN, or RC):

```bash
python -m ocr_pipeline.core.pipeline document.jpg
```

#### 2. Process Specific Document Types
You can force a specific extractor if needed:

```bash
# Aadhaar
python -m ocr_pipeline.core.pipeline aadhaar.jpg --type aadhaar

# PAN Card
python -m ocr_pipeline.core.pipeline pan.jpg --type pan

# Vehicle RC
python -m ocr_pipeline.core.pipeline rc.jpg --type vehicle_rc
```

## Python API

```python
from ocr_pipeline import OCRPipeline

# Initialize pipeline
pipeline = OCRPipeline()

# Process with auto-detection
result = pipeline.process_document('document.jpg', document_type='auto')

# Access results
print(f"Decision: {result.decision}")
print(f"Confidence: {result.confidence.final_score:.3f}")
print(f"Extracted Fields: {result.extracted_fields}")
print(f"Document Type: {result.document_type}")
```

### Advanced Usage

```python
from ocr_pipeline import OCRPipeline
from ocr_pipeline.utils import load_config

# Load custom configuration
config = load_config('config.yaml')
pipeline = OCRPipeline(config_path='config.yaml')

# Process document
result = pipeline.process_document(
    'document.jpg',
    document_type='aadhaar'  # or 'pan', 'vehicle_rc', 'auto'
)

# Check decision and confidence
if result.decision == 'accept':
    print("Document accepted!")
    print(f"Confidence: {result.confidence.final_score:.2%}")
    print(f"Extracted: {result.extracted_fields}")
elif result.decision == 'review':
    print("Document requires manual review")
    print(f"Reasons: {result.decision_result.reasons}")
else:
    print("Document rejected")
    print(f"Reasons: {result.decision_result.reasons}")
```

## Project Structure

```
OCR/
├── ocr_pipeline/              # Main package
│   ├── __init__.py            # Package initialization (includes Python 3.13+ compatibility)
│   ├── api/                   # API layer
│   │   ├── models.py          # Pydantic request/response models
│   │   └── server.py          # FastAPI application
│   ├── core/                  # Core pipeline logic
│   │   ├── pipeline.py        # Main OCRPipeline orchestrator
│   │   └── classification.py  # Document type detection
│   ├── compat/                # Compatibility shims
│   │   └── imghdr.py          # Python 3.13+ compatibility for paddleocr
│   ├── documents/             # Document-specific extractors
│   │   ├── aadhaar.py
│   │   ├── pan.py
│   │   └── vehicle_rc.py
│   ├── ocr/                   # OCR engine wrapper
│   │   ├── engine.py
│   │   └── models.py
│   ├── preprocessing/         # Image preprocessing
│   │   ├── corrections.py
│   │   ├── id_enhancer.py
│   │   └── pipeline.py
│   ├── quality/               # Image quality assessment
│   │   └── image_quality.py
│   ├── scoring/               # Confidence scoring and decision
│   │   ├── confidence.py
│   │   └── decision.py
│   ├── segmentation/          # Document segmentation
│   │   ├── document_detector.py
│   │   ├── region.py
│   │   └── segmentation_pipeline.py
│   ├── validation/            # Validation modules
│   │   ├── anchors.py
│   │   ├── business_rules.py
│   │   ├── distribution.py
│   │   └── spatial_validator.py
│   └── utils.py               # Utility functions
├── tests/                      # Test suite
│   ├── images/                # Test images
│   ├── test_quality.py
│   ├── test_segmentation.py
│   └── test_validation.py
├── config.yaml                # Configuration file
├── run.py                     # Entry point script
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Configuration

The system is configured via `config.yaml`. Key configuration sections:

- **Quality**: Image quality thresholds (blur, brightness, contrast)
- **OCR**: PaddleOCR engine settings
- **Preprocessing**: Image enhancement parameters
- **Scoring**: Confidence score weights
- **Decision**: Accept/review/reject thresholds
- **Validation**: Business rules and field patterns

See `config.yaml` for detailed configuration options.

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test files:

```bash
pytest tests/test_quality.py
pytest tests/test_validation.py
pytest tests/test_segmentation.py
```

## Development

### Installing in Development Mode

```bash
pip install -e .
```

### Running Tests with Coverage

```bash
pytest --cov=ocr_pipeline --cov-report=html
```

## Python Version Compatibility

- **Python 3.8-3.12**: Fully supported
- **Python 3.13+**: Supported with automatic compatibility shims for deprecated modules (e.g., `imghdr`)

The package automatically handles Python 3.13+ compatibility by providing shims for modules removed in newer Python versions.

## API Response Format

```json
{
  "status": "success",
  "document_type": "aadhaar",
  "decision": "accept",
  "confidence_score": 0.923,
  "reason": "Confidence score 0.923 exceeds accept threshold 0.85",
  "extracted_fields": {
    "aadhaar_number": "1234 5678 9012",
    "name": "JOHN DOE",
    "date_of_birth": "1990-01-15"
  },
  "processing_time": 2.45
}
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'imghdr'** (Python 3.13+)
   - This is automatically handled by the compatibility shim. Ensure you're importing `ocr_pipeline` before any paddleocr imports.

2. **Import errors after refactoring**
   - Make sure you're using the new import paths: `from ocr_pipeline import OCRPipeline`

3. **Configuration file not found**
   - Ensure `config.yaml` is in the project root directory.

## License

MIT License

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Tests are added for new features
- Documentation is updated accordingly
