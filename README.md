# OCR Pipeline

A high-performance, self-hosted OCR system designed to extract structured data from Indian Identity Documents (Aadhaar, PAN, Vehicle RC). It utilizes a multi-layered approach involving image preprocessing, document detection, dual-pass OCR, and a 10-component validation system.

## Features

### 🧠 Core Orchestration

- **Structured Data Extraction**: Returns strictly typed Pydantic models for Aadhaar, PAN, and RC documents.
- **Robust Fallback OCR**: Uses advanced preprocessing and full-text OCR when template matching is disabled or fails.
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

## Installation

### Prerequisites

#### Install UV (Recommended Package Manager)

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (via pip):**

```bash
pip install uv
```

#### System Dependencies

**macOS Users:**

```bash
# Install system dependencies via Homebrew
brew install ffmpeg

# Optional: For better performance
brew install libomp
```

**Linux Users:**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6 libxrender-dev

# Fedora/RHEL
sudo dnf install ffmpeg
```

### Setup with UV (Recommended)

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/ocr-pipeline.git
   cd ocr-pipeline
   ```

2. **Sync dependencies** (creates virtual environment automatically):

   ```bash
   # Install all dependencies and create .venv
   uv sync

   # Or with development dependencies
   uv sync --extra dev
   ```

3. **Verify installation**:

   ```bash
   # Test imports
   uv run python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
   uv run python -c "from paddleocr import PaddleOCR; print('PaddleOCR: OK')"
   uv run python -c "from ocr_pipeline import OCRPipeline; print('Pipeline: OK')"
   ```

4. **Start the API server**:

   ```bash
   # Using uv run
   uv run ocr-pipeline-api

   # Or directly with uvicorn
   uv run uvicorn ocr_pipeline.api.server:app --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### Alternative: Setup with pip

<details>
<summary>Click to expand legacy pip installation</summary>

1. **Clone and navigate**:

   ```bash
   git clone https://github.com/yourusername/ocr-pipeline.git
   cd ocr-pipeline
   ```

2. **Create virtual environment**:

   ```bash
   python3 -m venv .venv

   # Activate it
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -e .

   # Or with dev dependencies
   pip install -e ".[dev]"
   ```

4. **Verify and run** (same as above, but without `uv run` prefix)

</details>

## API Response Format

The API now returns a `structured_document` field containing the fully typed extraction result.

```json
{
  "document_path": "/tmp/tmpxyz.jpg",
  "document_type": "aadhaar",
  "decision": "accept",
  "confidence": {
    "final_score": 0.92,
    "ocr_confidence_score": 0.95,
    "regex_score": 1.0,
    ...
  },
  "structured_document": {
    "aadhaar_number": {
      "value": "1234 5678 9101",
      "confidence": 0.98
    },
    "name": {
      "value": "Jane Doe",
      "confidence": 0.95
    },
    "date_of_birth": {
      "value": "01/01/1980",
      "confidence": 0.99
    },
    "gender": {
      "value": "Female",
      "confidence": 0.99
    },
    "address": {
      "value": "123 Main St, Delhi",
      "confidence": 0.85
    },
    "overall_confidence": 0.92,
    "decision": "accept"
  },
  "processing_time": 2.45
}
```

## Troubleshooting

### macOS-Specific Issues

1. **OpenCV installation fails or imports fail**

   ```bash
   # Install ffmpeg first
   brew install ffmpeg

   # Reinstall opencv-python
   pip uninstall opencv-python opencv-python-headless
   pip install opencv-python>=4.8.0
   ```

2. **M1/M2 Mac (Apple Silicon) issues**

   ```bash
   # Use Python 3.9-3.11 for best compatibility
   # Avoid Python 3.13+ if experiencing issues

   # Install PaddlePaddle for Apple Silicon
   pip install paddlepaddle==2.6.2

   # If still having issues, try:
   arch -arm64 pip install opencv-python
   ```

3. **PaddlePaddle errors on Mac**

   ```bash
   # If you see "Symbol not found" errors
   pip uninstall paddlepaddle
   pip install paddlepaddle==2.6.2 --no-cache-dir
   ```

4. **Permission errors during installation**
   ```bash
   # Don't use sudo with pip, use virtual environment instead
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

### General Issues

1. **ModuleNotFoundError: No module named 'imghdr'** (Python 3.13+)
   - This is automatically handled by the compatibility shim in the project root.
   - Ensure you're importing `ocr_pipeline` before any paddleocr imports.
   - Recommended: Use Python 3.9-3.11 for best compatibility.

2. **Import errors after installation**
   - Make sure you're using the correct import paths: `from ocr_pipeline import OCRPipeline`
   - Verify installation: `pip list | grep ocr-pipeline`

3. **Configuration file not found**
   - Ensure `config.yaml` is in the project root directory.
   - If running from a different directory, set the config path explicitly.

4. **API server won't start**

   ```bash
   # Check if port 8000 is already in use
   lsof -i :8000  # macOS/Linux

   # Use a different port
   uvicorn ocr_pipeline.api.server:app --port 8080
   ```

5. **Low confidence scores or poor extraction**
   - Ensure images are at least 300 DPI
   - Check image quality (not blurry, good lighting)
   - Try preprocessing images before sending to API
   - Review `config.yaml` thresholds if needed

## License

MIT License

## Contributing

Contributions are welcome! Please ensure:

- Code follows PEP 8 style guidelines
- Tests are added for new features
- Documentation is updated accordingly
