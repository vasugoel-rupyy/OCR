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

### Quick Start (3 Steps)

1. **Install UV package manager**

   **Linux/macOS:**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   **Windows (PowerShell):**

   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone and install**

   ```bash
   git clone https://github.com/vasugoel-rupyy/OCR.git
   cd OCR
   uv sync
   # 'uv sync' automatically creates and manages a virtual environment (.venv)
   ```

3. **Run the API server**

   ```bash
   uv run ocr-pipeline-api
   ```


Visit `http://localhost:8000/docs` to access the API documentation.

---

## 🐳 Docker Deployment (Recommended)

**Docker is the recommended deployment method** to avoid platform-specific issues (like macOS segfaults) and ensure consistency across environments.

### Quick Start with Docker

```bash
# 1. Clone the repository
git clone https://github.com/vasugoel-rupyy/OCR.git
cd OCR

# 2. Build and start the service
docker compose up --build

# 3. Access the API at http://localhost:8000/docs
```

That's it! The service will be running in a Linux container with all dependencies pre-configured.

### Docker Commands

**Build the image:**
```bash
docker build -t ocr-pipeline:latest .
```

**Run the container:**
```bash
docker run -d \
  --name ocr-api \
  -p 8000:8000 \
  ocr-pipeline:latest
```

**View logs:**
```bash
docker logs -f ocr-api
```

**Stop the container:**
```bash
docker stop ocr-api
docker rm ocr-api
```

### Development Workflow with Docker

The `docker-compose.yml` is configured for hot-reload during development:

```bash
# Start in development mode (with volume mounts for live code changes)
docker compose up

# Your code changes in ocr_pipeline/ will be reflected immediately
# No need to rebuild for code changes, only for dependency changes
```

**To rebuild after changing dependencies:**
```bash
docker compose up --build
```

### Production Deployment

For production, build the optimized image and deploy to your preferred platform:

```bash
# Build production image
docker build -t ocr-pipeline:prod .

# Tag for your registry (e.g., Docker Hub, ECR, GCR)
docker tag ocr-pipeline:prod your-registry/ocr-pipeline:v1.2.0

# Push to registry
docker push your-registry/ocr-pipeline:v1.2.0
```

**Deployment platforms:**
- **AWS**: ECS, Fargate, or EC2 with Docker
- **Google Cloud**: Cloud Run or GKE
- **Azure**: Container Instances or AKS
- **Fly.io**: `fly launch` (Docker detected automatically)
- **Railway**: Connect GitHub repo and deploy

### Docker Troubleshooting

<details>
<summary><b>Container fails to start or crashes immediately</b></summary>

```bash
# Check logs for errors
docker logs ocr-api

# Run in interactive mode to see full output
docker run -it --rm -p 8000:8000 ocr-pipeline:latest
```

</details>

<details>
<summary><b>Port 8000 already in use</b></summary>

```bash
# Use a different port mapping
docker run -d -p 8080:8000 ocr-pipeline:latest

# Or with docker-compose, edit docker-compose.yml:
# ports:
#   - "8080:8000"
```

</details>

<details>
<summary><b>Out of memory errors</b></summary>

The OCR pipeline can be memory-intensive. Increase Docker's memory limit:

- Docker Desktop: Settings → Resources → Memory (set to at least 4GB)
- Command line: `docker run -m 4g ocr-pipeline:latest`

</details>

---


### System Dependencies (Optional but Recommended)

<details>
<summary><b>Click to expand system dependencies</b></summary>

Most modern systems have these already. Install only if you encounter issues:

**Ubuntu/Debian:**

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libsm6 libxext6
```

\*\*macOS (Homebrew):

```bash
brew install ffmpeg
```

**Fedora/RHEL:**

```bash
sudo dnf install ffmpeg
```

</details>

---

### Troubleshooting

<details>
<summary><b>Installation fails with "command not found: uv"</b></summary>

After installing UV, restart your terminal or run:

```bash
source $HOME/.cargo/env  # Linux/macOS
```

</details>

<details>
<summary><b>Error: "No module named 'setuptools'"</b></summary>

This has been fixed in the latest version. Make sure you have the latest `pyproject.toml`:

```bash
git pull origin main
uv sync --refresh
```

</details>

<details>
<summary><b>PaddleOCR import fails or crashes</b></summary>

This is usually due to OpenCV version conflicts. The project pins `opencv-python<=4.6.0.66` for compatibility:

```bash
# Verify correct version is installed
uv run python -c "import cv2; print(cv2.__version__)"
# Should show 4.6.0.66 or lower
```

</details>

<details>
<summary><b>API server won't start</b></summary>

Make sure port 8000 is not already in use:

```bash
# Check if port is in use
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Use a different port
uv run uvicorn ocr_pipeline.api.server:app --port 8080
```

</details>

---

### Alternative: Manual Installation with pip

<details>
<summary><b>Click to expand pip installation (not recommended)</b></summary>

1. **Clone and navigate**:

   ```bash
   git clone https://github.com/yourusername/ocr-pipeline.git
   cd ocr-pipeline
   ```

2. **Create virtual environment**:

   ```bash
   python3 -m venv .venv

   # Activate it:
   source .venv/bin/activate      # Linux/macOS
   .venv\Scripts\activate.bat     # Windows (Cmd)
   .venv\Scripts\Activate.ps1     # Windows (PowerShell)
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
