# Multi-stage Dockerfile for OCR Pipeline Service
# Optimized for Linux platform to avoid macOS-specific segfaults

# ============================================================================
# Stage 1: Builder - Install dependencies and pre-download models
# ============================================================================
FROM python:3.11-slim AS builder

# Install system dependencies required for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files needed for package build
COPY pyproject.toml README.md config.yaml imghdr.py ./
COPY ocr_pipeline/ ./ocr_pipeline/

# Install Python dependencies
# Use pip directly since we're in a container (no need for uv/venv)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ============================================================================
# Stage 2: Runtime - Final lightweight image
# ============================================================================
FROM python:3.11-slim

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    patch \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 ocruser

# Set working directory
WORKDIR /app

#  Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY ocr_pipeline/ ./ocr_pipeline/
COPY config.yaml ./
COPY imghdr.py ./

# Copy patches and scripts
COPY patches/ ./patches/
COPY scripts/ ./scripts/

# Make the patch script executable
RUN chmod +x ./scripts/apply-patches.sh

# Apply PaddleOCR patches
RUN ./scripts/apply-patches.sh

# Change ownership to non-root user
RUN chown -R ocruser:ocruser /app

# Switch to non-root user
USER ocruser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/docs', timeout=5)" || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the FastAPI server
CMD ["python3", "-m", "ocr_pipeline.api.server"]
