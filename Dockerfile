FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md config.yaml imghdr.py ./
COPY ocr_pipeline/ ./ocr_pipeline/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    patch \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 ocruser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY ocr_pipeline/ ./ocr_pipeline/
COPY config.yaml ./
COPY imghdr.py ./
COPY patches/ ./patches/
COPY scripts/ ./scripts/

RUN chmod +x ./scripts/apply-patches.sh
RUN ./scripts/apply-patches.sh

RUN chown -R ocruser:ocruser /app

USER ocruser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/docs', timeout=5)" || exit 1

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python3", "-m", "ocr_pipeline.api.server"]
