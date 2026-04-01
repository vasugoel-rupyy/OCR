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

RUN mkdir -p /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer \
    && mkdir -p /root/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer \
    && mkdir -p /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer \
    && apt-get update && apt-get install -y curl tar && \
    curl -L https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar -o /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/en_PP-OCRv3_det_infer.tar && \
    tar -xf /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/en_PP-OCRv3_det_infer.tar -C /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/ && \
    curl -L https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar -o /root/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer/en_PP-OCRv3_rec_infer.tar && \
    tar -xf /root/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer/en_PP-OCRv3_rec_infer.tar -C /root/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer/ && \
    curl -L https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar -o /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.tar && \
    tar -xf /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.tar -C /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ && \
    rm /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/*.tar \
    && rm /root/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer/*.tar \
    && rm /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/*.tar

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

RUN mkdir -p /app/shared_temp /home/ocruser/.paddleocr
COPY --from=builder /root/.paddleocr /home/ocruser/.paddleocr

RUN chmod +x ./scripts/apply-patches.sh
RUN ./scripts/apply-patches.sh

RUN chown -R ocruser:ocruser /app /app/shared_temp /home/ocruser/.paddleocr

USER ocruser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/docs', timeout=5)" || exit 1

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python3", "-m", "ocr_pipeline.api.server"]
