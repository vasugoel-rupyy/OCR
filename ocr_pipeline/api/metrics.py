from prometheus_client import Counter, Histogram, REGISTRY



REQUEST_COUNT = Counter(
    "ocr_request_total", 
    "Total OCR requests", 
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "ocr_request_latency_seconds", 
    "Request latency", 
    ["endpoint"]
)

TASK_ENQUEUED = Counter(
    "ocr_task_enqueued_total", 
    "Total tasks enqueued", 
    ["queue"]
)
