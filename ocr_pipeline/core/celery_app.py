import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "ocr_pipeline",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["ocr_pipeline.core.tasks"]
)   

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600, 
    result_expires=86400,       

    task_default_queue="ocr",
    task_queues={
        "ocr": {
            "exchange": "ocr",
            "routing_key": "ocr",
        },
        "llm": {
            "exchange": "llm",
            "routing_key": "llm",
        },
        "ocr_dlq": {
            "exchange": "ocr_dlq",
            "routing_key": "ocr_dlq",
        },
    },
    
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_publish_retry=True,
)

if __name__ == "__main__":
    app.start()
