import os
from celery import Celery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery
app = Celery(
    "ocr_pipeline",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["ocr_pipeline.core.tasks"]
)

# Configuration
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    result_expires=86400,  # 24 hours result persistence
    
    # Queue settings
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
    },
)

if __name__ == "__main__":
    app.start()
