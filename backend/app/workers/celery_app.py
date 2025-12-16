"""
Sentinel Quant - Celery Application Configuration
"""
from celery import Celery
from celery.schedules import crontab

from core.config import settings

# Create Celery app
celery_app = Celery(
    "sentinel_quant",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "workers.tasks.sentiment",
        "workers.tasks.monitoring",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Timezone
    timezone="UTC",
    enable_utc=True,
    
    # Task settings
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max
    task_soft_time_limit=240,  # Soft limit at 4 minutes
    
    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time
    worker_concurrency=2,
    
    # Result settings
    result_expires=3600,  # 1 hour
)

# Celery Beat Schedule (Periodic Tasks)
celery_app.conf.beat_schedule = {
    # Update sentiment every 15 minutes
    "update-sentiment-every-15-min": {
        "task": "workers.tasks.sentiment.update_market_sentiment",
        "schedule": crontab(minute="*/15"),
        "args": ()
    },
    
    # Health check every 5 minutes
    "health-check-every-5-min": {
        "task": "workers.tasks.monitoring.system_health_check",
        "schedule": crontab(minute="*/5"),
        "args": ()
    },
    
    # Update position prices every minute
    "update-positions-every-minute": {
        "task": "workers.tasks.monitoring.update_position_prices",
        "schedule": crontab(minute="*"),
        "args": ()
    },
}
