"""Gunicorn configuration for Empire AI."""

import multiprocessing
import os

# Server
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Workers — default to (2 * CPU cores) + 1, capped at 4 for Railway's memory.
# Override with WEB_CONCURRENCY env var on Railway if needed.
_default_workers = min(2 * multiprocessing.cpu_count() + 1, 4)
workers = int(os.environ.get("WEB_CONCURRENCY", _default_workers))
threads = 4  # 4 threads per worker = workers*4 concurrent requests
timeout = 300  # 5 min — long enough for directives, short enough to catch hangs
worker_class = "gthread"  # Threaded workers — better for I/O-bound LLM calls

# Logging
accesslog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")

# Graceful restart
graceful_timeout = 30

# Preload app for faster worker startup and shared memory
preload_app = True

# Worker lifecycle — recycle workers after N requests to prevent memory leaks
max_requests = 1000
max_requests_jitter = 100  # Stagger restarts so they don't all recycle at once
