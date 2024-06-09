#!/bin/bash

HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-5306}
WORKERS=${WORKERS:-4}
UVICORN_WORKER=${UVICORN_WORKER:-"uvicorn.workers.UvicornWorker"}
LOGLEVEL=${LOGLEVEL:-"debug"}

python3 -m uvicorn "src.app:app" \
    --host=${HOST} \
    --port=${PORT}