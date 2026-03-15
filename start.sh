#!/bin/sh
set -eu

role="${APP_ROLE:-api}"

case "$role" in
  api)
    exec uvicorn api.main:app --host 0.0.0.0 --port "${API_PORT:-8000}"
    ;;
  frontend)
    exec streamlit run frontend/app.py --server.port="${FRONTEND_PORT:-8501}" --server.address=0.0.0.0
    ;;
  *)
    echo "Unknown APP_ROLE: $role" >&2
    exit 1
    ;;
esac
