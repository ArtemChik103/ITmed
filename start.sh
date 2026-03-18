#!/bin/sh
set -eu

role="${APP_ROLE:-api}"

echo "=== ITmed start.sh ==="
echo "  Role:    $role"
echo "  Python:  $(python3 --version 2>&1 || echo 'not found')"
echo "======================"

case "$role" in
  api)
    echo "Starting API on port ${API_PORT:-8000}..."
    exec uvicorn api.main:app --host 0.0.0.0 --port "${API_PORT:-8000}"
    ;;
  frontend)
    echo "Starting Frontend on port ${FRONTEND_PORT:-8501}..."
    echo "  API_URL: ${API_URL:-not set}"
    exec streamlit run frontend/app.py --server.port="${FRONTEND_PORT:-8501}" --server.address=0.0.0.0
    ;;
  *)
    echo "Unknown APP_ROLE: $role" >&2
    exit 1
    ;;
esac
