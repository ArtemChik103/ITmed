FROM python:3.11.8-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    fonts-dejavu-core \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

RUN chmod +x start.sh

RUN useradd -m -s /bin/sh appuser
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/health || curl -f http://localhost:${FRONTEND_PORT:-8501}/_stcore/health || exit 1

CMD ["sh", "start.sh"]
