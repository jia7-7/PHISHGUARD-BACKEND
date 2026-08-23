FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY rule_engine ./rule_engine
COPY ai_service ./ai_service
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 phishguard \
    && chown -R phishguard:phishguard /app

USER phishguard
EXPOSE 10000

CMD ["python", "scripts/cloud_start.py"]
