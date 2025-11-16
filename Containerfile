FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir feedparser trafilatura python-dateutil openai

COPY opml_digest.py /app/opml_digest.py

CMD ["python", "/app/opml_digest.py"]

