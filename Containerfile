FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir feedparser trafilatura python-dateutil openai pyyaml jsonschema

COPY opml-digest.py /app/opml-digest.py
COPY config-schema.yaml /app/config-schema.yaml
COPY config-example.yaml /app/config-example.yaml

CMD ["python", "/app/opml-digest.py"]
