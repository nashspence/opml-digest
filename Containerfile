FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir requests openai trafilatura

COPY greader-digest.py /app/greader-digest.py

CMD ["python", "/app/greader-digest.py"]
