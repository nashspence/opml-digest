FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir requests openai trafilatura

COPY freshrss_digest.py /app/freshrss_digest.py

CMD ["python", "/app/freshrss_digest.py"]
