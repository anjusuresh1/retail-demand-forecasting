FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

COPY requirements-api.txt .

RUN pip install --no-cache-dir \
    --upgrade pip \
    && pip install --no-cache-dir \
    -r requirements-api.txt

COPY src ./src
COPY models ./models

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD [ "sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]