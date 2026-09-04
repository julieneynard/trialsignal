FROM python:3.11-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN uv pip install --system --no-cache .

EXPOSE 8000
CMD ["uvicorn", "trialsignal.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
