FROM python:3.11-slim AS base

# libgomp1: lightgbm's compiled booster links against libgomp (OpenMP) at
# import time, not just at training/inference time. python:3.11-slim doesn't
# ship it, so without this the process crashes on startup the moment
# load_model() unpickles a bundle that imports lightgbm -- "OSError:
# libgomp.so.1: cannot open shared object file" -- found deploying to Render.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

EXPOSE 8000
CMD ["uvicorn", "trialsignal.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
