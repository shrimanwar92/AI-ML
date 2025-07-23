FROM python:3.11-slim as base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages for pip and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment and install packages safely
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir pandas

COPY . /app

CMD ["python3"]