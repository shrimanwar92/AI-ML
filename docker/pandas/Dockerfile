FROM python:3.11-slim as base

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Cache dependencies in build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
CMD ["python3"]