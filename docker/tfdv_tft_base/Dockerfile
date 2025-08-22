FROM python:3.11-slim as base

WORKDIR /app

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages for pip and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary --timeout=120 -r requirements.txt

RUN pip install --no-cache-dir git+https://github.com/onnx/tensorflow-onnx
RUN pip install --no-cache-dir onnxruntime

# Optional: Remove build tools to reduce final image size
RUN apt-get remove -y build-essential && apt-get autoremove -y && apt-get clean

COPY . /app

CMD ["python3"]
