FROM python:3.11-bullseye as base

WORKDIR /app

# Non-interactive apt
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ONNX + ONNX Runtime + dependencies
RUN pip install --no-cache-dir onnx onnxruntime numpy pillow requests

RUN pip install --no-cache-dir onnxconverter_common


# Optional: remove build tools to reduce final image size
RUN apt-get remove -y build-essential && apt-get autoremove -y && apt-get clean

# Copy your code
COPY . /app

# Default command
CMD ["python3"]