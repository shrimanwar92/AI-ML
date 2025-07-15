# Stage 1: Base image already has TFDV, TFT, and Beam
FROM tfdv-tft:latest as base

WORKDIR /app

# Stage 2: Install SageMaker dependencies on top
FROM base as sagemaker-layer

# Set environment and install packages safely
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

# Install compatible dill first
RUN pip install dill==0.3.1.1

# Now install SageMaker and friends without triggering dill conflict
RUN pip install dill==0.3.1.1 && \
    pip install sagemaker==2.219.0

# Optional: Copy your scripts or set default command
COPY . /app

CMD ["python3"]
