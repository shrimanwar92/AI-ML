#!/bin/bash

# Create merged site-packages directory
mkdir -p docker/site-packages

# First venv: SageMaker-related packages
VENV_SM="/mnt/c/Users/shrim/Documents/src/AI-ML/feature-engineering/uci-census-data-sagemaker-pipelines/venv_sm/lib/python3.11/site-packages"

# Second venv: TFDV, TFT, TF-related packages
VENV_TFX="/home/nilays/tfx/lib/python3.11/site-packages"

# Link everything from first venv
for pkg in "$VENV_SM"/*; do
    ln -s "$pkg" docker/site-packages/ 2>/dev/null
done

# Link everything from second venv (skip if already exists)
for pkg in "$VENV_TFX"/*; do
    base=$(basename "$pkg")
    if [ ! -e docker/site-packages/"$base" ]; then
        ln -s "$pkg" docker/site-packages/ 2>/dev/null
    fi
done

echo "✅ Symlinks created for both venvs."
