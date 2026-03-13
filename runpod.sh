#!/bin/bash
set -e  # Exit on error

cd /workspace
export DEBIAN_FRONTEND=noninteractive

# Update REAM if exists, clone if not
if [ -d "REAM" ]; then
    echo "Updating existing REAM repository..."
    cd REAM
    git pull
else
    echo "Cloning REAM repository..."
    git clone https://github.com/JackBinary/REAM.git
    cd REAM
fi

# Create/update virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv .venv
fi

echo "Installing/updating dependencies..."

# Always upgrade pip/setuptools/wheel
.venv/bin/python -m pip install --upgrade pip setuptools wheel uv

# Install REAM package (editable)
VLLM_USE_PRECOMPILED=1 .venv/bin/python -m uv pip install --editable . --torch-backend auto

# Install other dependencies
.venv/bin/python -m uv pip install "transformers==5.3.0" huggingface_hub
.venv/bin/python -m uv pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==26.2.*" "dask-cudf-cu12==26.2.*" "cuml-cu12==26.2.*" \
    "cugraph-cu12==26.2.*" "nx-cugraph-cu12==26.2.*" "cuxfilter-cu12==26.2.*" \
    "cucim-cu12==26.2.*" "pylibraft-cu12==26.2.*" "raft-dask-cu12==26.2.*" \
    "cuvs-cu12==26.2.*" "nx-cugraph-cu12==26.2.*"

# Run tokenizer patch
echo "Running tokenizer patch..."
.venv/bin/python patch_tokenizer.py llmfan46/Qwen3.5-35B-A3B-heretic-v2

# Run REAM
echo "Starting REAM..."
exec .venv/bin/python -m ream.ream \
    --model_name llmfan46/Qwen3.5-35B-A3B-heretic-v2 \
    --compression_ratio .3125 \
    --dataset_name ream_mixed
