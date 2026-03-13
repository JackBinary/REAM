#!/bin/bash
cd /workspace
export DEBIAN_FRONTEND=noninteractive

# Install dependencies
apt update && apt install software-properties-common git -y && \
add-apt-repository ppa:deadsnakes/ppa -y && \
apt install python3.12 python3.12-venv python3.12-dev -y

# Clone and setup
git clone https://github.com/JackBinary/REAM.git && cd REAM
python3.12 -m venv .venv

# Use venv
.venv/bin/python -m pip install uv && \
.venv/bin/python -m uv pip install --upgrade pip setuptools wheel && \
VLLM_USE_PRECOMPILED=1 .venv/bin/python -m uv pip install --editable . --torch-backend auto && \
.venv/bin/python -m uv pip install "transformers==5.3.0"
.venv/bin/python -m uv pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==26.2.*" "dask-cudf-cu12==26.2.*" "cuml-cu12==26.2.*" \
    "cugraph-cu12==26.2.*" "nx-cugraph-cu12==26.2.*" "cuxfilter-cu12==26.2.*" \
    "cucim-cu12==26.2.*" "pylibraft-cu12==26.2.*" "raft-dask-cu12==26.2.*" \
    "cuvs-cu12==26.2.*" "nx-cugraph-cu12==26.2.*"

# Run REAM
.venv/bin/python patch_tokenizer.py llmfan46/Qwen3.5-35B-A3B-heretic-v2
.venv/bin/python -m ream.ream \
    --model_name llmfan46/Qwen3.5-35B-A3B-heretic-v2 \
    --compression_ratio .3125 \
    --dataset_name ream_mixed
