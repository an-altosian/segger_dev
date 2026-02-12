# Base image with CUDA 12.1 runtime and cuDNN
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install essential system tools
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    unzip \
    htop \
    vim \
    build-essential \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install global python tools
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir debugpy jupyterlab

# Set working directory
WORKDIR /workspace

# Install PyTorch with CUDA support first (from PyTorch index)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install pytorch-lightning and torchmetrics
# Note: 'lightning' is the new package name (was 'pytorch-lightning')
RUN pip install --no-cache-dir lightning torchmetrics

# Install cupy for CUDA operations
RUN pip install --no-cache-dir cupy-cuda12x

# Install dask distributed and geopandas for parallel processing
RUN pip install --no-cache-dir "dask[distributed]" dask-geopandas faiss-cpu

# Clone segger_dev from fork (includes boolean dtype fix)
COPY . /workspace/segger_dev

# Fix 1: Remove typo in train_model.py (line 62 has "uv a" which is invalid Python)
RUN sed -i '/^uv a$/d' /workspace/segger_dev/src/segger/cli/train_model.py || true

# Install segger
RUN pip install -e "/workspace/segger_dev"

# Pin squidpy to compatible version with anndata (install AFTER segger to override)
# Also pin click to avoid CLI utils issues
RUN pip install --no-cache-dir --force-reinstall squidpy==1.8.1 click==8.1.7

# Set environment variables
ENV PYTHONPATH=/workspace/segger_dev/src

# expose ports for debugpy and jupyterlab
EXPOSE 5678 8888
