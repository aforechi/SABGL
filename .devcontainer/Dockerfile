# Use Ubuntu 20.04 with CUDA pre-installed (development version for compiling code)
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 2.7 and development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python2 \
    python2-dev \
    python-is-python2 \
    build-essential \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gfortran \
    libatlas-base-dev \
    cmake \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 2.7 (deprecated, so we fetch an archived installer)
RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py \
    && python2 get-pip.py \
    && rm get-pip.py

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
