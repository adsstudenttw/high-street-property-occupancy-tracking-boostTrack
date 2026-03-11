FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    curl \
    build-essential \
    ninja-build \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -f /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

COPY boost-track-env.yml /tmp/boost-track-env.yml

RUN sed '/^prefix:/d' /tmp/boost-track-env.yml > /tmp/boost-track-env.clean.yml && \
    conda env create -n boostTrack -f /tmp/boost-track-env.clean.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/boostTrack/bin:/opt/conda/bin:${PATH}" \
    CONDA_DEFAULT_ENV=boostTrack \
    PYTHONPATH="/workspace:/workspace/external"

WORKDIR /workspace
COPY . /workspace

RUN mkdir -p /workspace/results /workspace/cache

CMD ["bash"]
