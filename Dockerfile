FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install uv

WORKDIR /workspace

COPY requirements-uv.txt /tmp/requirements-uv.txt

RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 && \
    uv pip install --python /opt/venv/bin/python -r /tmp/requirements-uv.txt

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/workspace:/workspace/external"

COPY . /workspace

RUN mkdir -p /workspace/results /workspace/cache

CMD ["bash"]
