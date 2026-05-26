# Machine Vision MultiModal — CUDA 12.1 / Python 3.11 / PyTorch 2.x
# RTX 30xx / 40xx 호환
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# Base utilities + Python 3.11 + libs needed by opencv / open3d / pyrender
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates curl wget git git-lfs build-essential \
        ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
        libxrender1 libxi6 libxrandr2 libxinerama1 libxcursor1 \
        libosmesa6 freeglut3-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils python3-pip \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install PyTorch for CUDA 12.1 first (large layer cached separately)
RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.* torchvision==0.19.* torchaudio==2.4.*

# Project sources
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

RUN pip install -r requirements.txt && pip install -e ".[anomaly,pose,pdm,viz,dev]"

# Default to an interactive shell, but allow `docker run ... mvmm ...`
CMD ["bash"]
