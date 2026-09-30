# Jetson iGPU image listed by NVIDIA for JetPack 6.2.
FROM nvcr.io/nvidia/pytorch:25.02-py3-igpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake ffmpeg git libsndfile1 \
        ninja-build python3-venv libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/venus

# Cache expensive native builds independently of application dependencies.
COPY docker/build-torch-libraries.sh ./docker/build-torch-libraries.sh
RUN bash docker/build-torch-libraries.sh

# Keep downloads across retries, and installed dependencies across source edits.
COPY demos/requirements.txt ./demos/requirements.txt
COPY harness/requirements.txt ./harness/requirements.txt
COPY docker/install-jetson-deps.sh ./docker/install-jetson-deps.sh
RUN --mount=type=cache,id=venus-jetson-uv,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy bash docker/install-jetson-deps.sh

COPY pyproject.toml README.md LICENSE install.sh ./
COPY demos/ ./demos/
COPY harness/ ./harness/
COPY docker/install-jetson.sh ./docker/install-jetson.sh

RUN --mount=type=cache,id=venus-jetson-uv,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy bash docker/install-jetson.sh \
    && rm -rf runtime/cache

# Cache installation even if verification fails or its checks change.
COPY docker/verify-jetson.sh ./docker/verify-jetson.sh
RUN bash docker/verify-jetson.sh

# Keep the built environment outside the source bind mount and runtime volume.
ENV VENUS_PYTHON=/opt/venus/runtime/venv/bin/python \
    PATH="/opt/venus/runtime/venv/bin:/opt/venus/runtime/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace
# Bypass the base image's entrypoint; the launcher performs GPU preflight.
ENTRYPOINT []
CMD ["bash", "start.sh", "--host", "0.0.0.0", "--web-port", "8032"]
