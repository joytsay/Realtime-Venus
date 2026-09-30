#!/usr/bin/env bash
set -Eeuo pipefail
cd /opt/venus

# Inherit NVIDIA's CUDA-enabled torch instead of installing a PyPI ARM CPU wheel.
python3 -m venv --system-site-packages runtime/venv
python_bin="$PWD/runtime/venv/bin/python"
"$python_bin" -m pip install --disable-pip-version-check uv==0.12.15
uv="$PWD/runtime/venv/bin/uv"
"$uv" pip install --python "$python_bin" 'setuptools>=68,<81' wheel 'numpy>=1.26.4,<2' packaging

# Build companion libraries against the exact NVIDIA torch ABI without allowing
# their upstream torch requirements to replace the NVIDIA installation.
export TORCH_CUDA_ARCH_LIST=8.7
export MAX_JOBS=2
git clone --depth 1 --branch v2.7.0 --recurse-submodules https://github.com/pytorch/audio.git /tmp/venus-audio
BUILD_VERSION=2.7.0 BUILD_SOX=0 USE_FFMPEG=0 USE_CUDA=1 \
    "$uv" pip install --python "$python_bin" --no-deps --no-build-isolation /tmp/venus-audio
git clone --depth 1 --branch v0.22.0 https://github.com/pytorch/vision.git /tmp/venus-vision
BUILD_VERSION=0.22.0 FORCE_CUDA=1 \
    "$uv" pip install --python "$python_bin" --no-deps --no-build-isolation /tmp/venus-vision

rm -rf /tmp/venus-audio /tmp/venus-vision
