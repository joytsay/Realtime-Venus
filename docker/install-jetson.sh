#!/usr/bin/env bash
set -Eeuo pipefail
cd /opt/venus
python_bin="$PWD/runtime/venv/bin/python"
uv="$PWD/runtime/venv/bin/uv"

# Override only the Docker environment. Decord2 supplies the decord module and
# has Linux ARM64 wheels; disable the unavailable legacy distribution.
"$python_bin" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import json
import torch

assert torch.version.cuda, 'The base image must supply CUDA-enabled PyTorch'
assert torch.__version__.startswith('2.7.'), torch.__version__
# These distributions are already supplied by the base image/build layers.
# uv must not resolve them against PyPI or replace the compiled libraries.
names = ('torch', 'torchaudio', 'torchvision')
Path('/tmp/venus-torch-versions.json').write_text(json.dumps({name: version(name) for name in names}))
overrides = [f'{name}; python_version < "0"' for name in names]
overrides.append('decord; python_version < "0"')
Path('/tmp/venus-overrides.txt').write_text('\n'.join(overrides) + '\n')
PY
"$uv" pip install --python "$python_bin" --override /tmp/venus-overrides.txt \
    -r demos/requirements.txt decord2==3.4.0
"$uv" pip install --python "$python_bin" --no-deps --no-build-isolation --editable .

"$python_bin" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import json
from demos.install import install_codex
for name, expected in json.loads(Path('/tmp/venus-torch-versions.json').read_text()).items():
    assert version(name) == expected, f'{name} was replaced during dependency installation'
install_codex()
import torch, torchaudio, torchvision, stepaudio2
from decord import VideoReader, cpu
import demos.server.app, demos.model.server
assert torch.version.cuda, 'CUDA-enabled PyTorch was replaced during installation'
print('Jetson build imports OK:', torch.__version__, 'CUDA:', torch.version.cuda)
PY
rm -f /tmp/venus-overrides.txt /tmp/venus-torch-versions.json
"$uv" cache clean
