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
    -r demos/requirements.txt decord2==3.4.0 transformers==4.52.4

# Transformers 4.52.4 imports DTensor even when torch was built without
# distributed support (NVIDIA's Jetson build). Guard that optional import and
# skip tensor-parallel plan validation when its style registry is unavailable.
# Keep this patch scoped to the Docker environment and the version above.
"$python_bin" - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import torch

if not torch.distributed.is_available():
    assert version('transformers') == '4.52.4', 'Recheck the Jetson compatibility patch'
    source = Path(find_spec('transformers').origin).with_name('modeling_utils.py')
    text = source.read_text()
    replacements = (
        (
            'import torch.distributed.tensor\n',
            'if torch.distributed.is_available():\n    import torch.distributed.tensor\n',
        ),
        (
            'if self._tp_plan is not None and is_torch_greater_or_equal("2.3"):',
            'if self._tp_plan is not None and ALL_PARALLEL_STYLES is not None and is_torch_greater_or_equal("2.3"):',
        ),
    )
    for original, guarded in replacements:
        if guarded not in text:
            assert text.count(original) == 1, 'Unexpected Transformers source; recheck the Jetson patch'
            text = text.replace(original, guarded)
    source.write_text(text)
PY

"$python_bin" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import json
for name, expected in json.loads(Path('/tmp/venus-torch-versions.json').read_text()).items():
    assert version(name) == expected, f'{name} was replaced during dependency installation'
PY
rm -f /tmp/venus-overrides.txt /tmp/venus-torch-versions.json
