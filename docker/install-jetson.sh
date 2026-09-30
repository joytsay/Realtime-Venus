#!/usr/bin/env bash
set -Eeuo pipefail
cd /opt/venus
python_bin="$PWD/runtime/venv/bin/python"
uv="$PWD/runtime/venv/bin/uv"

"$uv" pip install --python "$python_bin" --no-deps --no-build-isolation --editable .

# The checkpoint's Resampler is an ordinary nn.Module with an _init_weights
# helper. Transformers 4.52.4 mistakes it for a pretrained submodel and calls
# a nonexistent _initialize_weights method. Check the method actually used.
"$python_bin" - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

assert version('transformers') == '4.52.4', 'Recheck the composite-model compatibility patch'
source = Path(find_spec('transformers').origin).with_name('modeling_utils.py')
text = source.read_text()
original = 'if hasattr(module, "_init_weights"):\n                        module.smart_apply(module._initialize_weights)'
guarded = 'if hasattr(module, "_initialize_weights"):\n                        module.smart_apply(module._initialize_weights)'
if guarded not in text:
    assert text.count(original) == 1, 'Unexpected Transformers source; recheck the composite-model patch'
    source.write_text(text.replace(original, guarded))
PY
