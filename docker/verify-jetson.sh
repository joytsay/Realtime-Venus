#!/usr/bin/env bash
set -Eeuo pipefail
cd /opt/venus
python_bin="$PWD/runtime/venv/bin/python"

"$python_bin" - <<'PY'
import torch, torchaudio, torchvision, stepaudio2
from transformers.modeling_utils import PreTrainedModel
from transformers import Qwen3Config, Qwen3ForCausalLM
from decord import VideoReader, cpu
import demos.server.app, demos.model.server
assert torch.version.cuda, 'CUDA-enabled PyTorch was replaced during installation'
# Imports alone miss Qwen3's post_init tensor-parallel validation. Exercise the
# same constructor used by Omni without a checkpoint download or a build GPU.
config = Qwen3Config(
    vocab_size=32, hidden_size=16, intermediate_size=32,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
    head_dim=8, max_position_embeddings=32,
)
config._attn_implementation = 'sdpa'
model = Qwen3ForCausalLM(config).eval()
with torch.no_grad():
    assert model(torch.tensor([[1, 2]])).logits.shape == (1, 2, 32)

# Exercise checkpoint loading with a Resampler-like helper and a genuine
# pretrained child. Include a missing bias to check initialization as well.
from pathlib import Path
from tempfile import TemporaryDirectory
from safetensors.torch import save_file

class Resampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(16, 16)

    def _init_weights(self, module):
        raise AssertionError('Ordinary module helper must not be dispatched as a pretrained initializer')

class CompositeModel(PreTrainedModel):
    config_class = Qwen3Config
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.llm = Qwen3ForCausalLM(config)
        self.resampler = Resampler()

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.constant_(module.weight, 0.25)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.25)

with TemporaryDirectory() as directory:
    fixture = CompositeModel(config)
    config.save_pretrained(directory)
    weights = fixture.state_dict()
    del weights['resampler.projection.bias']
    save_file(weights, str(Path(directory) / 'model.safetensors'), metadata={'format': 'pt'})
    loaded = CompositeModel.from_pretrained(directory, local_files_only=True, attn_implementation='sdpa')
    assert torch.equal(loaded.resampler.projection.bias, torch.full((16,), 0.25))
    assert torch.equal(loaded.llm.model.embed_tokens.weight, fixture.llm.model.embed_tokens.weight)
    with torch.no_grad():
        assert loaded.llm(torch.tensor([[1, 2]])).logits.shape == (1, 2, 32)
print('Jetson build imports OK:', torch.__version__, 'CUDA:', torch.version.cuda)
PY
