# Docker

Targets Jetson AGX Orin with JetPack 6.2.1 (L4T 36.4.4). Requires Docker Compose
and the NVIDIA Container Toolkit configured on the host.
Download the model checkpoint and set `model.path` in
the repository's `config.json` before starting (see the [demo guide](../demos/README.md)).
Paths must be accessible inside `/workspace`.

From the repository root:

```sh
docker compose -f docker/compose.yaml up --build
```

Or from this directory:

```sh
docker compose up --build
```

The repository is bind-mounted at `/workspace`. The Compose file uses
`..:/workspace`, equivalent to `.:/workspace` from the repository root.
The container is named `duplex`. The image build installs dependencies into
`/opt/venus/runtime/venv`, outside the source mount. Startup only launches the
demo; logs and service state persist in the runtime volume. Rebuild the image
after changing dependencies. Follow the login instructions in the startup
output if prompted.

The base is `nvcr.io/nvidia/pytorch:25.02-py3-igpu`, listed for JetPack 6.2 in
[NVIDIA's compatibility matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html).
The venv inherits NVIDIA's CUDA-enabled PyTorch 2.7 prerelease. TorchAudio 2.7
and TorchVision 0.22 are compiled against it for Orin (SM 8.7), using two build
jobs. The build substitutes `decord2==3.4.0`, which supplies the `decord` module
and ARM64 wheels, for the unavailable `decord==0.6.0` wheel. These overrides
apply only to Docker; the normal installer retains its original pins.

Build-time import checks do not require a GPU. Validate GPU access on the Jetson
after building:

```sh
docker compose -f docker/compose.yaml run --rm venus python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'
```

This Jetson build still needs validation on the target hardware, including
model inference with the newer PyTorch version.

Open http://localhost:8032 once startup completes.

To stop from the repository root:

```sh
docker compose -f docker/compose.yaml down
```
