# Docker

Targets Jetson AGX Orin with JetPack 6.2.1 (L4T 36.4.4). Requires Docker Compose
and the NVIDIA Container Toolkit configured on the host.
Download the model checkpoint and set `model.path` in
the repository's `config.json` before starting (see the [demo guide](../demos/README.md)).
Paths must be accessible inside `/workspace`.

For the default Omni configuration, download the complete checkpoint using the
built image. From `docker/`:

```sh
docker compose run --rm venus python download_models.py --model omni --local-dir /workspace
docker compose up
```

The download is saved in the repository's `Realtime-Venus-Omni/` directory
through the bind mount and survives container recreation. If the checkpoint
already exists elsewhere, mount its directory and set `model.path` in
`config.json` to its path inside the container.

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
demo; model and web service output streams directly to Docker logs, and service
state persists in the runtime volume. View startup errors with
`docker compose logs -f venus` from this directory. Rebuild the image
after changing dependencies. Follow the login instructions in the startup
output if prompted.

For a session with microphone activity but no reply, model logs include audio
RMS, prefill/generation timings, and listen/speak decisions. Web logs report
whether output audio is queued or suppressed. A `started` message without its
matching `completed` message identifies an inference call still in progress.
The model's internal `/healthz` endpoint also reports the active inference
stage and elapsed time:

```sh
docker compose exec venus python -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8031/healthz").read().decode())'
```

Python dependencies are installed before application sources are copied, so
source edits reuse the dependency layer. A BuildKit cache retains uv downloads
across dependency changes and failed builds without adding them to the image.
The first build with this cache downloads packages once; subsequent builds on
the same builder reuse them unless the build cache has been pruned.
Application installation and model verification use separate `RUN` layers, so
retrying a failed check or editing its script reuses the completed install layer.
The Docker build does not install the Codex CLI; the configured task backend
uses the external llama.cpp server. The editable package install is a local
operation without network downloads.

The base is `nvcr.io/nvidia/pytorch:25.02-py3-igpu`, listed for JetPack 6.2 in
[NVIDIA's compatibility matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html).
The venv inherits NVIDIA's CUDA-enabled PyTorch 2.7 prerelease. TorchAudio 2.7
and TorchVision 0.22 are compiled against it for Orin (SM 8.7), using two build
jobs. The build substitutes `decord2==3.4.0`, which supplies the `decord` module
and ARM64 wheels, for the unavailable `decord==0.6.0` wheel. These overrides
apply only to Docker; the normal installer retains its original pins.

The image pins Transformers to 4.52.4 and guards its unconditional DTensor
import when PyTorch lacks distributed support. This keeps single-GPU model
loading usable with NVIDIA's Jetson build. It also skips tensor-parallel plan
validation when the parallel-style registry is unavailable, avoiding the
`NoneType` error during Qwen3 initialization. The build checks the
`PreTrainedModel` and application imports, then constructs a tiny Qwen3 model
and runs a CPU forward pass without downloading a checkpoint.
The application install layer also guards Transformers' composite initializer
against ordinary modules with an `_init_weights` helper, such as Omni's
Resampler. Verification loads a tiny composite checkpoint with a missing
parameter to check both initialization and preservation of loaded weights.

Build-time import checks do not require a GPU. Validate GPU access on the Jetson
after building:

```sh
docker compose -f docker/compose.yaml run --rm venus python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'
```

This Jetson build still needs validation on the target hardware, including
model inference with the newer PyTorch version.

Open https://192.168.5.151:8033 once startup completes. Caddy terminates HTTPS
and proxies HTTP and WebSocket traffic to Venus over the Compose network.
The existing http://192.168.5.151:8032 URL redirects to HTTPS, preserving the
path and query string. Proxy logs appear in `docker compose logs -f https`.

Apply the HTTPS setup from `docker/` without rebuilding the Venus image:

```sh
docker compose up -d --force-recreate
docker compose logs -f venus https
```

Caddy issues the IP certificate using a local CA and keeps its keys and
certificates in the `venus-https-data` volume. Export the public CA certificate
on the Jetson after the proxy starts:

```sh
docker compose cp https:/data/caddy/pki/authorities/local/root.crt /tmp/venus-local-ca.crt
```

Copy `/tmp/venus-local-ca.crt` to the computer running the browser and import it
as a trusted certificate authority. For Firefox, use Settings → Privacy &
Security → Certificates → View Certificates → Authorities → Import, and allow
the CA to identify websites. For Chrome or Edge, import it into the client's
trusted root certificate store using the browser's certificate manager.
Restart the browser and open https://192.168.5.151:8033. Trusting the CA is
required for microphone/camera access; bypassing a certificate warning alone
does not provide a reliably trusted origin.

If the Jetson's address changes, copy `.env.example` to `.env` in `docker/`,
set `VENUS_HTTPS_HOST` to the new hostname or IPv4 address (without scheme or
port), and recreate the services. `VENUS_HTTPS_PORT` changes the published
HTTPS port and redirect together. Keep the CA volume to preserve client trust.

To stop from the repository root:

```sh
docker compose -f docker/compose.yaml down
```

## Local llama.cpp backend

The checked-in `harness/config.json` now selects `llamacpp` for the task agent,
routing, spoken summaries, and direct text answers. It connects to the audio.cpp
AGX deployment's published port at `http://host.docker.internal:8082/v1`, with
model `bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M`. Compose supplies the Linux host
mapping; `localhost` inside Duplex would refer to Duplex itself.

Start the existing audio.cpp stack on the same AGX, then recreate Duplex to
apply the host mapping (from this repository's root):

```sh
docker compose -f /home/joy/Git/audio.cpp/examples/docker/agx-one-container/compose.yml up -d
curl -fsS http://127.0.0.1:8082/v1/models
docker compose -f docker/compose.yaml up -d --force-recreate
docker compose -f docker/compose.yaml logs -f venus
```

The model ID must match `/v1/models`. Change the URL/model in Settings → Task
connection or `harness/config.json` if needed. For a launch outside Docker, use
`http://127.0.0.1:8082/v1`; for a separate server, use its reachable IP address.
Select a Task workspace in Settings before starting a conversation.

When all active roles use llama.cpp, neither a Codex executable nor ChatGPT
login is needed. Startup and conversation admission check model availability.
This probe checks the model inventory, not successful inference or GPU capacity.
Selecting Codex for any active role restores its login requirement. Existing
configuration files without these new fields retain their Codex defaults.
To use a Codex role in this image, provide a Codex executable separately and
configure its path; the Docker build does not download it.

The local agent can fetch task context, list/read workspace files, and write
text artifacts. Continuations reuse the task workspace; inputs remain read-only.
It has no shell, web browser, or image/audio understanding. The streaming Venus
model still handles the conversation and still needs its checkpoint and GPU
memory. Visual task requests require a different provider; the local backend
reports that limitation. Audio understanding must come from Venus's textual
request or a capable multimodal provider.

The audio.cpp profile's 4096-token context is suitable for short requests.
Long contexts and multi-step file tasks may require increasing `LLAMA_CTX_SIZE`
in that stack, with additional shared GPU memory use. `llamacpp.max_tokens`
(default 1024) bounds each answer and `llamacpp.max_steps` (default 12) bounds
file-tool actions. Exhausted context/output limits fail explicitly.
