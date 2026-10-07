"""HTTP API for the Realtime-Venus-Omni duplex model."""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from demos.model.adapter import (
    AdapterError,
    ModelNotLoaded,
    OutputTimeout,
    RealtimeVenusOmniAdapter,
)
from demos.model.prompts import REALTIME_VENUS_SYSTEM_PROMPT
from demos.variants import model_name

_LOG = logging.getLogger("demos.model.server")


@dataclass
class _ServerConfig:
    model_path: str | None = None
    ref_audio_path: str | None = None
    prompt_wav_path: str | None = None
    memory_minutes: int = 40
    model_type: str = "omni"
    model_revision: str = "realtime-venus-omni-v1"
    model_name: str = "Realtime-Venus-Omni"
    system_prompt: str | None = None
    default_timeout_s: float = 30.0
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"


async def _build_adapter(config: _ServerConfig) -> RealtimeVenusOmniAdapter:
    return await RealtimeVenusOmniAdapter.load_real(
        config.model_path,
        memory_minutes=config.memory_minutes,
        model_type=config.model_type,
        ref_audio_path=config.ref_audio_path,
        prompt_wav_path=config.prompt_wav_path or config.ref_audio_path,
        model_revision=config.model_revision,
        model_name=config.model_name,
        system_prompt=config.system_prompt,
    )


def create_app(config: _ServerConfig) -> FastAPI:
    """Construct the FastAPI app; the adapter is loaded in the lifespan."""

    state: dict[str, Any] = {"adapter": None, "ready": False}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        adapter = await _build_adapter(config)
        app.state.adapter = adapter
        state["adapter"] = adapter
        state["ready"] = True
        _LOG.info("adapter ready (revision=%s)", adapter._model_revision)  # noqa: SLF001
        try:
            yield
        finally:
            # Best-effort: close an open session on shutdown.
            sess = getattr(adapter, "_session", None)  # noqa: SLF001
            if sess is not None and not sess.closed:
                try:
                    await adapter.close_session(
                        {
                            "session_id": sess.session_id,
                            "incarnation": sess.incarnation,
                            "reason": "server_shutdown",
                        }
                    )
                except Exception:  # noqa: BLE001
                    _LOG.warning("shutdown close failed", exc_info=True)

    app = FastAPI(
        title="demos.model Realtime-Venus-Omni ServingPort",
        version="0.1.0",
        lifespan=lifespan,
    )

    def adapter() -> RealtimeVenusOmniAdapter:
        a = state["adapter"]
        if a is None:
            raise ModelNotLoaded("adapter is not ready")
        return a

    def _err(exc: Exception) -> JSONResponse:
        code = getattr(exc, "status_code", 500)
        if code == 408 or isinstance(exc, OutputTimeout):
            code = 408
        return JSONResponse(
            status_code=int(code),
            content={"error": str(exc), "type": type(exc).__name__},
        )

    @app.get("/healthz")
    async def health():
        if not state["ready"]:
            return JSONResponse(status_code=503, content={"ready": False})
        a = adapter()
        h = await a.health()
        return {"ready": True, **h}

    @app.post("/sessions")
    async def open_session(req: Request):
        try:
            return await adapter().open_session(await req.json())
        except (AdapterError, ModelNotLoaded) as exc:
            return _err(exc)

    @app.delete("/sessions/{sid}")
    async def close_session(sid: str, incarnation: int, reason: str = "client_close"):
        try:
            return await adapter().close_session(
                {"session_id": sid, "incarnation": incarnation, "reason": reason}
            )
        except (AdapterError, ModelNotLoaded) as exc:
            return _err(exc)

    async def _body_with_session(req: Request, sid: str) -> dict[str, Any]:
        body = await req.json()
        if body.get("session_id") != sid:
            raise AdapterError(f"path session_id {sid!r} does not match body")
        body.setdefault("session_id", sid)
        return body

    @app.post("/sessions/{sid}/text")
    async def append_text(sid: str, req: Request):
        try:
            return await adapter().append_text(await _body_with_session(req, sid))
        except (AdapterError, ModelNotLoaded) as exc:
            return _err(exc)

    @app.post("/sessions/{sid}/audio")
    async def append_audio(sid: str, req: Request):
        try:
            return await adapter().append_audio(await _body_with_session(req, sid))
        except (AdapterError, ModelNotLoaded) as exc:
            return _err(exc)

    @app.post("/sessions/{sid}/video_frame")
    async def append_video_frame(sid: str, req: Request):
        try:
            return await adapter().append_video_frame(
                await _body_with_session(req, sid)
            )
        except (AdapterError, ModelNotLoaded) as exc:
            return _err(exc)

    @app.post("/sessions/{sid}/video_segment")
    async def append_video_segment(sid: str, req: Request):
        try:
            return await adapter().append_video_segment(
                await _body_with_session(req, sid)
            )
        except (AdapterError, ModelNotLoaded) as exc:
            return _err(exc)

    @app.post("/sessions/{sid}/prefill")
    async def append_prefill(sid: str, req: Request):
        try:
            return await adapter().append_prefill(await _body_with_session(req, sid))
        except (AdapterError, ModelNotLoaded) as exc:
            return _err(exc)

    @app.post("/sessions/{sid}/playback_ack")
    async def acknowledge_playback(sid: str, req: Request):
        try:
            return await adapter().acknowledge_playback(
                await _body_with_session(req, sid)
            )
        except (AdapterError, ModelNotLoaded) as exc:
            return _err(exc)

    @app.post("/sessions/{sid}/output")
    async def next_output_step(
        sid: str, incarnation: int, timeout_s: float | None = None
    ):
        try:
            a = adapter()
            ts = timeout_s if timeout_s is not None else config.default_timeout_s
            return await a.next_output_step(sid, incarnation=incarnation, timeout_s=ts)
        except (AdapterError, ModelNotLoaded, OutputTimeout) as exc:
            return _err(exc)

    return app


def _parse_args(argv: list[str] | None = None) -> _ServerConfig:
    p = argparse.ArgumentParser(
        description="FastAPI Realtime-Venus-Omni VenusOmni ServingPort server"
    )
    p.add_argument(
        "--model-path", default=None, help="path to model_weight/ (trust_remote_code)"
    )
    p.add_argument("--ref-audio", default=None, help="16 kHz reference voice wav")
    p.add_argument(
        "--prompt-wav", default=None, help="optional prompt_wav_path for token2wav init"
    )
    p.add_argument("--memory-minutes", type=int, default=40)
    p.add_argument("--model-type", choices=("audio", "omni"), default="omni")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument(
        "--system-prompt",
        default=None,
        help="duplex system-prompt text (default: Streaming Omni Conversation.)",
    )
    p.add_argument(
        "--system-prompt-file",
        default=None,
        help="read system prompt from a UTF-8 file (overrides --system-prompt)",
    )
    p.add_argument(
        "--system-prompt-model-default",
        action="store_true",
        help="use the model's built-in DEFAULT_SYSTEM_PROMPT (Chinese-only) instead of the streaming default",
    )
    p.add_argument("--default-timeout-s", type=float, default=30.0)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="info")
    a = p.parse_args(argv)
    if a.system_prompt_model_default:
        system_prompt = None
    elif a.system_prompt_file:
        system_prompt = Path(a.system_prompt_file).read_text(encoding="utf-8")
    elif a.system_prompt is not None:
        system_prompt = a.system_prompt
    else:
        system_prompt = REALTIME_VENUS_SYSTEM_PROMPT
    return _ServerConfig(
        model_path=a.model_path,
        ref_audio_path=a.ref_audio,
        prompt_wav_path=a.prompt_wav,
        memory_minutes=a.memory_minutes,
        model_type=a.model_type,
        model_revision=a.model_revision or f"realtime-venus-{a.model_type}-v1",
        model_name=a.model_name or model_name(a.model_type),
        system_prompt=system_prompt,
        default_timeout_s=a.default_timeout_s,
        host=a.host,
        port=a.port,
        log_level=a.log_level,
    )


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    config = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()
