"""The Realtime-Venus experience: one live session, uploads, settings and task results."""

import asyncio
import json
import secrets
from functools import partial
from pathlib import Path
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .configuration import WebConfiguration
from .resources import real_resources
from .session import WebAgentSession
from demos.variants import model_name


def create_app(
    *, settings_path, tokenizer_path, model_server_url="http://127.0.0.1:8031", model_type="omni", demo_path=None
):
    frontend_model = model_name(model_type)
    configuration = WebConfiguration(settings_path, demo_path=demo_path)
    static = Path(__file__).resolve().parents[1] / "static"
    sessions = {}
    app = FastAPI(
        title="Realtime-Venus", docs_url=None, redoc_url=None, openapi_url=None
    )
    app.mount("/static", StaticFiles(directory=static), name="static")

    def same_origin(request):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "Origin rejected")

    def authorized_session(sid, request):
        same_origin(request)
        session = sessions.get(sid)
        if session is None or not session.host:
            raise HTTPException(404, "Session has ended")
        if not secrets.compare_digest(
            request.headers.get("x-venus-session-token", ""), session.upload_token
        ):
            raise HTTPException(403, "Session token rejected")
        return session

    @app.get("/")
    async def index():
        return FileResponse(
            static / "index.html", headers={"Cache-Control": "no-store"}
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "product": "Realtime-Venus-Harness"}

    @app.get("/api/config")
    async def get_configuration():
        try:
            return Response(
                json.dumps(
                    {**await asyncio.to_thread(configuration.public), "active_sessions": len(sessions)}
                ),
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        except (ValueError, TypeError, OSError, KeyError):
            raise HTTPException(
                400, "Cannot read local settings / 无法读取配置"
            ) from None

    @app.post("/api/config")
    async def save_configuration(request: Request):
        same_origin(request)
        if not secrets.compare_digest(
            request.headers.get("x-venus-config-token", ""), configuration.token
        ):
            raise HTTPException(403, "Open settings before saving")
        if sessions:
            raise HTTPException(
                409, "End your conversation before changing settings / 请先结束会话"
            )
        raw = await request.body()
        if len(raw) > 128 * 1024:
            raise HTTPException(413, "Settings are too large")
        if sessions:
            raise HTTPException(409, "End your conversation before changing settings")
        try:
            result = configuration.save(json.loads(raw))
            if not result["ok"]:
                return Response(
                    json.dumps(result), status_code=422, media_type="application/json"
                )
            return await asyncio.to_thread(configuration.public)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None
        except (ValueError, TypeError, OSError, KeyError):
            raise HTTPException(
                400, "Invalid settings / 配置无效，请检查输入"
            ) from None

    @app.get("/api/status")
    async def status(fresh: bool = False):
        return {
            "model": frontend_model,
            "model_type": model_type,
            "input_modes": ["audio", "audio_file", "text"] if model_type == "audio" else ["camera", "video", "audio_file", "text"],
            "configured": await asyncio.to_thread(configuration.status, force_login=fresh),
            "busy": bool(sessions),
            "upload_limit_mb": 200,
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        if origin and urlsplit(origin).netloc != websocket.headers.get("host"):
            await websocket.close(code=1008)
            return
        mode = websocket.query_params.get("mode", model_type)
        source = websocket.query_params.get("source", "live")
        valid_source = {"live", "audio_file", "text"} if model_type == "audio" else {"live", "video", "audio_file", "text"}
        if mode != model_type or source not in valid_source:
            await websocket.accept()
            await websocket.send_json({"type": "fatal_error", "code": "input_mode", "error": "Input source does not match the deployed model"})
            await websocket.close(code=1008)
            return
        revision = configuration.revision()
        ready = False if sessions else await asyncio.to_thread(configuration.status, force_login=True)
        # Recheck the slot after awaiting the login probe: another client may have entered.
        if sessions or not ready or revision != configuration.revision():
            await websocket.accept()
            await websocket.send_json(
                {
                    "type": "fatal_error",
                    "code": "busy" if sessions else "configuration",
                    "error": "Another conversation is using Venus / 已有会话正在使用 Venus"
                    if sessions
                    else "Set up your connection first / 请先完成连接设置",
                }
            )
            await websocket.close(code=1008)
            return
        session = WebAgentSession(
            websocket,
            partial(
                real_resources,
                server_url=model_server_url,
                tokenizer_path=tokenizer_path,
                model_type=model_type,
                demo_path=demo_path,
            ),
            mode=mode,
            input_source=source,
            settings_path=str(configuration.path),
        )
        sessions[session.session_id] = session
        try:
            await session.run()
        finally:
            sessions.pop(session.session_id, None)

    @app.post("/api/sessions/{sid}/video", status_code=202)
    async def upload_video(sid: str, request: Request):
        session = authorized_session(sid, request)
        try:
            await session.upload_video(request.stream())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"accepted": True}

    @app.post("/api/sessions/{sid}/audio", status_code=202)
    async def upload_audio(sid: str, request: Request):
        session = authorized_session(sid, request)
        try:
            await session.upload_audio(request.stream())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"accepted": True}

    @app.get("/api/sessions/{sid}/works/{work_id}/artifacts/{index}")
    async def download_artifact(sid: str, work_id: str, index: int, request: Request):
        session = authorized_session(sid, request)
        try:
            result = session.host.session.get_work_result(work_id) or {}
            artifacts = result.get("artifacts", [])
            if index < 0 or index >= len(artifacts):
                raise ValueError("Unknown artifact")
            data = await asyncio.to_thread(
                session.host.session.read_work_artifact, work_id, index
            )
            name = Path(artifacts[index]["path"]).name
        except (ValueError, KeyError, OSError, RuntimeError):
            raise HTTPException(404, "Result is unavailable / 结果暂不可用") from None
        return Response(
            data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename*=UTF-8''" + quote(name),
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
            },
        )

    return app
