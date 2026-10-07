"""Harness-side remote client implementing ``VenusOmniServingPort`` over HTTP.

Runs in the Venus harness environment (Python >=3.11; the project imports
``harness.bridge`` and ``harness.core``).  It translates the real serving DTOs
(``harness/bridge/serving.py``) to the JSON wire protocol the FastAPI server speaks
(:mod:`demos.model.wire`).  Pass an instance to
``VenusOmniServingHost.open(serving=...)``.

Wire: bytes <-> base64; tuple[int,...] <-> JSON list; Mapping <-> JSON object;
enums <-> ``.value`` strings.  Field names are unchanged (mirror serving.py).
"""

from __future__ import annotations

import base64
from typing import Any, Mapping

import httpx

from harness.bridge.serving import (
    AudioAppend,
    TextAppend,
    AudioFormat,
    CloseSession,
    GeneratedAudio,
    InputAccepted,
    ModelOutputStep,
    OpenSession,
    PlaybackAccepted,
    PlaybackAck,
    PrefillAppend,
    PrefillApplied,
    PrefillDeferred,
    RawTokenMode,
    SessionCapabilities,
    SessionClosed,
    SessionOpened,
    VideoFrameAppend,
    VideoSegmentAppend,
)


class RemoteServingError(RuntimeError):
    pass


class RemoteOutputTimeout(TimeoutError):
    pass


class RemoteSessionClosed(RemoteServingError):
    pass


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _ub64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


class RemoteOmniServingPort:
    """``VenusOmniServingPort`` backed by the FastAPI model server."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 60.0,
        length_penalty: float | None = None,
    ) -> None:
        if length_penalty is not None:
            from .settings import DuplexSettings

            DuplexSettings(length_penalty=length_penalty)
        self._length_penalty = length_penalty
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- request helper -----------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any | None = None,
    ) -> dict[str, Any]:
        resp = await self._client.request(method, path, params=params, json=json)
        if resp.status_code == 200:
            return resp.json()
        detail = ""
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        if resp.status_code == 408:
            raise RemoteOutputTimeout(str(detail))
        if resp.status_code == 409 and path.endswith("/prefill"):
            raise PrefillDeferred(str(detail))
        if resp.status_code == 410:
            raise RemoteSessionClosed(str(detail))
        raise RemoteServingError(f"HTTP {resp.status_code}: {detail}")

    # -- the 8 methods ------------------------------------------------------

    async def open_session(self, request: OpenSession) -> SessionOpened:
        body = {
            "session_id": request.session_id,
            "model": request.model,
            "protocol_version": request.protocol_version,
        }
        if self._length_penalty is not None:
            body["length_penalty"] = self._length_penalty
        j = await self._request("POST", "/sessions", json=body)
        caps = j["capabilities"]
        capabilities = SessionCapabilities(
            special_token_ids=dict(caps["special_token_ids"]),
            raw_token_mode=RawTokenMode(caps["raw_token_mode"]),
            same_kv_prefill=bool(caps.get("same_kv_prefill", True)),
            duplex_media=bool(caps.get("duplex_media", True)),
            ordered_input_fence=bool(caps.get("ordered_input_fence", True)),
            native_audio_output=bool(caps.get("native_audio_output", True)),
            video_segments=bool(caps.get("video_segments", False)),
            max_sessions=int(caps.get("max_sessions", 1)),
            delegate_audio_filtered=caps.get("delegate_audio_filtered", False),
        )
        return SessionOpened(
            session_id=j["session_id"],
            incarnation=int(j["incarnation"]),
            model=j["model"],
            model_revision=j["model_revision"],
            capabilities=capabilities,
            opened_at_ms=int(j["opened_at_ms"]),
        )

    async def append_text(self, request: TextAppend) -> InputAccepted:
        j = await self._request("POST", f"/sessions/{request.session_id}/text", json={
            "session_id": request.session_id, "incarnation": request.incarnation,
            "event_seq": request.event_seq, "text": request.text,
        })
        return InputAccepted(**j)

    async def append_audio(self, request: AudioAppend) -> InputAccepted:
        body = {
            "session_id": request.session_id,
            "incarnation": request.incarnation,
            "event_seq": request.event_seq,
            "start_ms": request.start_ms,
            "end_ms": request.end_ms,
            "data": _b64(request.data),
            "format": request.format.value,
            "sample_rate_hz": request.sample_rate_hz,
            "channels": request.channels,
            "sample_width_bytes": request.sample_width_bytes,
        }
        j = await self._request(
            "POST", f"/sessions/{request.session_id}/audio", json=body
        )
        return InputAccepted(
            session_id=j["session_id"],
            incarnation=int(j["incarnation"]),
            event_seq=int(j["event_seq"]),
            input_seq=int(j["input_seq"]),
            accepted_at_ms=int(j["accepted_at_ms"]),
        )

    async def append_video_frame(self, request: VideoFrameAppend) -> InputAccepted:
        body = {
            "session_id": request.session_id,
            "incarnation": request.incarnation,
            "event_seq": request.event_seq,
            "captured_at_ms": request.captured_at_ms,
            "data": _b64(request.data),
            "mime_type": request.mime_type,
        }
        j = await self._request(
            "POST", f"/sessions/{request.session_id}/video_frame", json=body
        )
        return InputAccepted(
            session_id=j["session_id"],
            incarnation=int(j["incarnation"]),
            event_seq=int(j["event_seq"]),
            input_seq=int(j["input_seq"]),
            accepted_at_ms=int(j["accepted_at_ms"]),
        )

    async def append_video_segment(self, request: VideoSegmentAppend) -> InputAccepted:
        body = {
            "session_id": request.session_id,
            "incarnation": request.incarnation,
            "event_seq": request.event_seq,
            "start_ms": request.start_ms,
            "end_ms": request.end_ms,
            "data": _b64(request.data),
            "fps": request.fps,
            "mime_type": request.mime_type,
        }
        j = await self._request(
            "POST", f"/sessions/{request.session_id}/video_segment", json=body
        )
        return InputAccepted(
            session_id=j["session_id"],
            incarnation=int(j["incarnation"]),
            event_seq=int(j["event_seq"]),
            input_seq=int(j["input_seq"]),
            accepted_at_ms=int(j["accepted_at_ms"]),
        )

    async def next_output_step(
        self, session_id: str, *, incarnation: int, timeout_s: float | None = None
    ) -> ModelOutputStep:
        params = {"incarnation": incarnation}
        if timeout_s is not None:
            params["timeout_s"] = timeout_s
        j = await self._request("POST", f"/sessions/{session_id}/output", params=params)
        audio_json = j.get("audio")
        audio = None
        if audio_json is not None:
            audio = GeneratedAudio(
                data=_ub64(audio_json["data"]),
                sample_rate_hz=int(audio_json.get("sample_rate_hz", 24000)),
                channels=int(audio_json.get("channels", 1)),
                sample_width_bytes=int(audio_json.get("sample_width_bytes", 2)),
                format=AudioFormat(audio_json.get("format", "pcm_s16le")),
            )
        return ModelOutputStep(
            session_id=j["session_id"],
            incarnation=int(j["incarnation"]),
            generation_id=j["generation_id"],
            generation_epoch=int(j["generation_epoch"]),
            step_seq=int(j["step_seq"]),
            input_seq_cutoff=int(j["input_seq_cutoff"]),
            at_ms=int(j["at_ms"]),
            total_token_ids=tuple(int(x) for x in j["total_token_ids"]),
            audio=audio,
            audio_chunk_seq=(
                None if j.get("audio_chunk_seq") is None else int(j["audio_chunk_seq"])
            ),
            turn_finished=bool(j["turn_finished"]),
            finish_reason=j.get("finish_reason"),
        )

    async def append_prefill(self, request: PrefillAppend) -> PrefillApplied:
        body = {
            "session_id": request.session_id,
            "incarnation": request.incarnation,
            "work_id": request.work_id,
            "attempt_id": request.attempt_id,
            "text_list": list(request.text_list),
            "visibility": request.visibility.value,
            "resume_generation": bool(request.resume_generation),
        }
        j = await self._request(
            "POST", f"/sessions/{request.session_id}/prefill", json=body
        )
        return PrefillApplied(
            session_id=j["session_id"],
            incarnation=int(j["incarnation"]),
            work_id=j["work_id"],
            attempt_id=j["attempt_id"],
            generation_id=j["generation_id"],
            generation_epoch=int(j["generation_epoch"]),
            kv_position=int(j["kv_position"]),
            applied_at_ms=int(j["applied_at_ms"]),
            deduplicated=bool(j.get("deduplicated", False)),
        )

    async def acknowledge_playback(self, request: PlaybackAck) -> PlaybackAccepted:
        body = {
            "session_id": request.session_id,
            "incarnation": request.incarnation,
            "utterance_id": request.utterance_id,
            "cumulative_played_chunks": request.cumulative_played_chunks,
            "at_ms": request.at_ms,
            "caused_by_work_id": request.caused_by_work_id,
        }
        j = await self._request(
            "POST", f"/sessions/{request.session_id}/playback_ack", json=body
        )
        return PlaybackAccepted(
            session_id=j["session_id"],
            incarnation=int(j["incarnation"]),
            utterance_id=j["utterance_id"],
            cumulative_played_chunks=int(j["cumulative_played_chunks"]),
            accepted_at_ms=int(j["accepted_at_ms"]),
        )

    async def close_session(self, request: CloseSession) -> SessionClosed:
        params = {"incarnation": request.incarnation, "reason": request.reason}
        j = await self._request(
            "DELETE", f"/sessions/{request.session_id}", params=params
        )
        return SessionClosed(
            session_id=j["session_id"],
            incarnation=int(j["incarnation"]),
            closed_at_ms=int(j["closed_at_ms"]),
        )


__all__ = [
    "RemoteOmniServingPort",
    "RemoteServingError",
    "RemoteOutputTimeout",
    "RemoteSessionClosed",
]
