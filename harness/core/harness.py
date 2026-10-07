"""Public async facade for explicit requests and streamed duplex adapters."""

from __future__ import annotations

import asyncio
from typing import Optional

from .backend import DelegateBackend
from .config import HarnessConfig
from .frontend import assemble_frontend_message
from .models import (
    AudioChunk,
    ContextSnapshot,
    DelegateRequest,
    DelegateResult,
    VideoFrame,
    VideoSegment,
)
from .scheduler import DelegateScheduler
from .session import DuplexProtocolSession, EventListener


class DelegateHarness:
    """Manage isolated delegate lanes for arbitrary realtime frontends.

    Frontends may submit explicit requests through ``submit_delegate`` or use
    the optional streamed ``<delegate>`` protocol methods. This class never
    imports or invokes a frontend model or TTS implementation.
    """

    def __init__(
        self,
        config: Optional[HarnessConfig] = None,
        *,
        backend: DelegateBackend,
        event_listener: Optional[EventListener] = None,
        scheduler: Optional[DelegateScheduler] = None,
    ) -> None:
        self._config = config or HarnessConfig()
        self._backend = backend
        self._event_listener = event_listener
        self._scheduler = scheduler or DelegateScheduler(self._config.runtime)
        self._sessions: dict[str, DuplexProtocolSession] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def open_session(
        self,
        session_id: str,
        *,
        backend_name: Optional[str] = None,
        identity: Optional[str] = None,
        event_listener: Optional[EventListener] = None,
    ) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        selected_backend = backend_name or self._config.default_backend
        async with self._lock:
            self._ensure_open()
            if session_id in self._sessions:
                raise ValueError("session is already open: {}".format(session_id))
            self._sessions[session_id] = DuplexProtocolSession(
                session_id=session_id,
                config=self._config,
                backend=self._backend,
                backend_name=selected_backend,
                scheduler=self._scheduler,
                identity=identity,
                event_listener=event_listener or self._event_listener,
            )

    async def ingest_user_file(self, session_id: str, file) -> int:
        return await (await self._session(session_id)).ingest_file(file)

    async def ingest_user_text(self, session_id: str, text: str) -> int:
        return await (await self._session(session_id)).ingest_text(text)

    async def ingest_user_audio(self, session_id: str, chunk: AudioChunk) -> int:
        return await (await self._session(session_id)).ingest_audio(chunk)

    async def ingest_user_video_segment(
        self, session_id: str, segment: VideoSegment
    ) -> int:
        return await (await self._session(session_id)).ingest_video_segment(segment)

    async def ingest_user_video_frame(self, session_id: str, frame: VideoFrame) -> int:
        return await (await self._session(session_id)).ingest_video_frame(frame)

    async def append_tts_text(
        self, session_id: str, utterance_id: str, text: str
    ) -> None:
        await (await self._session(session_id)).append_tts_text(utterance_id, text)

    async def set_identity(self, session_id: str, identity: Optional[str]) -> str:
        """Set or reset the public assistant name for a live session."""

        return await (await self._session(session_id)).set_identity(identity)

    async def submit_delegate(
        self,
        session_id: str,
        query: str,
        *,
        snapshot: Optional[ContextSnapshot] = None,
        identity: Optional[str] = None,
        at_ms: Optional[int] = None,
    ) -> str:
        """Submit a query without coupling the caller to a text-tag protocol."""

        return await (await self._session(session_id)).submit_delegate(
            query,
            snapshot=snapshot,
            identity=identity,
            at_ms=at_ms,
        )

    async def submit_request(self, request: DelegateRequest) -> str:
        """Submit a caller-built request through the shared task runtime."""

        return await (await self._session(request.session_id)).submit_request(request)

    async def acknowledge_tts_playback(
        self,
        session_id: str,
        utterance_id: str,
        played_chars: int,
        *,
        at_ms: Optional[int] = None,
    ) -> str:
        return await (await self._session(session_id)).acknowledge_tts_playback(
            utterance_id,
            played_chars,
            at_ms=at_ms,
        )

    async def on_model_text(
        self,
        session_id: str,
        delta: str,
        *,
        at_ms: Optional[int] = None,
        cutoff_sequence: Optional[int] = None,
    ) -> str:
        return await (await self._session(session_id)).on_model_text(
            delta,
            at_ms=at_ms,
            cutoff_sequence=cutoff_sequence,
        )

    async def finish_model_turn(
        self, session_id: str, *, at_ms: Optional[int] = None
    ) -> str:
        return await (await self._session(session_id)).finish_model_turn(at_ms=at_ms)

    async def next_delegate_result(
        self,
        session_id: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> DelegateResult:
        return await (await self._session(session_id)).next_result(timeout_s)

    async def next_frontend_message(
        self,
        session_id: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> dict[str, object]:
        """Consume one result and return its ready-to-inject model message.

        A pending message ends that delivery event, while its request continues
        in the background and later publishes a final message with the same ID.
        """

        result = await self.next_delegate_result(session_id, timeout_s=timeout_s)
        return assemble_frontend_message(result)

    async def discard_feedback(self, session_id, feedback_id):
        session = await self._session(session_id)
        await session._runtime.delivery.discard_feedback(feedback_id)

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.aclose()

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        await asyncio.gather(*(session.aclose() for session in sessions))
        await self._backend.aclose()

    async def _session(self, session_id: str) -> DuplexProtocolSession:
        async with self._lock:
            self._ensure_open()
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("unknown harness session: {}".format(session_id))
        return session

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("DelegateHarness is closed")


class DuplexHarness(DelegateHarness):
    """Backward-compatible name for existing duplex-model integrations."""
