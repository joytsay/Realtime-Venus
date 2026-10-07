"""Duplex text/media protocol adapter around the frontend-neutral runtime."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Optional
from uuid import uuid4

from .backend import DelegateBackend
from .buffers import SessionBuffers
from .config import HarnessConfig
from .delegate_parser import DelegateParser
from .events import Clock, EventEmitter, EventListener, now_ms
from .models import (
    AudioChunk,
    ContextSnapshot,
    DelegateRequest,
    DelegateResult,
    InputFile,
    VideoFrame,
    VideoSegment,
)
from .policy import ContextWindowPolicy
from .runtime import DelegateSessionRuntime
from .scheduler import DelegateScheduler

MAX_IDENTITY_CHARS = 80


def _normalise_identity(identity: Optional[str], default_identity: str) -> str:
    value = " ".join((identity if identity is not None else default_identity).split())
    if not value:
        value = " ".join(default_identity.split())
    return value[:MAX_IDENTITY_CHARS]


class DuplexProtocolSession:
    """Translate streamed duplex-model signals into explicit delegate requests.

    This layer owns only the frontend protocol and its volatile evidence. The
    runtime receives complete ``DelegateRequest`` objects and has no knowledge
    of control tags, tokenization, TTS, or a particular duplex model.
    """

    def __init__(
        self,
        *,
        session_id: str,
        config: HarnessConfig,
        backend: DelegateBackend,
        backend_name: str,
        scheduler: DelegateScheduler,
        identity: Optional[str] = None,
        event_listener: Optional[EventListener] = None,
        clock: Clock = now_ms,
    ) -> None:
        self.session_id = session_id
        self._config = config
        self._backend_name = backend_name
        self._clock = clock
        self._emit = EventEmitter(session_id, event_listener)
        self._identity = _normalise_identity(identity, config.default_identity)
        self._parser = DelegateParser(config.delegate.max_query_chars)
        self._policy = ContextWindowPolicy(config.buffer)
        self._buffers = SessionBuffers(
            retention_ms=config.buffer.retention_ms,
            max_audio_bytes=config.buffer.max_audio_bytes,
            max_video_bytes=config.buffer.max_video_bytes,
            max_video_frames=config.buffer.max_video_frames,
        )
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._input_files: list[tuple[int, InputFile]] = []
        self._input_bytes = 0
        self._open_snapshot: Optional[ContextSnapshot] = None
        self._open_identity: Optional[str] = None
        self._closed = False
        self._runtime = DelegateSessionRuntime(
            session_id=session_id,
            config=config,
            backend=backend,
            backend_name=backend_name,
            scheduler=scheduler,
            clock=clock,
            emit=self._emit,
        )

    async def ingest_file(self, file: InputFile) -> int:
        async with self._lock:
            self._ensure_open()
            if self._input_bytes + len(file.data) > 64 * 1024 * 1024:
                raise ValueError("session attachment budget exceeded (64 MiB)")
            # Attachments do not advance the media sequence agreed with Serving.
            self._input_files.append((self._sequence, file))
            self._input_bytes += len(file.data)
            return self._sequence

    async def ingest_text(self, text: str) -> int:
        async with self._lock:
            self._ensure_open()
            data = text.encode("utf-8")
            if self._input_bytes + len(data) > 64 * 1024 * 1024:
                raise ValueError("session attachment budget exceeded (64 MiB)")
            self._sequence += 1
            self._input_files.append((self._sequence, InputFile(
                f"typed-message-{self._sequence}.txt", data, "text/plain"
            )))
            self._input_bytes += len(data)
            return self._sequence

    async def ingest_audio(self, chunk: AudioChunk) -> int:
        async with self._lock:
            self._ensure_open()
            self._sequence += 1
            self._buffers.add_audio(self._sequence, chunk)
            await self._runtime.delivery.observe_user_audio(chunk)
            return self._sequence

    async def ingest_video_segment(self, segment: VideoSegment) -> int:
        async with self._lock:
            self._ensure_open()
            self._sequence += 1
            self._buffers.add_video_segment(self._sequence, segment)
            return self._sequence

    async def ingest_video_frame(self, frame: VideoFrame) -> int:
        async with self._lock:
            self._ensure_open()
            self._sequence += 1
            self._buffers.add_video_frame(self._sequence, frame)
            return self._sequence

    async def append_tts_text(self, utterance_id: str, text: str) -> None:
        async with self._lock:
            self._ensure_open()
            self._buffers.played_tts.append_text(utterance_id, text)
            await self._runtime.delivery.observe_tts_text(utterance_id, text)

    async def set_identity(self, identity: Optional[str]) -> str:
        async with self._lock:
            self._ensure_open()
            self._identity = _normalise_identity(
                identity, self._config.default_identity
            )
            return self._identity

    async def acknowledge_tts_playback(
        self,
        utterance_id: str,
        played_chars: int,
        *,
        at_ms: Optional[int] = None,
    ) -> str:
        async with self._lock:
            self._ensure_open()
            playback_at_ms = self._clock() if at_ms is None else at_ms
            newly_played = self._buffers.played_tts.advance(
                utterance_id,
                played_chars,
                playback_at_ms,
            )
            await self._runtime.delivery.acknowledge_playback(
                utterance_id, played_chars
            )
            return newly_played

    async def submit_delegate(
        self,
        query: str,
        *,
        snapshot: Optional[ContextSnapshot] = None,
        identity: Optional[str] = None,
        at_ms: Optional[int] = None,
    ) -> str:
        """Submit work directly, without emitting model-specific control tags."""

        query = query.strip()
        if not query:
            raise ValueError("delegate query must not be empty")
        if len(query) > self._config.delegate.max_query_chars:
            raise ValueError("delegate query exceeds configured character limit")
        async with self._lock:
            self._ensure_open()
            created_at_ms = self._clock() if at_ms is None else at_ms
            selected_snapshot = snapshot or self._capture_locked(created_at_ms)
            if selected_snapshot.session_id != self.session_id:
                raise ValueError("context snapshot belongs to another session")
            selected_identity = _normalise_identity(identity, self._identity)
            self._emit_snapshot(selected_snapshot, created_at_ms)
            self._emit(
                "delegate_opened",
                created_at_ms,
                details={
                    "snapshot_id": selected_snapshot.snapshot_id,
                    "source": "explicit",
                },
            )
            request = self._build_request(
                query, selected_snapshot, selected_identity, created_at_ms
            )
            return await self._runtime.submit(request)

    async def submit_request(self, request: DelegateRequest) -> str:
        """Submit a fully formed request from a non-duplex frontend adapter."""

        async with self._lock:
            self._ensure_open()
            return await self._runtime.submit(request)

    async def on_model_text(
        self,
        delta: str,
        *,
        at_ms: Optional[int] = None,
        cutoff_sequence: Optional[int] = None,
    ) -> str:
        """Parse one streamed control-protocol delta and return visible text."""

        async with self._lock:
            self._ensure_open()
            now = self._clock() if at_ms is None else at_ms
            step = self._parser.feed(delta)
            if step.delegate_candidate_abandoned:
                self._clear_open_delegate()
                self._emit(
                    "delegate_candidate_abandoned",
                    now,
                    details={"reason": "partial delegate tag resolved as visible text"},
                )
            if step.delegate_candidate_started:
                self._open_snapshot = self._capture_locked(
                    now,
                    cutoff_sequence=cutoff_sequence,
                )
                self._open_identity = self._identity
                self._emit_snapshot(self._open_snapshot, now)
            if step.opened_delegate:
                if self._open_snapshot is None:
                    self._emit(
                        "delegate_rejected",
                        now,
                        details={
                            "reason": "delegate opening tag has no frozen context"
                        },
                    )
                else:
                    self._emit(
                        "delegate_opened",
                        now,
                        details={"snapshot_id": self._open_snapshot.snapshot_id},
                    )
            if step.protocol_error:
                self._clear_open_delegate()
                self._emit(
                    "delegate_rejected", now, details={"reason": step.protocol_error}
                )
                return step.visible_text
            if step.delegate_query is not None:
                if self._open_snapshot is None:
                    self._emit(
                        "delegate_rejected",
                        now,
                        details={
                            "reason": "delegate query closed without an opening snapshot"
                        },
                    )
                else:
                    request = self._build_request(
                        step.delegate_query,
                        self._open_snapshot,
                        self._open_identity or self._identity,
                        now,
                    )
                    await self._runtime.submit(request)
                self._clear_open_delegate()
            await self._runtime.delivery.observe_frontend_text(step.visible_text)
            return step.visible_text

    async def finish_model_turn(self, *, at_ms: Optional[int] = None) -> str:
        async with self._lock:
            self._ensure_open()
            now = self._clock() if at_ms is None else at_ms
            step = self._parser.finish_turn()
            if step.protocol_error:
                self._emit(
                    "delegate_rejected", now, details={"reason": step.protocol_error}
                )
            elif step.delegate_candidate_abandoned:
                self._emit(
                    "delegate_candidate_abandoned",
                    now,
                    details={
                        "reason": "model turn ended during a partial delegate tag"
                    },
                )
            self._clear_open_delegate()
            await self._runtime.delivery.finish_frontend_turn(step.visible_text)
            return step.visible_text

    async def next_result(self, timeout_s: Optional[float] = None) -> DelegateResult:
        return await self._runtime.next_result(timeout_s)

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._clear_open_delegate()
        await self._runtime.aclose()

    def _capture_locked(
        self,
        cutoff_ms: int,
        *,
        cutoff_sequence: Optional[int] = None,
    ) -> ContextSnapshot:
        selected_sequence = (
            self._sequence if cutoff_sequence is None else cutoff_sequence
        )
        if isinstance(selected_sequence, bool) or not isinstance(
            selected_sequence, int
        ):
            raise TypeError("cutoff_sequence must be an integer")
        if selected_sequence < 0 or selected_sequence > self._sequence:
            raise ValueError("cutoff_sequence is outside the ingested media range")
        snapshot = self._policy.capture(
            session_id=self.session_id,
            buffers=self._buffers,
            cutoff_ms=cutoff_ms,
            cutoff_sequence=selected_sequence,
        )
        files = {f.name: f for seq, f in self._input_files if seq <= selected_sequence}
        return replace(snapshot, input_files=tuple(files.values()))

    def _emit_snapshot(self, snapshot: ContextSnapshot, at_ms: int) -> None:
        self._emit(
            "delegate_context_frozen",
            at_ms,
            details={
                "snapshot_id": snapshot.snapshot_id,
                "audio_chunks": len(snapshot.audio_chunks),
                "video_segments": len(snapshot.video_segments),
                "video_frames": len(snapshot.video_frames),
            },
        )

    def _build_request(
        self,
        query: str,
        snapshot: ContextSnapshot,
        identity: str,
        created_at_ms: int,
    ) -> DelegateRequest:
        return DelegateRequest(
            work_id=str(uuid4()),
            session_id=self.session_id,
            query=query,
            snapshot=snapshot,
            created_at_ms=created_at_ms,
            expires_at_ms=created_at_ms + self._config.delegate.result_ttl_ms,
            backend_name=self._backend_name,
            routing_locked=self._config.delegate.default_routing_locked,
            allow_web_search=self._config.delegate.default_allow_web_search,
            identity=identity,
            input_mode=self._config.delegate.default_input_mode,
            operation=self._config.delegate.default_operation,
            response_mode=self._config.delegate.default_response_mode,
            language=self._config.language.value,
            available_operations=tuple(
                operation.value
                for operation, capability in self._config.capabilities.items()
                if capability.enabled
            ),
        )

    def _clear_open_delegate(self) -> None:
        self._open_snapshot = None
        self._open_identity = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("harness session is closed")


# Compatibility for integrations that imported the old internal class name.
SessionDelegateRunner = DuplexProtocolSession

__all__ = ["DuplexProtocolSession", "EventListener", "SessionDelegateRunner"]
