"""Realtime-Venus-Omni sessions: frozen delegate inputs, background execution, private backend prefill and native Talker playback receipts."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from harness.core.backend import DelegateBackend
from harness.core.config import HarnessConfig
from harness.core.frontend import assemble_frontend_message
from harness.core.harness import DelegateHarness
from harness.core.models import AudioChunk, InputFile, VideoFrame, VideoSegment

from .boundary import VenusOmniDuplexBoundary
from .config import venus_harness_config
from .contracts import BackendInjection, ModelStep
from .execution import AgentDelegateBackend


@dataclass(slots=True)
class _DeliveryReceipt:
    work_id: str
    track_completed_work: bool = False
    feedback_id: str = ""
    utterance_id: str = ""
    total_chunks: int = 0
    played_chunks: int = 0
    frontend_turn_finished: bool = False


class VenusOmniSession:
    """One live VenusOmni conversation and its model-safe delivery state."""

    def __init__(
        self,
        *,
        owner: VenusOmniAgentHarness,
        session_id: str,
        tokenizer: Any,
        special_token_ids: dict[str, int] | None = None,
    ) -> None:
        self._owner = owner
        self.session_id = session_id
        self.boundary = VenusOmniDuplexBoundary(
            harness=owner.delegate_harness,
            session_id=session_id,
            tokenizer=tokenizer,
            special_token_ids=special_token_ids,
        )
        self._receipts: dict[str, _DeliveryReceipt] = {}
        self._request_by_utterance: dict[str, str] = {}
        self._used_utterance_ids: set[str] = set()
        self._backend_utterance_ids: set[str] = set()
        self._chunk_char_boundaries: dict[str, list[int]] = {}
        self._played_chunks: dict[str, int] = {}
        # Media ingress and raw-token observation share one serial lane.  This
        # makes the raw <delegate> observation and its T0 sequence cutoff
        # atomic with respect to later camera/microphone ingress.
        self._protocol_lane = asyncio.Lock()
        # All receipt and Work delivery transitions use one lock so playback
        # completion cannot race another acknowledgement.
        self._delivery_lock = asyncio.Lock()
        self._closed = False

    async def ingest_user_file(
        self, name: str, data: bytes, mime_type: str = "application/octet-stream"
    ) -> int:
        self._ensure_open()
        async with self._protocol_lane:
            return await self._owner.delegate_harness.ingest_user_file(
                self.session_id, InputFile(name, data, mime_type)
            )

    def get_work_result(self, work_id: str):
        self._ensure_open()
        return self._owner.execution_backend.get_work_result(self.session_id, work_id)

    def read_work_artifact(self, work_id: str, index: int = 0) -> bytes:
        self._ensure_open()
        return self._owner.execution_backend.read_work_artifact(
            self.session_id, work_id, index
        )

    def get_work_events(self, work_id: str):
        self._ensure_open()
        return self._owner.execution_backend.get_work_events(self.session_id, work_id)

    def health_status(self):
        return {
            "backends": self._owner.execution_backend.refresh_backend_health(),
            "storage_error": self._storage_error(),
        }

    def _storage_error(self):
        from harness.jobs.messages import local_text

        return local_text(
            self._owner.execution_backend.journal.error,
            self._owner.config.language.value,
        )

    def recovered_works(self):
        return [
            dict(w)
            for w in self._owner.execution_backend.journal.recovered
            if w["session_id"] == self.session_id
        ]

    def list_works(self):
        self._ensure_open()
        return [
            w for w in self._owner.list_works() if w["session_id"] == self.session_id
        ]

    async def ingest_user_text(self, text: str) -> int:
        self._ensure_open()
        async with self._protocol_lane:
            return await self._owner.delegate_harness.ingest_user_text(self.session_id, text)

    async def ingest_user_audio(self, chunk: AudioChunk) -> int:
        self._ensure_open()
        async with self._protocol_lane:
            return await self._owner.delegate_harness.ingest_user_audio(
                self.session_id, chunk
            )

    async def ingest_user_video_segment(self, segment: VideoSegment) -> int:
        self._ensure_open()
        async with self._protocol_lane:
            return await self._owner.delegate_harness.ingest_user_video_segment(
                self.session_id,
                segment,
            )

    async def ingest_user_video_frame(self, frame: VideoFrame) -> int:
        self._ensure_open()
        async with self._protocol_lane:
            return await self._owner.delegate_harness.ingest_user_video_frame(
                self.session_id,
                frame,
            )

    async def consume_total_ids(
        self,
        total_ids: Any,
        *,
        at_ms: int,
        cutoff_sequence: int | None = None,
        delegate_audio_filtered: bool = False,
    ) -> ModelStep:
        """Consume a cumulative VenusOmni token buffer exactly once.

        A model with verified span filtering may attach safe confirmation
        speech, including buffered tails, to private token steps. Legacy
        models still mute those steps. Stream resets always suppress audio.
        """

        self._ensure_open()
        async with self._protocol_lane:
            consume_kwargs: dict[str, int] = {"at_ms": at_ms}
            if cutoff_sequence is not None:
                consume_kwargs["cutoff_sequence"] = cutoff_sequence
            token_step = await self.boundary.consume_total_ids(
                total_ids,
                **consume_kwargs,
            )
        return ModelStep(
            visible_text=token_step.visible_text,
            closed_delegate_queries=token_step.delegate_queries,
            suppress_audio=token_step.stream_reset
            or (token_step.saw_delegate and not delegate_audio_filtered),
            new_token_count=token_step.consumed_token_count,
        )

    async def finish_model_turn(
        self,
        *,
        at_ms: int,
        caused_by_work_id: str = "",
        caused_by_feedback_id: str = "",
        next_total_ids: Any = (),
    ) -> str:
        """Finish parsing independently of playback acknowledgement."""

        self._ensure_open()
        async with self._protocol_lane:
            tail = await self.boundary.finish_turn(
                at_ms=at_ms,
                next_total_ids=next_total_ids,
            )
        if caused_by_work_id:
            async with self._delivery_lock:
                receipt = self._receipts.get(caused_by_feedback_id or caused_by_work_id)
                if receipt is not None:
                    receipt.frontend_turn_finished = True
            await self._finalize_delivery_if_ready(
                caused_by_feedback_id or caused_by_work_id
            )
        return tail

    async def append_playback_chunk(
        self,
        utterance_id: str,
        chunk_seq: int,
        text: str,
        *,
        caused_by_work_id: str = "",
        caused_by_feedback_id: str = "",
    ) -> None:
        """Register one ordered native Talker audio chunk and optional transcript."""

        self._ensure_open()
        if not utterance_id:
            raise ValueError("utterance_id is required")
        if chunk_seq < 1:
            raise ValueError("chunk_seq must be positive")
        async with self._delivery_lock:
            receipt = None
            if caused_by_work_id:
                receipt = self._receipts.get(caused_by_feedback_id or caused_by_work_id)
                if receipt is None or receipt.work_id != caused_by_work_id:
                    raise ValueError(f"unknown backend delivery: {caused_by_work_id}")
                if receipt.utterance_id and receipt.utterance_id != utterance_id:
                    raise ValueError(
                        "one backend result cannot own multiple utterance IDs"
                    )
                mapped = self._request_by_utterance.get(utterance_id)
                if mapped is not None and mapped != (
                    caused_by_feedback_id or caused_by_work_id
                ):
                    raise ValueError(
                        "utterance_id is already bound to another backend result"
                    )
                if (
                    not receipt.utterance_id
                    and utterance_id in self._used_utterance_ids
                ):
                    raise ValueError(
                        "utterance_id must be globally new for each backend delivery"
                    )
            elif utterance_id in self._backend_utterance_ids:
                raise ValueError("backend utterance text requires caused_by_work_id")

            boundaries = self._chunk_char_boundaries.setdefault(utterance_id, [])
            if chunk_seq != len(boundaries) + 1:
                raise ValueError(
                    "Talker chunk_seq must be contiguous within an utterance"
                )
            if text:
                await self._owner.delegate_harness.append_tts_text(
                    self.session_id,
                    utterance_id,
                    text,
                )
            boundaries.append((boundaries[-1] if boundaries else 0) + len(text))
            self._used_utterance_ids.add(utterance_id)
            if receipt is not None:
                receipt.utterance_id = utterance_id
                receipt.total_chunks = len(boundaries)
                self._request_by_utterance[utterance_id] = (
                    caused_by_feedback_id or caused_by_work_id
                )
                self._backend_utterance_ids.add(utterance_id)

    async def acknowledge_playback_chunks(
        self,
        utterance_id: str,
        cumulative_played_chunks: int,
        *,
        at_ms: int | None = None,
    ) -> str:
        """Acknowledge cumulative native Talker chunks played by the device."""

        self._ensure_open()
        work_id = ""
        async with self._delivery_lock:
            boundaries = self._chunk_char_boundaries.get(utterance_id)
            if boundaries is None:
                raise ValueError(f"unknown playback utterance: {utterance_id}")
            previous_chunks = self._played_chunks.get(utterance_id, 0)
            if cumulative_played_chunks < previous_chunks:
                raise ValueError(
                    "playback acknowledgement must be cumulative and monotonic"
                )
            if cumulative_played_chunks > len(boundaries):
                raise ValueError(
                    "playback acknowledgement exceeds queued Talker chunks"
                )
            work_id = self._request_by_utterance.get(utterance_id, "")
            if not work_id and utterance_id in self._backend_utterance_ids:
                raise ValueError("backend utterance is no longer active")
            target_chars = (
                boundaries[cumulative_played_chunks - 1]
                if cumulative_played_chunks
                else 0
            )
            previous_chars = boundaries[previous_chunks - 1] if previous_chunks else 0
            played = ""
            if target_chars > previous_chars:
                played = await self._owner.delegate_harness.acknowledge_tts_playback(
                    self.session_id,
                    utterance_id,
                    target_chars,
                    at_ms=at_ms,
                )
            self._played_chunks[utterance_id] = cumulative_played_chunks
            if work_id:
                receipt = self._receipts[work_id]
                receipt.played_chunks = cumulative_played_chunks
        if work_id:
            await self._finalize_delivery_if_ready(work_id)
        return played

    async def next_backend_injection(
        self,
        *,
        timeout_s: float | None = None,
    ) -> BackendInjection:
        """Return one at-most-once private prefill for VenusOmni.

        Managed progress and terminal results have separate feedback identities.
        Legacy unversioned pending messages are skipped. Stale progress is
        revalidated again by the host immediately before touching model KV.
        """

        self._ensure_open()
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            if remaining == 0.0:
                raise TimeoutError
            result = await self._owner.delegate_harness.next_delegate_result(
                self.session_id,
                timeout_s=remaining,
            )
            feedback = self._owner.execution_backend.feedback
            if (
                result.status == "pending" and not result.feedback_id
            ) or not feedback.valid(result):
                continue
            if result.status != "pending" and (
                result.metadata.get("stale") is True
                or (
                    result.expires_at_ms
                    and result.completed_at_ms > result.expires_at_ms
                )
                or (
                    result.metadata.get("execution", {}).get("capability") == "general"
                    and result.expires_at_ms
                    and time.time_ns() // 1_000_000 > result.expires_at_ms
                )
            ):
                # Capability/API fallbacks can be produced before the
                # execution backend creates a Work record.  They still expire
                # normally, but there is no Work lifecycle to mutate.
                work = self._owner.execution_backend.get_work(result.work_id)
                if work is not None and str(work["state"]) in {
                    "completed",
                    "delivering",
                }:
                    await self._owner.execution_backend.mark_delivery_failed(
                        result.work_id,
                        "delegate result expired before frontend delivery",
                    )
                continue
            message = assemble_frontend_message(result)
            prepared = self.boundary.prepare_backend_prefill(message)
            if prepared is None:
                continue
            work = self._owner.execution_backend.get_work(result.work_id)
            work_state = str(work["state"]) if work is not None else ""
            if (
                work_state in {"cancelled", "cancelling"}
                and result.metadata.get("kind") != "cancelled"
            ):
                continue
            track_completed_work = (
                result.status == "completed" and work_state == "completed"
            )
            try:
                async with self._delivery_lock:
                    if track_completed_work:
                        await self._owner.execution_backend.mark_delivering(
                            result.work_id
                        )
                    self._receipts[result.delivery_id] = _DeliveryReceipt(
                        result.work_id,
                        track_completed_work=track_completed_work,
                        feedback_id=result.delivery_id,
                    )
            except BaseException:
                self.boundary.rollback_backend_prefill_reservation(result.delivery_id)
                raise
            return BackendInjection(
                work_id=prepared.work_id,
                text=prepared.text,
                result=result,
            )

    async def cancel_work(self, work_id: str) -> dict[str, Any]:
        self._ensure_open()
        self._owner.execution_backend.get_work_result(self.session_id, work_id)
        result = await self._owner.execution_backend.cancel_work(work_id)
        if str(result["state"]) == "cancelled":
            async with self._delivery_lock:
                receipt = self._receipts.pop(work_id, None)
                if receipt and receipt.utterance_id:
                    self._request_by_utterance.pop(receipt.utterance_id, None)
                    self._chunk_char_boundaries.pop(receipt.utterance_id, None)
                    self._played_chunks.pop(receipt.utterance_id, None)
        return result

    def injection_is_current(self, injection):
        return self._owner.execution_backend.feedback.valid(injection.result)

    async def discard_injection(self, injection):
        async with self._delivery_lock:
            self._receipts.pop(injection.feedback_id, None)
        self._owner.execution_backend.feedback.delivered(
            injection.work_id, injection.feedback_id, "superseded"
        )
        await self._owner.delegate_harness.discard_feedback(
            self.session_id, injection.feedback_id
        )

    @property
    def feedback_timeout_s(self):
        return self._owner.config.result_management.max_playback_wait_ms / 1000

    def feedback_failed(self, injection):
        backend = self._owner.execution_backend
        work = backend.work_manager.get(injection.work_id)
        if work:
            work.delivery_error = "前台回传或播放未能确认完成，已保留任务结果。"
            if injection.result.status != "pending":
                work.delivery_status = "unknown"
            backend.feedback.health["frontend"] = {
                "status": "unavailable",
                "code": "DELIVERY_UNCONFIRMED",
            }
        backend.feedback.delivered(injection.work_id, injection.feedback_id, "unknown")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._owner._close_session(self.session_id)

    async def _finalize_delivery_if_ready(self, work_id: str) -> None:
        async with self._delivery_lock:
            receipt = self._receipts.get(work_id)
            if receipt is None:
                return
            ready = bool(
                receipt.frontend_turn_finished
                and receipt.total_chunks > 0
                and receipt.played_chunks >= receipt.total_chunks
            )
            if not ready:
                return
            if receipt.track_completed_work:
                await self._owner.execution_backend.mark_delivered(receipt.work_id)
            self._owner.execution_backend.feedback.delivered(
                receipt.work_id, receipt.feedback_id, "delivered"
            )
            self._receipts.pop(work_id, None)
            if receipt.utterance_id:
                self._request_by_utterance.pop(receipt.utterance_id, None)
                self._chunk_char_boundaries.pop(receipt.utterance_id, None)
                self._played_chunks.pop(receipt.utterance_id, None)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("VenusOmni session is closed")


class VenusOmniAgentHarness:
    """Composition root for VenusOmni boundary, Work/General Agent/Skill, and execution."""

    def __init__(
        self,
        direct_backend: DelegateBackend,
        *,
        planner: Any = None,
        general_agent: Any = None,
        general_config: Any = None,
        skills: Any = None,
        config: HarnessConfig | None = None,
        event_listener: Any = None,
        feedback_config: Any = None,
    ) -> None:
        self.config = config or venus_harness_config()
        self.execution_backend = AgentDelegateBackend(
            direct_backend,
            planner=planner,
            general_agent=general_agent,
            general_config=general_config,
            skills=skills,
            feedback_config=feedback_config,
        )
        self.delegate_harness = DelegateHarness(
            self.config,
            backend=self.execution_backend,
            event_listener=event_listener,
        )
        self._sessions: dict[str, VenusOmniSession] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def open_session(
        self,
        session_id: str,
        *,
        tokenizer: Any,
        identity: str | None = None,
        special_token_ids: dict[str, int] | None = None,
    ) -> VenusOmniSession:
        async with self._lock:
            if self._closed:
                raise RuntimeError("VenusOmni agent harness is closed")
            if session_id in self._sessions:
                raise ValueError(f"session is already open: {session_id}")
            await self.delegate_harness.open_session(
                session_id,
                backend_name=self.config.default_backend,
                identity=identity,
            )
            session = VenusOmniSession(
                owner=self,
                session_id=session_id,
                tokenizer=tokenizer,
                special_token_ids=special_token_ids,
            )
            self._sessions[session_id] = session
            return session

    def get_work(self, work_id: str) -> dict[str, Any] | None:
        return self.execution_backend.get_work(work_id)

    def list_works(self, *, active_only: bool = False) -> list[dict[str, object]]:
        return self.execution_backend.list_works(active_only=active_only)

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
            for session in sessions:
                session._closed = True
        await self.delegate_harness.aclose()

    async def _close_session(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
        await self.delegate_harness.close_session(session_id)
        self.execution_backend.close_session_history(session_id)


__all__ = ["VenusOmniAgentHarness", "VenusOmniSession"]
