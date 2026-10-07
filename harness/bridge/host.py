"""Production-side orchestration between VenusOmni serving and the Harness.

This module owns ordering, not model inference.  It is the only layer allowed
to join the two independent contracts:

* :class:`VenusOmniServingPort` owns one model/KV session and emits raw tokens;
* :class:`VenusOmniSession` owns delegate parsing, context and delivery state;
* :class:`PlaybackPort` owns the ordered native Talker audio-chunk queue.

The host deliberately keeps backend-result collection away from the model
port.  A result waiter only puts :class:`BackendInjection` objects in a local
queue; the serving/model owner must explicitly call :meth:`begin_backend_turn`
to apply a private prefill to the KV cache.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from harness.core.models import AudioChunk, VideoFrame, VideoSegment
from harness.core.venus import venus_special_token_ids

from .contracts import BackendInjection, ModelStep
from .runtime import VenusOmniAgentHarness, VenusOmniSession
from .serving import (
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
    SessionClosed,
    SessionOpened,
    VenusOmniServingPort,
    VideoFrameAppend,
    VideoSegmentAppend,
)


class HostProtocolError(RuntimeError):
    """The serving or playback implementation broke an ordering fence."""


class AudioDisposition(str, Enum):
    """What the host did with audio attached to one model output step."""

    NONE = "none"
    QUEUED = "queued"
    SUPPRESSED = "suppressed"


@dataclass(frozen=True, slots=True)
class PlaybackEnqueue:
    """One safe native Talker chunk admitted to ordered playback."""

    session_id: str
    incarnation: int
    utterance_id: str
    chunk_seq: int
    text: str
    audio: GeneratedAudio
    generation_id: str
    generation_epoch: int
    caused_by_work_id: str = ""
    attempt_id: str = ""

    def __post_init__(self) -> None:
        if not self.session_id or not self.utterance_id or not self.generation_id:
            raise ValueError("playback enqueue identifiers must not be blank")
        if self.incarnation < 1 or self.generation_epoch < 0 or self.chunk_seq < 1:
            raise ValueError("invalid playback session/generation fence")
        if not isinstance(self.audio, GeneratedAudio):
            raise TypeError("audio must be one native Talker GeneratedAudio chunk")
        if self.caused_by_work_id and not self.attempt_id:
            raise ValueError("backend playback requires attempt_id")
        if self.attempt_id and not self.caused_by_work_id:
            raise ValueError("attempt_id is only valid for backend playback")


@runtime_checkable
class PlaybackPort(Protocol):
    """Ordered sink for native VenusOmni Talker audio chunks."""

    async def enqueue(self, request: PlaybackEnqueue) -> None: ...


class FeedbackSuperseded(RuntimeError):
    """Safe to poll again: the queued progress was superseded before prefill."""


@dataclass(frozen=True, slots=True)
class BackendTurn:
    """One backend prefill attempt currently associated with model output."""

    work_id: str
    attempt_number: int
    attempt_id: str
    utterance_id: str
    generation_id: str
    generation_epoch: int
    feedback_id: str = ""


@dataclass(frozen=True, slots=True)
class ProcessedOutput:
    """Harness-safe projection of one serving output step."""

    output: ModelOutputStep
    model_step: ModelStep
    utterance_id: str
    caused_by_work_id: str
    audio_disposition: AudioDisposition
    tail: str = ""


@dataclass(slots=True)
class _GenerationState:
    generation_id: str
    generation_epoch: int
    utterance_id: str
    caused_by_work_id: str = ""
    attempt_id: str = ""
    feedback_id: str = ""
    last_step_seq: int = 0
    last_audio_chunk_seq: int = 0
    cumulative_ids: tuple[int, ...] = ()
    suppress_remaining_audio: bool = False


@dataclass(slots=True)
class _PlaybackState:
    utterance_id: str
    generation_id: str
    generation_epoch: int
    caused_by_work_id: str = ""
    attempt_id: str = ""
    total_chunks: int = 0
    played_chunks: int = 0
    generation_finished: bool = False


@dataclass(slots=True)
class _BackendDelivery:
    injection: BackendInjection
    attempt_number: int
    attempt_id: str
    utterance_id: str
    generation_id: str
    generation_epoch: int
    generation_finished: bool = False
    last_activity: float = field(default_factory=time.monotonic)

    def public(self) -> BackendTurn:
        return BackendTurn(
            work_id=self.injection.work_id,
            feedback_id=self.injection.feedback_id,
            attempt_number=self.attempt_number,
            attempt_id=self.attempt_id,
            utterance_id=self.utterance_id,
            generation_id=self.generation_id,
            generation_epoch=self.generation_epoch,
        )


@dataclass(slots=True)
class _PendingPrefill:
    injection: BackendInjection
    attempt_number: int
    attempt_id: str
    utterance_id: str
    request: PrefillAppend


@dataclass(frozen=True, slots=True)
class _CollectorFailure:
    error: BaseException


@dataclass(frozen=True, slots=True)
class _HostClosed:
    pass


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


class VenusOmniServingHost:
    """Serialize one serving session against one :class:`VenusOmniSession`.

    Construct directly only when both sessions have already been opened.  The
    :meth:`open` classmethod is preferred because it cross-checks the serving
    special-token mapping against the local tokenizer before the Harness sees
    any model output.
    """

    def __init__(
        self,
        *,
        session: VenusOmniSession,
        serving: VenusOmniServingPort,
        playback: PlaybackPort,
        opened: SessionOpened,
        mute_delegate_audio: bool = False,
        clock: Callable[[], int] = _now_ms,
    ) -> None:
        if session.session_id != opened.session_id:
            raise ValueError("VenusOmni and serving session IDs do not match")
        self.session = session
        self.serving = serving
        self.playback = playback
        self.opened = opened
        self._mute_delegate_audio = mute_delegate_audio
        self._clock = clock
        self._event_seq = 0
        self._last_input_seq = 0
        self._last_harness_sequence = 0
        self._last_output_input_cutoff = 0
        self._generation_epoch_floor = 0
        self._finished_generations: set[tuple[str, int]] = set()
        self._finished_generation_order: deque[tuple[str, int]] = deque()
        self._generation: _GenerationState | None = None
        self._backend_delivery: _BackendDelivery | None = None
        self._pending_prefill: _PendingPrefill | None = None
        self._deferred_injection: BackendInjection | None = None
        self._playback: dict[str, _PlaybackState] = {}
        self._attempts: dict[str, int] = {}
        self._backend_queue: asyncio.Queue[
            BackendInjection | _CollectorFailure | _HostClosed
        ] = asyncio.Queue(maxsize=16)
        self._collector_task: asyncio.Task[None] | None = None
        self._feedback_monitor_task = None
        self._owner_lock = asyncio.Lock()
        self._output_pull_lock = asyncio.Lock()
        self._begin_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._active_pull_task: asyncio.Task[Any] | None = None
        self._pending_pulled_output = False
        self._closing = False
        self._closed = False
        self._poisoned: BaseException | None = None

    @classmethod
    async def open(
        cls,
        *,
        agent: VenusOmniAgentHarness,
        serving: VenusOmniServingPort,
        playback: PlaybackPort,
        session_id: str,
        model: str,
        tokenizer: Any,
        identity: str | None = None,
        mute_delegate_audio: bool = False,
        clock: Callable[[], int] = _now_ms,
    ) -> VenusOmniServingHost:
        """Open serving first, validate tokens, then open the Harness session.

        ``mute_delegate_audio`` distrusts model-side TTS filtering: from the
        first private chunk through turn completion, no waveform is admitted.
        This also blocks buffered speech emitted after a closing delegate tag.
        """

        requested = OpenSession(session_id=session_id, model=model)
        opened = await serving.open_session(requested)
        try:
            if opened.session_id != requested.session_id:
                raise HostProtocolError("serving returned a different session_id")
            if opened.model != requested.model:
                raise HostProtocolError("serving returned a different model")
            local_token_ids = venus_special_token_ids(tokenizer)
            serving_token_ids = dict(opened.capabilities.special_token_ids)
            if any(
                serving_token_ids.get(token) != token_id
                for token, token_id in local_token_ids.items()
            ):
                raise HostProtocolError(
                    "serving special_token_ids do not match the local tokenizer"
                )
            session = await agent.open_session(
                session_id,
                tokenizer=tokenizer,
                identity=identity,
                special_token_ids=local_token_ids,
            )
        except BaseException:
            await _close_serving_after_failed_open(serving, opened)
            raise
        host = cls(
            session=session,
            serving=serving,
            playback=playback,
            opened=opened,
            mute_delegate_audio=mute_delegate_audio,
            clock=clock,
        )
        host.start_backend_collector()
        return host

    @property
    def active_backend_turn(self) -> BackendTurn | None:
        delivery = self._backend_delivery
        return delivery.public() if delivery is not None else None

    @property
    def queued_backend_results(self) -> int:
        return (
            self._backend_queue.qsize()
            + int(self._deferred_injection is not None)
            + int(self._pending_prefill is not None)
        )

    def start_backend_collector(self) -> None:
        """Start the result-only waiter; it never invokes the serving port."""

        self._ensure_usable()
        if self._collector_task is None:
            self._collector_task = asyncio.create_task(
                self._collect_backend_results(),
                name=f"venus-backend-results:{self.session.session_id}",
            )

    async def attach_file(
        self, name: str, data: bytes, mime_type: str = "application/octet-stream"
    ) -> None:
        """Attach session files before media starts, without changing Serving's media sequence."""
        async with self._owner_lock:
            self._ensure_usable()
            if (
                self._last_input_seq
                or self._generation is not None
                or self._model_owner_busy()
            ):
                raise RuntimeError(
                    "attach files before beginning the serving media session"
                )
            await self.session.ingest_user_file(name, data, mime_type)

    async def append_text(self, text: str) -> InputAccepted:
        async with self._owner_lock:
            self._ensure_usable()
            append = getattr(self.serving, "append_text", None)
            if append is None:
                raise RuntimeError("serving does not support typed messages")
            request = TextAppend(self.opened.session_id, self.opened.incarnation,
                                 self._next_event_seq(), text)
            sequence = await self.session.ingest_user_text(text)
            return await self._append_serving_input(request, append, harness_sequence=sequence)

    async def append_audio(self, chunk: AudioChunk) -> InputAccepted:
        """Buffer audio in Harness first, then fence the same bytes in serving."""

        audio_format = _audio_format(chunk.mime_type)
        async with self._owner_lock:
            self._ensure_usable()
            event_seq = self._next_event_seq()
            request = AudioAppend(
                session_id=self.opened.session_id,
                incarnation=self.opened.incarnation,
                event_seq=event_seq,
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
                data=bytes(chunk.data),
                format=audio_format,
                sample_rate_hz=chunk.sample_rate_hz,
                channels=chunk.channels,
                sample_width_bytes=chunk.sample_width_bytes,
            )
            harness_sequence = await self.session.ingest_user_audio(chunk)
            return await self._append_serving_input(
                request,
                self.serving.append_audio,
                harness_sequence=harness_sequence,
            )

    async def append_video_frame(self, frame: VideoFrame) -> InputAccepted:
        """Buffer one frame in Harness before making it visible to the model."""

        async with self._owner_lock:
            self._ensure_usable()
            event_seq = self._next_event_seq()
            request = VideoFrameAppend(
                session_id=self.opened.session_id,
                incarnation=self.opened.incarnation,
                event_seq=event_seq,
                captured_at_ms=frame.captured_at_ms,
                data=bytes(frame.data),
                mime_type=frame.mime_type,
            )
            harness_sequence = await self.session.ingest_user_video_frame(frame)
            return await self._append_serving_input(
                request,
                self.serving.append_video_frame,
                harness_sequence=harness_sequence,
            )

    async def append_video_segment(self, segment: VideoSegment) -> InputAccepted:
        """Buffer an encoded segment, if serving explicitly supports segments."""

        if not self.opened.capabilities.video_segments:
            raise ValueError("serving session does not support video segments")
        async with self._owner_lock:
            self._ensure_usable()
            event_seq = self._next_event_seq()
            request = VideoSegmentAppend(
                session_id=self.opened.session_id,
                incarnation=self.opened.incarnation,
                event_seq=event_seq,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                data=bytes(segment.data),
                fps=segment.fps,
                mime_type=segment.mime_type,
            )
            harness_sequence = await self.session.ingest_user_video_segment(segment)
            return await self._append_serving_input(
                request,
                self.serving.append_video_segment,
                harness_sequence=harness_sequence,
            )

    async def next_output(self, *, timeout_s: float | None = None) -> ProcessedOutput:
        """Pull, validate and safely expose exactly one model output step."""

        # Do not hold the mutation lane while waiting for a potentially slow
        # output stream. Media append may advance concurrently. Private
        # prefill fails fast while this pull is active instead of racing a
        # returned output across a generation fence.
        async with self._output_pull_lock:
            task = asyncio.current_task()
            if task is None:
                raise RuntimeError("next_output must run in an asyncio task")
            self._active_pull_task = task
            try:
                self._ensure_usable()
                output = await self.serving.next_output_step(
                    self.opened.session_id,
                    incarnation=self.opened.incarnation,
                    timeout_s=timeout_s,
                )
                self._pending_pulled_output = True
                async with self._owner_lock:
                    self._ensure_usable()
                    return await self._process_output_fail_closed(output)
            finally:
                self._pending_pulled_output = False
                if self._active_pull_task is task:
                    self._active_pull_task = None

    async def process_output(self, output: ModelOutputStep) -> ProcessedOutput:
        """Process a pushed output DTO using the same owner/fence checks."""

        if self._model_owner_busy():
            raise RuntimeError("another model output pull is active")
        self._pending_pulled_output = True
        try:
            async with self._owner_lock:
                self._ensure_usable()
                return await self._process_output_fail_closed(output)
        finally:
            self._pending_pulled_output = False

    async def begin_backend_turn(
        self, *, timeout_s: float | None = None
    ) -> BackendTurn:
        """Skip superseded progress internally while preserving the caller's deadline."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            remaining = (
                None if deadline is None else max(0, deadline - time.monotonic())
            )
            try:
                return await self._begin_backend_turn_once(timeout_s=remaining)
            except FeedbackSuperseded:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError from None
                continue

    async def _begin_backend_turn_once(
        self,
        *,
        timeout_s: float | None = None,
    ) -> BackendTurn:
        """Apply one queued result from the model/GPU owner coroutine."""

        async with self._begin_lock:
            self._ensure_usable()
            if self._model_owner_busy():
                raise RuntimeError("cannot prefill while a model output pull is active")
            if self._generation is not None:
                raise RuntimeError(
                    "cannot prefill while another model generation is active"
                )
            if self._backend_delivery is not None:
                raise RuntimeError("the previous backend delivery is still active")
            pending = self._pending_prefill
            injection: BackendInjection | None = None
            if pending is None:
                if self._deferred_injection is not None:
                    injection = self._deferred_injection
                    self._deferred_injection = None
                else:
                    item = await _queue_get(self._backend_queue, timeout_s)
                    if isinstance(item, _HostClosed):
                        raise RuntimeError(
                            "VenusOmni serving host closed while waiting for result"
                        )
                    if isinstance(item, _CollectorFailure):
                        raise RuntimeError(
                            "backend result collector failed"
                        ) from item.error
                    injection = item
            async with self._owner_lock:
                self._ensure_usable()
                # State may have changed while this coroutine waited for a
                # result.  Never overwrite an in-flight foreground/backend
                # generation; preserve the dequeued result for a later owner
                # iteration.
                if (
                    self._model_owner_busy()
                    or self._generation is not None
                    or self._backend_delivery is not None
                ):
                    if injection is not None:
                        if self._deferred_injection is not None:
                            self._fail_protocol("multiple deferred backend injections")
                        self._deferred_injection = injection
                    raise RuntimeError(
                        "model became busy while waiting for backend result"
                    )
                if pending is None:
                    assert injection is not None
                    if (
                        injection.result.feedback_id
                        and not self.session.injection_is_current(injection)
                    ):
                        await self.session.discard_injection(injection)
                        raise FeedbackSuperseded(
                            "queued feedback is no longer relevant"
                        )
                    attempt_number = self._attempts.get(injection.feedback_id, 0) + 1
                    self._attempts[injection.feedback_id] = attempt_number
                    unique = uuid4().hex
                    attempt_id = (
                        f"{injection.feedback_id}:attempt:{attempt_number}:{unique}"
                    )
                    utterance_id = f"backend:{injection.feedback_id}:attempt:{attempt_number}:{unique}"
                    request = PrefillAppend(
                        session_id=self.opened.session_id,
                        incarnation=self.opened.incarnation,
                        work_id=injection.work_id,
                        attempt_id=attempt_id,
                        text_list=(injection.text,),
                    )
                    pending = _PendingPrefill(
                        injection=injection,
                        attempt_number=attempt_number,
                        attempt_id=attempt_id,
                        utterance_id=utterance_id,
                        request=request,
                    )
                    self._pending_prefill = pending
                try:
                    applied = await asyncio.wait_for(
                        self.serving.append_prefill(pending.request),
                        getattr(self.session, "feedback_timeout_s", 60),
                    )
                except PrefillDeferred:
                    # The serving owner guarantees that nothing was applied.
                    # Keep the exact attempt while the caller drains output.
                    raise TimeoutError from None
                except Exception:  # noqa: BLE001 - one idempotent ambiguity retry
                    # The first response may have been lost after the remote
                    # side applied it.  Retry the exact same idempotency key,
                    # never allocate a new attempt for an ambiguous failure.
                    try:
                        applied = await asyncio.wait_for(
                            self.serving.append_prefill(pending.request),
                            getattr(self.session, "feedback_timeout_s", 60),
                        )
                    except PrefillDeferred:
                        raise TimeoutError from None
                    except Exception as exc:
                        self._poisoned = exc
                        if pending.injection.result.feedback_id:
                            self.session.feedback_failed(pending.injection)
                        raise RuntimeError(
                            "backend prefill failed twice with the same attempt_id"
                        ) from exc
                self._validate_prefill_applied(
                    applied,
                    pending.injection.work_id,
                    pending.attempt_id,
                )
                key = (applied.generation_id, applied.generation_epoch)
                if key in self._finished_generations:
                    self._fail_protocol(
                        "prefill resumed an already-finished generation"
                    )
                if applied.generation_epoch < self._generation_epoch_floor:
                    self._fail_protocol(
                        "prefill generation predates a finished generation"
                    )
                self._generation = _GenerationState(
                    generation_id=applied.generation_id,
                    generation_epoch=applied.generation_epoch,
                    utterance_id=pending.utterance_id,
                    caused_by_work_id=pending.injection.work_id,
                    attempt_id=pending.attempt_id,
                    feedback_id=pending.injection.feedback_id,
                )
                self._playback[pending.utterance_id] = _PlaybackState(
                    utterance_id=pending.utterance_id,
                    generation_id=applied.generation_id,
                    generation_epoch=applied.generation_epoch,
                    caused_by_work_id=pending.injection.work_id,
                    attempt_id=pending.attempt_id,
                )
                self._backend_delivery = _BackendDelivery(
                    injection=pending.injection,
                    attempt_number=pending.attempt_number,
                    attempt_id=pending.attempt_id,
                    utterance_id=pending.utterance_id,
                    generation_id=applied.generation_id,
                    generation_epoch=applied.generation_epoch,
                )
                self._pending_prefill = None
                if (
                    pending.injection.result.feedback_id
                    and self._feedback_monitor_task is None
                ):
                    self._feedback_monitor_task = asyncio.create_task(
                        self._monitor_feedback()
                    )
                return self._backend_delivery.public()

    async def acknowledge_playback(
        self,
        utterance_id: str,
        cumulative_played_chunks: int,
        *,
        at_ms: int | None = None,
    ) -> str:
        """Commit cumulative native Talker chunks actually played."""

        async with self._owner_lock:
            self._ensure_usable()
            return await self._acknowledge_playback_locked(
                utterance_id,
                cumulative_played_chunks,
                self._clock() if at_ms is None else at_ms,
            )

    async def close(self, *, reason: str = "client_close") -> None:
        """Stop the waiter and close both sides even if one close fails."""

        async with self._close_lock:
            # Wait only for an already-started mutation, then fence every new
            # one.  Never wait for _output_pull_lock: next_output(timeout=None)
            # may be blocked until close_session tears down the remote stream.
            async with self._owner_lock:
                if self._closed or self._closing:
                    return
                self._closing = True
            if self._feedback_monitor_task:
                self._feedback_monitor_task.cancel()
                await asyncio.gather(
                    self._feedback_monitor_task, return_exceptions=True
                )
            if (
                self._backend_delivery
                and self._backend_delivery.injection.result.feedback_id
            ):
                self.session.feedback_failed(self._backend_delivery.injection)
            if self._backend_queue.full():
                self._backend_queue.get_nowait()
            self._backend_queue.put_nowait(_HostClosed())
            collector = self._collector_task
            if collector is not None:
                collector.cancel()
                await asyncio.gather(collector, return_exceptions=True)
            pull = self._active_pull_task
            current = asyncio.current_task()
            if pull is not None and pull is not current:
                pull.cancel()
            errors: list[BaseException] = []
            try:
                closed = await self.serving.close_session(
                    CloseSession(
                        session_id=self.opened.session_id,
                        incarnation=self.opened.incarnation,
                        reason=reason,
                    )
                )
                self._validate_closed(closed)
            except Exception as exc:  # noqa: BLE001 - both close attempts must run
                errors.append(exc)
            if pull is not None and pull is not current:
                await asyncio.gather(pull, return_exceptions=True)
            async with self._owner_lock:
                try:
                    await self.session.close()
                except Exception as exc:  # noqa: BLE001 - report after both closes
                    errors.append(exc)
                self._closed = True
                self._closing = False
            if errors:
                raise RuntimeError(
                    "failed to close VenusOmni host cleanly"
                ) from errors[0]

    def _feedback_fault(self):
        delivery = self._backend_delivery
        if delivery and delivery.injection.result.feedback_id:
            self.session.feedback_failed(delivery.injection)

    async def _monitor_feedback(self):
        timeout = self.session.feedback_timeout_s
        while not self._closed and not self._closing:
            await asyncio.sleep(min(timeout / 4, 1))
            delivery = self._backend_delivery
            if delivery and time.monotonic() - delivery.last_activity >= timeout:
                self._feedback_fault()
                self._poisoned = TimeoutError(
                    "frontend feedback/ACK inactivity timeout"
                )
                return

    async def _collect_backend_results(self) -> None:
        try:
            while not self._closed:
                injection = await self.session.next_backend_injection()
                await self._backend_queue.put(injection)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface through the queue
            if not self._closed:
                await self._backend_queue.put(_CollectorFailure(exc))

    async def _append_serving_input(
        self,
        request: TextAppend | AudioAppend | VideoFrameAppend | VideoSegmentAppend,
        append: Callable[[Any], Any],
        *,
        harness_sequence: int,
    ) -> InputAccepted:
        try:
            if isinstance(harness_sequence, bool) or not isinstance(
                harness_sequence, int
            ):
                self._fail_protocol("Harness ingest did not return an integer sequence")
            if harness_sequence != self._last_harness_sequence + 1:
                self._fail_protocol("Harness media sequence must be contiguous")
            accepted = await append(request)
            self._validate_input_accepted(accepted, request.event_seq)
        except BaseException as exc:
            self._poisoned = exc
            self._feedback_fault()
            raise
        self._last_harness_sequence = harness_sequence
        self._last_input_seq = accepted.input_seq
        return accepted

    async def _process_output_locked(self, output: ModelOutputStep) -> ProcessedOutput:
        self._validate_output_fence(output)
        if self._backend_delivery:
            self._backend_delivery.last_activity = time.monotonic()
        generation = self._select_generation(output)
        token_ids = output.total_token_ids
        if self.opened.capabilities.raw_token_mode is RawTokenMode.DELTA:
            generation.cumulative_ids += token_ids
            token_ids = generation.cumulative_ids
        else:
            if (
                output.total_token_ids
                and generation.cumulative_ids
                and output.total_token_ids[: len(generation.cumulative_ids)]
                != generation.cumulative_ids
            ):
                self._fail_protocol(
                    "cumulative token buffer is not append-only within generation"
                )
            generation.cumulative_ids = token_ids

        # A terminal empty step only announces turn completion.  Feeding the
        # empty buffer before finish would erase a parser-held visible tail.
        if output.total_token_ids:
            model_step = await self.session.consume_total_ids(
                token_ids,
                at_ms=output.at_ms,
                cutoff_sequence=output.input_seq_cutoff,
                delegate_audio_filtered=(
                    self.opened.capabilities.delegate_audio_filtered
                    and not self._mute_delegate_audio
                ),
            )
        else:
            model_step = ModelStep("", (), False, 0)

        if self._mute_delegate_audio:
            # TTS can flush private syllables after the closing token, or on a
            # later step without text. Keep the gate closed until a fresh turn.
            generation.suppress_remaining_audio |= model_step.suppress_audio
            if generation.suppress_remaining_audio:
                model_step = replace(model_step, suppress_audio=True)

        playback_state = self._playback[generation.utterance_id]
        disposition = AudioDisposition.NONE
        safe_audio = output.audio
        if output.audio is not None:
            assert output.audio_chunk_seq is not None
            if output.audio_chunk_seq != generation.last_audio_chunk_seq + 1:
                self._fail_protocol(
                    "Talker audio_chunk_seq must be contiguous within a generation"
                )
            generation.last_audio_chunk_seq = output.audio_chunk_seq
        if model_step.suppress_audio:
            safe_audio = None
            if output.audio is not None:
                disposition = AudioDisposition.SUPPRESSED
        elif output.audio is not None:
            disposition = AudioDisposition.QUEUED

        if safe_audio is not None:
            playback_chunk_seq = playback_state.total_chunks + 1
            await self.playback.enqueue(
                PlaybackEnqueue(
                    session_id=self.opened.session_id,
                    incarnation=self.opened.incarnation,
                    utterance_id=generation.utterance_id,
                    chunk_seq=playback_chunk_seq,
                    text=model_step.visible_text,
                    audio=safe_audio,
                    generation_id=generation.generation_id,
                    generation_epoch=generation.generation_epoch,
                    caused_by_work_id=generation.caused_by_work_id,
                    attempt_id=generation.attempt_id,
                )
            )
            await self.session.append_playback_chunk(
                generation.utterance_id,
                playback_chunk_seq,
                model_step.visible_text,
                caused_by_work_id=generation.caused_by_work_id,
                **(
                    {"caused_by_feedback_id": generation.feedback_id}
                    if generation.feedback_id
                    and generation.feedback_id != generation.caused_by_work_id
                    else {}
                ),
            )
            playback_state.total_chunks += 1

        tail = ""
        if output.turn_finished:
            tail = await self._finish_generation_locked(output, generation)
        return ProcessedOutput(
            output=output,
            model_step=model_step,
            utterance_id=generation.utterance_id,
            caused_by_work_id=generation.caused_by_work_id,
            audio_disposition=disposition,
            tail=tail,
        )

    async def _process_output_fail_closed(
        self,
        output: ModelOutputStep,
    ) -> ProcessedOutput:
        try:
            return await self._process_output_locked(output)
        except BaseException as exc:
            # Once a serving step has been observed, token cursor, playback queue or
            # Harness state may already have advanced.  Retrying the same step
            # is unsafe, so every failure after observation poisons this host.
            self._poisoned = exc
            self._feedback_fault()
            raise

    def _select_generation(self, output: ModelOutputStep) -> _GenerationState:
        current = self._generation
        key = (output.generation_id, output.generation_epoch)
        if current is None:
            if key in self._finished_generations:
                self._fail_protocol("serving emitted output for a finished generation")
            if output.generation_epoch < self._generation_epoch_floor:
                self._fail_protocol(
                    "serving emitted output before the generation epoch floor"
                )
            delivery = self._backend_delivery
            if delivery is not None and not delivery.generation_finished:
                if key != (delivery.generation_id, delivery.generation_epoch):
                    self._fail_protocol(
                        "backend prefill output generation does not match ACK"
                    )
                utterance_id = delivery.utterance_id
                work_id = delivery.injection.work_id
                attempt_id = delivery.attempt_id
            else:
                unique = uuid4().hex
                utterance_id = f"foreground:{self.opened.session_id}:{output.generation_id}:{unique}"
                work_id = ""
                attempt_id = ""
                self._playback[utterance_id] = _PlaybackState(
                    utterance_id=utterance_id,
                    generation_id=output.generation_id,
                    generation_epoch=output.generation_epoch,
                )
            current = _GenerationState(
                generation_id=output.generation_id,
                generation_epoch=output.generation_epoch,
                utterance_id=utterance_id,
                caused_by_work_id=work_id,
                attempt_id=attempt_id,
                feedback_id=delivery.injection.feedback_id if work_id else "",
            )
            self._generation = current
        elif key != (current.generation_id, current.generation_epoch):
            self._fail_protocol("serving changed generation before turn_finished")
        if output.step_seq != current.last_step_seq + 1:
            self._fail_protocol(
                "serving step_seq must be contiguous within a generation"
            )
        current.last_step_seq = output.step_seq
        return current

    async def _finish_generation_locked(
        self,
        output: ModelOutputStep,
        generation: _GenerationState,
    ) -> str:
        tail = await self.session.finish_model_turn(
            at_ms=output.at_ms,
            caused_by_work_id=generation.caused_by_work_id,
            **(
                {"caused_by_feedback_id": generation.feedback_id}
                if generation.feedback_id
                and generation.feedback_id != generation.caused_by_work_id
                else {}
            ),
            next_total_ids=(),
        )
        playback_state = self._playback[generation.utterance_id]
        if generation.caused_by_work_id and playback_state.total_chunks == 0:
            self._fail_protocol(
                "backend generation finished without any Talker audio chunks"
            )
        playback_state.generation_finished = True
        self._remember_finished_generation(generation)
        delivery = self._backend_delivery
        if (
            delivery is not None
            and delivery.injection.work_id == generation.caused_by_work_id
        ):
            delivery.generation_finished = True
        self._generation = None
        self._cleanup_playback_if_complete(playback_state)
        return tail

    async def _acknowledge_playback_locked(
        self,
        utterance_id: str,
        cumulative_played_chunks: int,
        at_ms: int,
    ) -> str:
        state = self._playback.get(utterance_id)
        if state is None:
            raise ValueError(f"unknown or completed utterance: {utterance_id}")
        if cumulative_played_chunks < state.played_chunks:
            raise ValueError(
                "playback acknowledgement must be cumulative and monotonic"
            )
        if cumulative_played_chunks > state.total_chunks:
            raise ValueError("playback acknowledgement exceeds queued Talker chunks")
        try:
            accepted = await self.serving.acknowledge_playback(
                PlaybackAck(
                    session_id=self.opened.session_id,
                    incarnation=self.opened.incarnation,
                    utterance_id=utterance_id,
                    cumulative_played_chunks=cumulative_played_chunks,
                    at_ms=at_ms,
                    caused_by_work_id=state.caused_by_work_id,
                )
            )
            self._validate_playback_accepted(
                accepted,
                utterance_id,
                cumulative_played_chunks,
            )
            newly_played = await self.session.acknowledge_playback_chunks(
                utterance_id,
                cumulative_played_chunks,
                at_ms=at_ms,
            )
        except BaseException as exc:
            # The remote and local playback ledgers must advance together.
            # Once either side was contacted, continuing after an error could
            # incorrectly mark a Work delivered, so fail closed.
            self._poisoned = exc
            self._feedback_fault()
            raise
        if self._backend_delivery and cumulative_played_chunks > state.played_chunks:
            self._backend_delivery.last_activity = time.monotonic()
        state.played_chunks = cumulative_played_chunks
        self._cleanup_playback_if_complete(state)
        return newly_played

    def _cleanup_playback_if_complete(self, state: _PlaybackState) -> None:
        if not state.generation_finished:
            return
        if state.total_chunks > 0 and state.played_chunks < state.total_chunks:
            return
        self._playback.pop(state.utterance_id, None)
        delivery = self._backend_delivery
        if (
            delivery is not None
            and delivery.utterance_id == state.utterance_id
            and state.total_chunks > 0
        ):
            self._attempts.pop(delivery.injection.feedback_id, None)
            self._backend_delivery = None

    def _remember_finished_generation(self, generation: _GenerationState) -> None:
        self._generation_epoch_floor = max(
            self._generation_epoch_floor,
            generation.generation_epoch + 1,
        )
        key = (generation.generation_id, generation.generation_epoch)
        if key in self._finished_generations:
            return
        self._finished_generations.add(key)
        self._finished_generation_order.append(key)
        while len(self._finished_generation_order) > 256:
            expired = self._finished_generation_order.popleft()
            self._finished_generations.discard(expired)

    def _model_owner_busy(self) -> bool:
        """Whether a pulled output owns or may next mutate generation state."""

        return self._active_pull_task is not None or self._pending_pulled_output

    def _validate_input_accepted(self, accepted: InputAccepted, event_seq: int) -> None:
        if accepted.session_id != self.opened.session_id:
            self._fail_protocol("input ACK has wrong session_id")
        if accepted.incarnation != self.opened.incarnation:
            self._fail_protocol("input ACK has wrong incarnation")
        if accepted.event_seq != event_seq:
            self._fail_protocol("input ACK has wrong event_seq")
        expected_input_seq = self._last_input_seq + 1
        if accepted.input_seq != expected_input_seq:
            self._fail_protocol("serving input_seq must start at 1 and be contiguous")
        if accepted.input_seq != self._last_harness_sequence + 1:
            self._fail_protocol(
                "serving input_seq must match the Harness media sequence"
            )

    def _validate_output_fence(self, output: ModelOutputStep) -> None:
        if output.session_id != self.opened.session_id:
            self._fail_protocol("model output has wrong session_id")
        if output.incarnation != self.opened.incarnation:
            self._fail_protocol("model output has wrong incarnation")
        if output.input_seq_cutoff > self._last_input_seq:
            self._fail_protocol("model output refers to an unknown future input_seq")
        if output.input_seq_cutoff < self._last_output_input_cutoff:
            self._fail_protocol("model output input_seq_cutoff moved backwards")
        self._last_output_input_cutoff = output.input_seq_cutoff

    def _validate_prefill_applied(
        self,
        applied: PrefillApplied,
        work_id: str,
        attempt_id: str,
    ) -> None:
        if applied.session_id != self.opened.session_id:
            self._fail_protocol("prefill ACK has wrong session_id")
        if applied.incarnation != self.opened.incarnation:
            self._fail_protocol("prefill ACK has wrong incarnation")
        if applied.work_id != work_id or applied.attempt_id != attempt_id:
            self._fail_protocol("prefill ACK does not match request/attempt")

    def _validate_playback_accepted(
        self,
        accepted: PlaybackAccepted,
        utterance_id: str,
        cumulative_played_chunks: int,
    ) -> None:
        if accepted.session_id != self.opened.session_id:
            self._fail_protocol("playback ACK has wrong session_id")
        if accepted.incarnation != self.opened.incarnation:
            self._fail_protocol("playback ACK has wrong incarnation")
        if accepted.utterance_id != utterance_id:
            self._fail_protocol("playback ACK has wrong utterance_id")
        if accepted.cumulative_played_chunks != cumulative_played_chunks:
            self._fail_protocol("playback ACK changed cumulative chunk count")

    def _validate_closed(self, closed: SessionClosed) -> None:
        if closed.session_id != self.opened.session_id:
            raise HostProtocolError("close ACK has wrong session_id")
        if closed.incarnation != self.opened.incarnation:
            raise HostProtocolError("close ACK has wrong incarnation")

    def _next_event_seq(self) -> int:
        self._event_seq += 1
        return self._event_seq

    def _fail_protocol(self, message: str) -> None:
        error = HostProtocolError(message)
        self._poisoned = error
        self._feedback_fault()
        raise error

    def _ensure_usable(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("VenusOmni serving host is closed")
        if self._poisoned is not None:
            raise RuntimeError(
                "VenusOmni serving host is in a failed state"
            ) from self._poisoned


async def _queue_get(
    queue: asyncio.Queue[BackendInjection | _CollectorFailure | _HostClosed],
    timeout_s: float | None,
) -> BackendInjection | _CollectorFailure | _HostClosed:
    if timeout_s is None:
        return await queue.get()
    if timeout_s == 0:
        try:
            return queue.get_nowait()
        except asyncio.QueueEmpty:
            raise TimeoutError from None
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout_s)
    except TimeoutError:
        raise TimeoutError from None


async def _close_serving_after_failed_open(
    serving: VenusOmniServingPort,
    opened: SessionOpened,
) -> None:
    try:
        await serving.close_session(
            CloseSession(
                session_id=opened.session_id,
                incarnation=opened.incarnation,
                reason="host_open_failed",
            )
        )
    except Exception:  # noqa: BLE001 - preserve the original open failure
        return


def _audio_format(mime_type: str) -> AudioFormat:
    normalized = mime_type.lower().strip()
    if normalized in {"audio/pcm", "audio/pcm16", "audio/s16le"}:
        return AudioFormat.PCM_S16LE
    if normalized in {"audio/wav", "audio/x-wav", "audio/wave"}:
        return AudioFormat.WAV
    raise ValueError(f"unsupported VenusOmni audio mime_type: {mime_type}")


__all__ = [
    "AudioDisposition",
    "BackendTurn",
    "HostProtocolError",
    "PlaybackEnqueue",
    "PlaybackPort",
    "ProcessedOutput",
    "VenusOmniServingHost",
]
