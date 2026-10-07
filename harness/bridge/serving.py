"""Stable transport-neutral contract for a Realtime-Venus-Omni serving host.

The contract deliberately exposes model/session mechanics instead of an
OpenAI-style chat abstraction.  A conforming port keeps one resumable model
session, returns unfiltered token IDs before audio becomes irreversible, and
can append a private backend result to the same KV cache.

Transport implementations may project these values onto WebSocket, gRPC, or
an in-process model loop.  They must preserve the ordering and idempotency
semantics documented by :class:`VenusOmniServingPort`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from harness.core.venus import VENUS_PROTOCOL_TOKENS

PROTOCOL_VERSION = "realtime-venus-harness/2"


class PrefillDeferred(RuntimeError):
    """Prefill was not applied; drain pending model output before retrying."""


class RawTokenMode(str, Enum):
    """Shape of token IDs emitted by the serving implementation."""

    CUMULATIVE = "cumulative"
    DELTA = "delta"


class AudioFormat(str, Enum):
    """Audio encodings admitted at the serving boundary."""

    PCM_S16LE = "pcm_s16le"
    WAV = "wav"


class PrefillVisibility(str, Enum):
    """Visibility of text appended directly to model context."""

    PRIVATE = "private"


@dataclass(frozen=True, slots=True)
class OpenSession:
    """Request a fresh KV-owning VenusOmni session."""

    session_id: str
    model: str
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        _require_text(self.model, "model")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol_version: {self.protocol_version!r}")


@dataclass(frozen=True, slots=True)
class SessionCapabilities:
    """Capabilities proven by a successfully opened serving session.

    All non-optional booleans are hard requirements for the current Harness.
    An implementation that cannot provide one of them must reject session
    creation instead of returning a partially compatible session.
    """

    special_token_ids: Mapping[str, int]
    raw_token_mode: RawTokenMode = RawTokenMode.CUMULATIVE
    same_kv_prefill: bool = True
    duplex_media: bool = True
    ordered_input_fence: bool = True
    native_audio_output: bool = True
    video_segments: bool = False
    max_sessions: int = 1
    # Optional: the native Talker excludes private spans before synthesis.
    delegate_audio_filtered: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_token_mode", RawTokenMode(self.raw_token_mode))
        for name in (
            "same_kv_prefill",
            "duplex_media",
            "ordered_input_fence",
            "native_audio_output",
            "video_segments",
            "delegate_audio_filtered",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        hard_flags = {
            "same_kv_prefill": self.same_kv_prefill,
            "duplex_media": self.duplex_media,
            "ordered_input_fence": self.ordered_input_fence,
            "native_audio_output": self.native_audio_output,
        }
        missing_capabilities = [
            name for name, value in hard_flags.items() if value is not True
        ]
        if missing_capabilities:
            raise ValueError(
                "serving session is missing hard capabilities: "
                + ", ".join(missing_capabilities)
            )
        _require_positive_int(self.max_sessions, "max_sessions")
        object.__setattr__(
            self,
            "special_token_ids",
            _validated_special_token_ids(self.special_token_ids),
        )


@dataclass(frozen=True, slots=True)
class SessionOpened:
    """Successful session handshake and its model/tokenizer identity."""

    session_id: str
    incarnation: int
    model: str
    model_revision: str
    capabilities: SessionCapabilities
    opened_at_ms: int

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        _require_positive_int(self.incarnation, "incarnation")
        _require_text(self.model, "model")
        _require_text(self.model_revision, "model_revision")
        if not isinstance(self.capabilities, SessionCapabilities):
            raise TypeError("capabilities must be SessionCapabilities")
        _require_nonnegative_int(self.opened_at_ms, "opened_at_ms")


@dataclass(frozen=True, slots=True)
class AudioAppend:
    """One timestamped microphone chunk admitted before model observation."""

    session_id: str
    incarnation: int
    event_seq: int
    start_ms: int
    end_ms: int
    data: bytes
    format: AudioFormat = AudioFormat.PCM_S16LE
    sample_rate_hz: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2

    def __post_init__(self) -> None:
        _validate_session_event(self.session_id, self.incarnation, self.event_seq)
        _require_nonnegative_int(self.start_ms, "start_ms")
        _require_nonnegative_int(self.end_ms, "end_ms")
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        audio_format = AudioFormat(self.format)
        object.__setattr__(self, "format", audio_format)
        data = _require_bytes(self.data, "data")
        object.__setattr__(self, "data", data)
        if self.sample_rate_hz != 16_000:
            raise ValueError("VenusOmni input audio must use 16000 Hz")
        if self.channels != 1:
            raise ValueError("VenusOmni input audio must be mono")
        if self.sample_width_bytes != 2:
            raise ValueError("VenusOmni input audio must use 16-bit samples")
        if audio_format is AudioFormat.PCM_S16LE and len(data) % 2:
            raise ValueError("pcm_s16le data must contain complete 16-bit samples")


@dataclass(frozen=True, slots=True)
class TextAppend:
    """A typed user message in the session-wide input order."""

    session_id: str
    incarnation: int
    event_seq: int
    text: str

    def __post_init__(self) -> None:
        _validate_session_event(self.session_id, self.incarnation, self.event_seq)
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 4000:
            raise ValueError("text must contain 1 to 4000 characters")


@dataclass(frozen=True, slots=True)
class VideoFrameAppend:
    """One encoded camera frame in the session-wide input order."""

    session_id: str
    incarnation: int
    event_seq: int
    captured_at_ms: int
    data: bytes
    mime_type: str = "image/jpeg"

    def __post_init__(self) -> None:
        _validate_session_event(self.session_id, self.incarnation, self.event_seq)
        _require_nonnegative_int(self.captured_at_ms, "captured_at_ms")
        object.__setattr__(self, "data", _require_bytes(self.data, "data"))
        if self.mime_type not in {"image/jpeg", "image/png"}:
            raise ValueError("video frame mime_type must be image/jpeg or image/png")


@dataclass(frozen=True, slots=True)
class VideoSegmentAppend:
    """Optional, already encoded MP4 segment input."""

    session_id: str
    incarnation: int
    event_seq: int
    start_ms: int
    end_ms: int
    data: bytes
    fps: float
    mime_type: str = "video/mp4"

    def __post_init__(self) -> None:
        _validate_session_event(self.session_id, self.incarnation, self.event_seq)
        _require_nonnegative_int(self.start_ms, "start_ms")
        _require_nonnegative_int(self.end_ms, "end_ms")
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        object.__setattr__(self, "data", _require_bytes(self.data, "data"))
        if self.mime_type != "video/mp4":
            raise ValueError("video segment mime_type must be video/mp4")
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise TypeError("fps must be a finite number")
        if not math.isfinite(float(self.fps)) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")


@dataclass(frozen=True, slots=True)
class InputAccepted:
    """Fence proving that an input event entered the ordered model stream.

    For one session incarnation, ``input_seq`` starts at 1 and advances by
    exactly one for every accepted audio/frame/segment event.  This keeps the
    serving fence and Harness media sequence deterministically mappable
    without an unbounded per-event translation table.
    """

    session_id: str
    incarnation: int
    event_seq: int
    input_seq: int
    accepted_at_ms: int

    def __post_init__(self) -> None:
        _validate_session_event(self.session_id, self.incarnation, self.event_seq)
        _require_positive_int(self.input_seq, "input_seq")
        _require_nonnegative_int(self.accepted_at_ms, "accepted_at_ms")


@dataclass(frozen=True, slots=True)
class GeneratedAudio:
    """One unplayed native Talker waveform chunk."""

    data: bytes
    sample_rate_hz: int = 24_000
    channels: int = 1
    sample_width_bytes: int = 2
    format: AudioFormat = AudioFormat.PCM_S16LE

    def __post_init__(self) -> None:
        audio_format = AudioFormat(self.format)
        object.__setattr__(self, "format", audio_format)
        data = _require_bytes(self.data, "data")
        object.__setattr__(self, "data", data)
        _require_positive_int(self.sample_rate_hz, "sample_rate_hz")
        _require_positive_int(self.channels, "channels")
        _require_positive_int(self.sample_width_bytes, "sample_width_bytes")
        if audio_format is AudioFormat.PCM_S16LE:
            if self.sample_width_bytes != 2:
                raise ValueError("pcm_s16le output must use 16-bit samples")
            frame_size = self.channels * self.sample_width_bytes
            if len(data) % frame_size:
                raise ValueError("pcm_s16le output must contain complete audio frames")


@dataclass(frozen=True, slots=True)
class ModelOutputStep:
    """One ordered Thinker token / Talker audio update from VenusOmni.

    ``total_token_ids`` is cumulative or delta according to the session
    capability.  It must contain raw model IDs before chat/tool parsing.  The
    attached audio is one Talker chunk and cannot enter playback until the
    Harness has consumed these IDs and excluded private delegate content.

    The newly appended token suffix represented by one step belongs to exactly
    one ``input_seq_cutoff``.  Serving must flush/split output at an input fence
    instead of merging new tokens from two input cutoffs into one step.  In
    cumulative mode, a repeated old prefix is allowed; this rule applies to
    the newly appended suffix.

    ``generation_epoch`` is monotonic for the whole session incarnation.  A
    new generation uses an epoch greater than every generation that has
    already finished.
    """

    session_id: str
    incarnation: int
    generation_id: str
    generation_epoch: int
    step_seq: int
    input_seq_cutoff: int
    at_ms: int
    total_token_ids: tuple[int, ...]
    audio: GeneratedAudio | None = None
    audio_chunk_seq: int | None = None
    turn_finished: bool = False
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        _require_positive_int(self.incarnation, "incarnation")
        _require_text(self.generation_id, "generation_id")
        _require_nonnegative_int(self.generation_epoch, "generation_epoch")
        _require_positive_int(self.step_seq, "step_seq")
        _require_nonnegative_int(self.input_seq_cutoff, "input_seq_cutoff")
        _require_nonnegative_int(self.at_ms, "at_ms")
        object.__setattr__(
            self,
            "total_token_ids",
            _validated_token_ids(self.total_token_ids),
        )
        if self.audio is not None and not isinstance(self.audio, GeneratedAudio):
            raise TypeError("audio must be GeneratedAudio or None")
        if self.audio is None and self.audio_chunk_seq is not None:
            raise ValueError("audio_chunk_seq requires a Talker audio chunk")
        if self.audio is not None:
            _require_positive_int(self.audio_chunk_seq, "audio_chunk_seq")
        if not isinstance(self.turn_finished, bool):
            raise TypeError("turn_finished must be a boolean")
        if self.turn_finished:
            _require_text(self.finish_reason, "finish_reason")
        elif self.finish_reason is not None:
            raise ValueError("finish_reason requires turn_finished=True")
        if not self.total_token_ids and self.audio is None and not self.turn_finished:
            raise ValueError("an unfinished output step cannot be empty")


@dataclass(frozen=True, slots=True)
class PrefillAppend:
    """One idempotent private backend injection into the existing KV cache."""

    session_id: str
    incarnation: int
    work_id: str
    attempt_id: str
    text_list: tuple[str, ...]
    visibility: PrefillVisibility = PrefillVisibility.PRIVATE
    resume_generation: bool = True

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        _require_positive_int(self.incarnation, "incarnation")
        _require_text(self.work_id, "work_id")
        _require_text(self.attempt_id, "attempt_id")
        texts = tuple(self.text_list)
        if len(texts) != 1:
            raise ValueError("text_list must contain exactly one backend prefill")
        _validate_backend_prefill(texts[0])
        object.__setattr__(self, "text_list", texts)
        visibility = PrefillVisibility(self.visibility)
        object.__setattr__(self, "visibility", visibility)
        if visibility is not PrefillVisibility.PRIVATE:
            raise ValueError("backend prefill must be private")
        if self.resume_generation is not True:
            raise ValueError("backend prefill must resume generation")


@dataclass(frozen=True, slots=True)
class PrefillApplied:
    """Acknowledgement that one attempt was applied to the same session KV."""

    session_id: str
    incarnation: int
    work_id: str
    attempt_id: str
    generation_id: str
    generation_epoch: int
    kv_position: int
    applied_at_ms: int
    deduplicated: bool = False

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        _require_positive_int(self.incarnation, "incarnation")
        _require_text(self.work_id, "work_id")
        _require_text(self.attempt_id, "attempt_id")
        _require_text(self.generation_id, "generation_id")
        _require_nonnegative_int(self.generation_epoch, "generation_epoch")
        _require_nonnegative_int(self.kv_position, "kv_position")
        _require_nonnegative_int(self.applied_at_ms, "applied_at_ms")
        if not isinstance(self.deduplicated, bool):
            raise TypeError("deduplicated must be a boolean")


@dataclass(frozen=True, slots=True)
class PlaybackAck:
    """Cumulative Talker chunk progress reported by the playback device."""

    session_id: str
    incarnation: int
    utterance_id: str
    cumulative_played_chunks: int
    at_ms: int
    caused_by_work_id: str = ""

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        _require_positive_int(self.incarnation, "incarnation")
        _require_text(self.utterance_id, "utterance_id")
        _require_nonnegative_int(
            self.cumulative_played_chunks,
            "cumulative_played_chunks",
        )
        _require_nonnegative_int(self.at_ms, "at_ms")
        if self.caused_by_work_id:
            _require_text(self.caused_by_work_id, "caused_by_work_id")


@dataclass(frozen=True, slots=True)
class PlaybackAccepted:
    """Serving-side acknowledgement of monotonic Talker chunk progress."""

    session_id: str
    incarnation: int
    utterance_id: str
    cumulative_played_chunks: int
    accepted_at_ms: int

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        _require_positive_int(self.incarnation, "incarnation")
        _require_text(self.utterance_id, "utterance_id")
        _require_nonnegative_int(
            self.cumulative_played_chunks,
            "cumulative_played_chunks",
        )
        _require_nonnegative_int(self.accepted_at_ms, "accepted_at_ms")


@dataclass(frozen=True, slots=True)
class CloseSession:
    """Release one session and its retained KV state."""

    session_id: str
    incarnation: int
    reason: str = "client_close"

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        _require_positive_int(self.incarnation, "incarnation")
        _require_text(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class SessionClosed:
    """Fence proving that the session can no longer emit model output."""

    session_id: str
    incarnation: int
    closed_at_ms: int

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        _require_positive_int(self.incarnation, "incarnation")
        _require_nonnegative_int(self.closed_at_ms, "closed_at_ms")


@runtime_checkable
class VenusOmniServingPort(Protocol):
    """Required async boundary implemented by a VenusOmni serving host.

    Implementations must serialize the data/generation owner commands for one
    session; media append may advance while an output pull is blocked, and
    ``close_session`` is the control-plane exception which must be callable
    concurrently to unblock that pull.  Input acceptance fences all earlier
    media before a corresponding ``ModelOutputStep``.
    ``append_prefill`` is idempotent by ``attempt_id`` and preserves the same
    KV cache. Playback acknowledgement is the cumulative number of native
    Talker audio chunks actually played, never generation or enqueue
    completion. Repeating
    ``append_prefill`` with the same ``attempt_id`` is idempotent and returns
    the original application fence.  ``close_session`` must be callable while
    ``next_output_step`` is blocked and must unblock that pending pull.
    """

    async def open_session(self, request: OpenSession) -> SessionOpened: ...

    async def append_text(self, request: TextAppend) -> InputAccepted: ...

    async def append_audio(self, request: AudioAppend) -> InputAccepted: ...

    async def append_video_frame(self, request: VideoFrameAppend) -> InputAccepted: ...

    async def append_video_segment(
        self, request: VideoSegmentAppend
    ) -> InputAccepted: ...

    async def next_output_step(
        self,
        session_id: str,
        *,
        incarnation: int,
        timeout_s: float | None = None,
    ) -> ModelOutputStep: ...

    async def append_prefill(self, request: PrefillAppend) -> PrefillApplied: ...

    async def acknowledge_playback(self, request: PlaybackAck) -> PlaybackAccepted: ...

    async def close_session(self, request: CloseSession) -> SessionClosed: ...


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-blank text")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _require_bytes(value: object, name: str) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _validate_session_event(session_id: str, incarnation: int, event_seq: int) -> None:
    _require_text(session_id, "session_id")
    _require_positive_int(incarnation, "incarnation")
    _require_positive_int(event_seq, "event_seq")


def _validated_token_ids(values: object) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("total_token_ids must be an iterable of integers")
    try:
        token_ids = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("total_token_ids must be an iterable of integers") from exc
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("total_token_ids must contain only integers")
        if token_id < 0:
            raise ValueError("total_token_ids must not contain negative IDs")
    return token_ids


def _validated_special_token_ids(values: object) -> Mapping[str, int]:
    if not isinstance(values, Mapping):
        raise TypeError("special_token_ids must be a mapping")
    token_ids: dict[str, int] = {}
    for token, token_id in values.items():
        _require_text(token, "special token")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("special token IDs must be integers")
        if token_id < 0:
            raise ValueError("special token IDs must not be negative")
        token_ids[token] = token_id
    missing = [token for token in VENUS_PROTOCOL_TOKENS if token not in token_ids]
    if missing:
        raise ValueError("missing VenusOmni protocol tokens: " + ", ".join(missing))
    if len(set(token_ids.values())) != len(token_ids):
        raise ValueError("special_token_ids must assign unique IDs")
    return MappingProxyType(token_ids)


def _validate_backend_prefill(text: object) -> str:
    _require_text(text, "backend prefill")
    assert isinstance(text, str)
    open_marker = "<backend>"
    close_marker = "</backend>"
    if not text.startswith(open_marker) or not text.endswith(close_marker):
        raise ValueError("backend prefill must have exactly one outer <backend> block")
    if text.count(open_marker) != 1 or text.count(close_marker) != 1:
        raise ValueError("backend prefill must have exactly one outer <backend> block")
    body = text[len(open_marker) : -len(close_marker)]
    if not body.strip():
        raise ValueError("backend prefill body must not be blank")
    forbidden = [token for token in VENUS_PROTOCOL_TOKENS if token in body]
    if forbidden:
        raise ValueError(
            "backend prefill body contains reserved tokens: " + ", ".join(forbidden)
        )
    return text


__all__ = [
    "PROTOCOL_VERSION",
    "AudioAppend",
    "TextAppend",
    "AudioFormat",
    "CloseSession",
    "GeneratedAudio",
    "InputAccepted",
    "ModelOutputStep",
    "OpenSession",
    "PlaybackAccepted",
    "PlaybackAck",
    "PrefillAppend",
    "PrefillApplied",
    "PrefillDeferred",
    "PrefillVisibility",
    "RawTokenMode",
    "SessionCapabilities",
    "SessionClosed",
    "SessionOpened",
    "VenusOmniServingPort",
    "VideoFrameAppend",
    "VideoSegmentAppend",
]
