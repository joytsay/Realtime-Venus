"""Serialize a Realtime-Venus-Omni model session and expose the serving contract."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Mapping

from demos.model import wire as sw
from demos.model.decoding import configure_length_penalty
from demos.model.settings import DuplexSettings

_LOG = logging.getLogger("demos.model.adapter")

# 12 protocol tokens (mirror harness.core.venus.VENUS_PROTOCOL_TOKENS; hardcoded
# so the server never imports the harness).
PROTOCOL_TOKENS = (
    "<|speak|>",
    "<|listen|>",
    "<|turn_eos|>",
    "<|chunk_eos|>",
    "<unit>",
    "</unit>",
    "<image>",
    "</image>",
    "<delegate>",
    "</delegate>",
    "<backend>",
    "</backend>",
)


# ---------------------------------------------------------------------------
# Errors -> mapped to HTTP status by server.py
# ---------------------------------------------------------------------------


class AdapterError(RuntimeError):
    status_code = 400


class UnknownSession(AdapterError):
    status_code = 404


class IncarnationMismatch(AdapterError):
    status_code = 400


class EventSeqMismatch(AdapterError):
    status_code = 400


class SessionBusy(AdapterError):
    status_code = 409


class Unsupported(AdapterError):
    status_code = 400


class SessionClosedError(AdapterError):
    status_code = 410


class OutputTimeout(TimeoutError):
    status_code = 408


class ModelNotLoaded(AdapterError):
    status_code = 503


# ---------------------------------------------------------------------------
# Model backend protocol the adapter calls.  Real model = HuggingFaceMemoryDuplex;
# ---------------------------------------------------------------------------


def _official(model: Any) -> Any:
    """The object owning ``total_ids`` and the same-KV backend prefill."""
    return getattr(model, "official_duplex", model)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class _Session:
    __slots__ = (
        "session_id",
        "incarnation",
        "model",
        "closed",
        "event_seq",
        "input_seq",
        "consumed_input_seq",
        "audio_buf",
        "frame_buf",
        "frames",
        "texts",
        "audio_samples",
        "queue",
        "driver",
        "stop",
        "gen_epoch",
        "fg_gen",
        "resume_gen",
        "step_seq",
        "audio_seq",
        "prev_total_len",
        "ack_chunks",
        "prefill_attempts",
        "output_requested",
        "output_inflight",
        "inference_stage",
        "inference_started_at",
    )

    def __init__(self, session_id: str, incarnation: int, model: str) -> None:
        self.session_id = session_id
        self.incarnation = incarnation
        self.model = model
        self.closed = False
        self.event_seq = 0
        self.input_seq = 0
        self.consumed_input_seq = 0
        self.audio_samples = 0  # count of 16k samples buffered
        self.audio_buf = bytearray()
        self.frames: list[bytes] = []
        self.texts: list[str] = []
        self.queue: asyncio.Queue = asyncio.Queue()
        self.driver: asyncio.Task | None = None
        self.stop = asyncio.Event()
        self.gen_epoch = 0
        self.fg_gen: tuple[str, int] | None = None
        self.resume_gen: tuple[str, int] | None = None
        self.step_seq = 0
        self.audio_seq = 0
        self.prev_total_len = 0
        self.ack_chunks: dict[str, int] = {}
        self.prefill_attempts: dict[str, dict[str, Any]] = {}
        self.output_requested = False
        self.output_inflight = False
        self.inference_stage = "idle"
        self.inference_started_at: float | None = None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class RealtimeVenusOmniAdapter:
    """Serves one live duplex session on top of one model backend.

    Single session (``max_sessions=1``), matching the model's single KV cache.
    """

    def __init__(
        self,
        model: Any,
        *,
        special_token_ids: Mapping[str, int],
        model_revision: str,
        model_name: str = "Realtime-Venus-Omni",
        ref_audio: Any | None = None,
        prompt_wav_path: str | None = None,
        system_prompt: str | None = None,
        chunk_seconds: float = 1.0,
        sample_rate_hz: int = 16_000,
        max_new_speak_tokens_per_chunk: int = 50,
    ) -> None:
        self._model = model
        self._official = _official(model)
        self._needs_duplex = False
        self._special = dict(special_token_ids)
        self._model_revision = model_revision
        self._model_name = model_name
        self._ref_audio = ref_audio
        self._prompt_wav_path = prompt_wav_path
        # Custom duplex system prompt (identity).  None -> model's built-in
        # DEFAULT_SYSTEM_PROMPT (checkpoint model implementation:110).  Foreground prepare
        # forwards it via prefix_system_prompt (runtime.py:106 / engine
        # model_adapter.py:169 / official modeling.py:2795).
        self._system_prompt = system_prompt
        self._chunk_samples = int(sample_rate_hz * chunk_seconds)
        self._max_speak = int(max_new_speak_tokens_per_chunk)
        self._length_penalty = None
        self._decoding_implementation = "not_prepared"
        self._generation_options: dict[str, Any] = {}
        self._session: _Session | None = None
        self._opening_session: _Session | None = None
        self._open_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()  # serializes model forward passes
        self._wakeup = asyncio.Event()

    # -- construction -------------------------------------------------------

    @property
    def delegate_audio_filtered(self) -> bool:
        """Recognize the released model's span filter and early TTS flush.

        The current checkpoint exposes two session-state flags rather than a
        public capability attribute. Both must exist; legacy checkpoints keep
        the conservative Harness gate. An explicit model declaration takes
        precedence when a future checkpoint provides one.
        """
        declared = getattr(self._official, "delegate_audio_filtered", None)
        if declared is not None:
            return declared is True
        return (
            callable(getattr(self._official, "_convert_results_to_tts_input", None))
            and type(getattr(self._official, "_delegate_tts_skip", None)) is bool
            and type(getattr(self._official, "_delegate_turn_preflushed", None)) is bool
        )

    @classmethod
    async def load_real(
        cls,
        model_path: str,
        *,
        memory_minutes: int,
        ref_audio_path: str | None,
        prompt_wav_path: str | None = None,
        model_revision: str = "realtime-venus-omni-v1",
        model_name: str = "Realtime-Venus-Omni",
        system_prompt: str | None = None,
        chunk_seconds: float = 1.0,
        max_new_speak_tokens_per_chunk: int = 50,
        model_type: str = "omni",
    ) -> "RealtimeVenusOmniAdapter":
        model, tokenizer, ref_audio = await asyncio.to_thread(
            _load_real_blocking,
            model_path,
            memory_minutes,
            ref_audio_path,
            model_type,
        )
        special = {
            token: int(tokenizer.convert_tokens_to_ids(token))
            for token in PROTOCOL_TOKENS
        }
        missing = [t for t, v in special.items() if v is None or v < 0]
        if missing:
            raise ModelNotLoaded(
                "tokenizer missing protocol tokens: " + ", ".join(missing)
            )
        if len(set(special.values())) != len(special):
            raise ModelNotLoaded("protocol token ids are not unique")
        return cls(
            model,
            special_token_ids=special,
            model_revision=model_revision,
            model_name=model_name,
            ref_audio=ref_audio,
            prompt_wav_path=prompt_wav_path,
            system_prompt=system_prompt,
            chunk_seconds=chunk_seconds,
            max_new_speak_tokens_per_chunk=max_new_speak_tokens_per_chunk,
        )

    # -- 8 methods ----------------------------------------------------------

    async def open_session(self, req: dict[str, Any]) -> dict[str, Any]:
        protocol = req.get("protocol_version", sw.PROTOCOL_VERSION)
        if protocol != sw.PROTOCOL_VERSION:
            raise AdapterError(f"unsupported protocol_version: {protocol!r}")
        session_id = str(req["session_id"])
        model = str(req["model"])
        try:
            # Older non-browser clients keep the checkpoint's original behavior.
            length_penalty = DuplexSettings(
                length_penalty=req.get("length_penalty", 1.0)
            ).length_penalty
        except ValueError as exc:
            raise AdapterError(str(exc)) from exc
        async with self._open_lock:
            if self._session is not None and not self._session.closed:
                raise SessionBusy("a session is already open; max_sessions=1")
            incarnation = (self._session.incarnation + 1) if self._session else 1
            sess = _Session(session_id, incarnation, model)
            # Re-prepare the model for this session (prepare is session-scoped,
            # single-use: reset_memory first if a previous session used it).
            self._opening_session = sess
            _LOG.info("model prepare started session=%s", session_id)
            try:
                await self._model_call(
                    sess, "prepare", self._prepare_for_session, length_penalty
                )
            finally:
                self._opening_session = None
            self._session = sess
            sess.driver = asyncio.create_task(
                self._driver_loop(), name=f"adapter-driver:{session_id}"
            )
        _LOG.info(
            "open_session %s incarnation=%d length_penalty=%s decoding=%s",
            session_id,
            incarnation,
            self._length_penalty,
            self._decoding_implementation,
        )
        return sw.session_opened(
            session_id=session_id,
            incarnation=incarnation,
            model=model,
            model_revision=self._model_revision,
            special_token_ids=self._special,
            opened_at_ms=sw.now_ms(),
            video_segments=False,
            max_sessions=1,
            delegate_audio_filtered=self.delegate_audio_filtered,
        )

    def _prepare_for_session(self, length_penalty: float = 1.0) -> None:
        """Block in the model thread: reset + prepare a fresh duplex session."""

        from harness.instructions import read_instructions

        instructions = read_instructions().strip()
        system_prompt = self._system_prompt
        if instructions:
            from demos.model.prompts import REALTIME_VENUS_SYSTEM_PROMPT

            system_prompt = (system_prompt or REALTIME_VENUS_SYSTEM_PROMPT) + "\n\n" + instructions
        _LOG.info(
            "model prepare instructions chars=%d custom_instructions_chars=%d",
            len(system_prompt or ""), len(instructions),
        )

        if self._needs_duplex:
            self._model = self._model.as_duplex()
            self._official = _official(self._model)
            self._needs_duplex = False
        # If the runtime was already prepared by a previous session, reset first
        # so prepare() does not raise "already prepared" (runtime.py:104).
        runtime = getattr(self._model, "memory_runtime", None)
        if runtime is not None and hasattr(runtime, "reset_memory"):
            try:
                runtime.reset_memory()
            except Exception:  # noqa: BLE001 - reset may not be needed on a fresh load
                _LOG.debug("reset_memory skipped/failed", exc_info=True)
        started = time.monotonic()
        _LOG.info("model voice and system prompt initialization started")
        self._model.prepare(
            prefix_system_prompt=system_prompt,
            ref_audio=self._ref_audio,
            prompt_wav_path=self._prompt_wav_path,
        )
        _LOG.info(
            "model voice and system prompt initialization completed elapsed_s=%.3f",
            time.monotonic() - started,
        )
        try:
            options, implementation = configure_length_penalty(
                self._official, length_penalty
            )
        except ValueError as exc:
            raise Unsupported(str(exc)) from exc
        self._generation_options = options
        self._length_penalty = length_penalty
        self._decoding_implementation = implementation

    async def append_text(self, req: dict[str, Any]) -> dict[str, Any]:
        sess = self._require(req)
        text = req.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise AdapterError("text must contain 1 to 4000 characters")
        event_seq = int(req["event_seq"])
        async with self._turn_lock:
            if len(sess.texts) >= 8:
                raise AdapterError("too many pending typed messages")
            self._fence_event(sess, event_seq)
            sess.texts.append(text.strip())
            sess.input_seq += 1
            self._wakeup.set()
        return sw.input_accepted(
            session_id=sess.session_id, incarnation=sess.incarnation,
            event_seq=event_seq, input_seq=sess.input_seq, accepted_at_ms=sw.now_ms(),
        )

    async def append_audio(self, req: dict[str, Any]) -> dict[str, Any]:
        sess = self._require(req)
        event_seq = int(req["event_seq"])
        async with self._turn_lock:
            self._fence_event(sess, event_seq)
            pcm, sample_rate = sw.parse_audio_append(req)
            if sample_rate != 16_000:
                raise AdapterError("input audio must be 16000 Hz")
            sess.audio_buf.extend(pcm)
            sess.audio_samples += len(pcm) // 2
            sess.input_seq += 1
            self._wakeup.set()
            if sess.input_seq == 1 or sess.input_seq % 10 == 0:
                _LOG.info(
                    "audio received session=%s input_seq=%d samples=%d buffered_samples=%d",
                    sess.session_id, sess.input_seq, len(pcm) // 2, sess.audio_samples,
                )
        return sw.input_accepted(
            session_id=sess.session_id,
            incarnation=sess.incarnation,
            event_seq=event_seq,
            input_seq=sess.input_seq,
            accepted_at_ms=sw.now_ms(),
        )

    async def append_video_frame(self, req: dict[str, Any]) -> dict[str, Any]:
        if self._model_name == "Realtime-Venus-Audio":
            raise Unsupported("The Audio checkpoint does not accept video frames")
        sess = self._require(req)
        event_seq = int(req["event_seq"])
        async with self._turn_lock:
            self._fence_event(sess, event_seq)
            frame_bytes, _mime = sw.parse_frame_append(req)
            sess.frames.append(frame_bytes)
            sess.input_seq += 1
            self._wakeup.set()
        return sw.input_accepted(
            session_id=sess.session_id,
            incarnation=sess.incarnation,
            event_seq=event_seq,
            input_seq=sess.input_seq,
            accepted_at_ms=sw.now_ms(),
        )

    async def append_video_segment(self, req: dict[str, Any]) -> dict[str, Any]:
        # This adapter accepts sampled frames, not encoded video segments.
        raise Unsupported("video segments are not supported (video_segments=false)")

    async def next_output_step(
        self,
        session_id: str,
        *,
        incarnation: int,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        sess = self._require_by_id(session_id, incarnation)
        if sess.queue.empty() and sess.closed:
            raise SessionClosedError(f"session closed: {session_id}")
        if sess.queue.empty() and not sess.output_inflight:
            sess.output_requested = True
            self._wakeup.set()
        try:
            if timeout_s is None:
                item = await sess.queue.get()
            elif timeout_s <= 0:
                item = sess.queue.get_nowait()
            else:
                item = await asyncio.wait_for(sess.queue.get(), timeout_s)
        except asyncio.TimeoutError as exc:
            raise OutputTimeout("no model output within timeout") from exc
        except asyncio.QueueEmpty as exc:
            raise OutputTimeout("no model output ready") from exc
        if item is None:
            raise SessionClosedError(f"session closed: {session_id}")
        return item

    async def append_prefill(self, req: dict[str, Any]) -> dict[str, Any]:
        sess = self._require(req)
        work_id = str(req["work_id"])
        attempt_id = str(req["attempt_id"])
        text_list = tuple(req["text_list"])
        if len(text_list) != 1:
            raise AdapterError("text_list must contain exactly one backend prefill")
        backend_text = text_list[0]
        # idempotency by attempt_id for retry-safe feedback
        existing = sess.prefill_attempts.get(attempt_id)
        if existing is not None:
            if existing["text"] != backend_text:
                raise AdapterError("attempt_id reused with different prefill text")
            return sw.prefill_applied(
                session_id=sess.session_id,
                incarnation=sess.incarnation,
                work_id=work_id,
                attempt_id=attempt_id,
                generation_id=existing["generation_id"],
                generation_epoch=existing["generation_epoch"],
                kv_position=existing["kv_position"],
                applied_at_ms=existing["applied_at_ms"],
                deduplicated=True,
            )
        async with self._turn_lock:
            self._ensure_not_closed(sess)
            sess.output_requested = False
            if (
                sess.fg_gen is not None
                or sess.resume_gen is not None
                or not sess.queue.empty()
            ):
                raise SessionBusy("drain pending model output before backend prefill")
            gen_id, gen_epoch = self._allocate_generation(sess, prefix="backend")
            # Bypass the memory hijack: write <backend> directly to the live KV via
            # the official duplex, then generate resumes on the same KV.
            await asyncio.to_thread(
                self._official.streaming_prefill, None, None, [backend_text], 1, False
            )
            sess.resume_gen = (gen_id, gen_epoch)
            sess.step_seq = 0
            sess.audio_seq = 0
            sess.prev_total_len = len(self._official.total_ids)
            kv_position = sess.prev_total_len
            sess.prefill_attempts[attempt_id] = {
                "text": backend_text,
                "generation_id": gen_id,
                "generation_epoch": gen_epoch,
                "kv_position": kv_position,
                "applied_at_ms": sw.now_ms(),
            }
            self._wakeup.set()
        _LOG.info(
            "append_prefill work=%s attempt=%s gen=%s epoch=%d",
            work_id,
            attempt_id,
            gen_id,
            gen_epoch,
        )
        return sw.prefill_applied(
            session_id=sess.session_id,
            incarnation=sess.incarnation,
            work_id=work_id,
            attempt_id=attempt_id,
            generation_id=gen_id,
            generation_epoch=gen_epoch,
            kv_position=kv_position,
            applied_at_ms=sw.now_ms(),
            deduplicated=False,
        )

    async def acknowledge_playback(self, req: dict[str, Any]) -> dict[str, Any]:
        sess = self._require(req)
        utterance_id = str(req["utterance_id"])
        cumulative = int(req["cumulative_played_chunks"])
        async with self._turn_lock:
            self._ensure_not_closed(sess)
            previous = sess.ack_chunks.get(utterance_id, 0)
            if cumulative < previous:
                raise AdapterError("playback acknowledgement moved backwards")
            sess.ack_chunks[utterance_id] = cumulative
        return sw.playback_accepted(
            session_id=sess.session_id,
            incarnation=sess.incarnation,
            utterance_id=utterance_id,
            cumulative_played_chunks=cumulative,
            accepted_at_ms=sw.now_ms(),
        )

    async def close_session(self, req: dict[str, Any]) -> dict[str, Any]:
        reason = str(req.get("reason", "client_close"))
        sess = self._require(req, allow_closed=True)
        async with self._open_lock:
            if sess.closed and sess.stop.is_set():
                return sw.session_closed(
                    session_id=sess.session_id,
                    incarnation=sess.incarnation,
                    closed_at_ms=sw.now_ms(),
                )
            sess.closed = True
            sess.stop.set()
            self._wakeup.set()
            # Unblock any pending next_output_step pull.
            sess.queue.put_nowait(None)
            # Finish the in-flight worker before destroying KV. Keep the open
            # lock until cleanup ends, so a browser reconnect cannot race it.
            stop_fn = getattr(self._official, "set_session_stop", None)
            if callable(stop_fn):
                stop_fn()
            async with self._turn_lock:
                await asyncio.to_thread(self._teardown_model)
            if sess.driver is not None:
                sess.driver.cancel()
                try:
                    await sess.driver
                except asyncio.CancelledError:
                    pass
        _LOG.info("close_session %s reason=%s", sess.session_id, reason)
        return sw.session_closed(
            session_id=sess.session_id,
            incarnation=sess.incarnation,
            closed_at_ms=sw.now_ms(),
        )

    def _teardown_model(self) -> None:
        as_simplex = getattr(self._model, "as_simplex", None)
        if callable(as_simplex):
            self._model = as_simplex()  # releases KV / returns simplex model
            self._official = _official(self._model)
            self._needs_duplex = callable(getattr(self._model, "as_duplex", None))

    async def health(self) -> dict[str, Any]:
        sess = self._opening_session or self._session
        sp = self._system_prompt
        return {
            "model_revision": self._model_revision,
            "initializing": self._opening_session is not None,
            "delegate_audio_filtered": self.delegate_audio_filtered,
            "duplex_decoding": {
                "length_penalty": self._length_penalty,
                "implementation": self._decoding_implementation,
            },
            "uses_model_default_system_prompt": sp is None,
            "system_prompt_preview": (sp[:120] + "…") if sp else None,
            "session": None
            if sess is None
            else {
                "session_id": sess.session_id,
                "incarnation": sess.incarnation,
                "closed": sess.closed,
                "input_seq": sess.input_seq,
                "output_step_seq": sess.step_seq,
                "buffered_audio_samples": sess.audio_samples,
                "inference_stage": sess.inference_stage,
                "inference_elapsed_s": (
                    round(time.monotonic() - sess.inference_started_at, 3)
                    if sess.inference_started_at is not None else None
                ),
            },
        }

    # -- driver -------------------------------------------------------------

    async def _driver_loop(self) -> None:
        sess = self._session
        if sess is None:
            return
        try:
            while not sess.closed:
                acted = False
                async with self._turn_lock:
                    if sess.closed:
                        break
                    if sess.output_requested:
                        sess.output_inflight = True
                        try:
                            acted = await self._step_once(sess)
                            if acted:
                                sess.output_requested = False
                        finally:
                            sess.output_inflight = False
                if not acted:
                    if self._wakeup.is_set():
                        self._wakeup.clear()
                        continue
                    self._wakeup.clear()
                    await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - never let the driver die silently
            _LOG.exception("driver loop crashed")
            sess.closed = True
            sess.queue.put_nowait(None)

    async def _step_once(self, sess: _Session) -> bool:
        if sess.resume_gen is not None:
            gen_id, gen_epoch = sess.resume_gen
            if sess.step_seq:
                bucket = self._collect_bucket(sess)
                if bucket is None:
                    import numpy as np

                    bucket = _Bucket(
                        audio=np.zeros(self._chunk_samples, dtype=np.float32), frames=[]
                    )
                await self._prefill_bucket(sess, bucket, is_backend=True)
            step, finished = await self._generate_step(
                sess, gen_id, gen_epoch, is_backend=True
            )
            await sess.queue.put(step)
            if finished:
                sess.resume_gen = None
            return True
        bucket = self._collect_bucket(sess)
        if bucket is None:
            return False
        await self._prefill_bucket(sess, bucket, is_backend=False)
        if sess.fg_gen is None:
            sess.fg_gen = self._allocate_generation(sess, prefix="foreground")
            sess.step_seq = 0
            sess.audio_seq = 0
            sess.prev_total_len = len(self._official.total_ids)
        gen_id, gen_epoch = sess.fg_gen
        step, finished = await self._generate_step(
            sess, gen_id, gen_epoch, is_backend=False
        )
        await sess.queue.put(step)
        if finished:
            sess.fg_gen = None
        return True

    async def _prefill_bucket(
        self, sess: _Session, bucket: _Bucket, *, is_backend: bool
    ) -> None:
        import numpy as np

        rms = (
            float(np.sqrt(np.mean(np.square(bucket.audio))))
            if len(bucket.audio) else 0.0
        )
        _LOG.info(
            "model prefill started session=%s backend=%s samples=%d frames=%d rms=%.6f",
            sess.session_id, is_backend, len(bucket.audio), len(bucket.frames), rms,
        )
        model = self._official if is_backend else self._model
        frames = None if self._model_name == "Realtime-Venus-Audio" else bucket.frames
        await self._model_call(
            sess, "prefill", model.streaming_prefill,
            bucket.audio, frames, bucket.texts or None, 1, False,
        )

    async def _model_call(
        self, sess: _Session, stage: str, function: Callable[..., Any], *args, **kwargs
    ) -> Any:
        started = time.monotonic()
        sess.inference_stage = stage
        sess.inference_started_at = started
        try:
            result = await asyncio.to_thread(function, *args, **kwargs)
        except BaseException:
            sess.inference_stage = f"{stage}_failed"
            raise
        else:
            sess.inference_stage = "idle"
            _LOG.info(
                "model %s completed session=%s elapsed_s=%.3f",
                stage, sess.session_id, time.monotonic() - started,
            )
            return result
        finally:
            sess.inference_started_at = None

    async def _generate_step(
        self, sess: _Session, gen_id: str, gen_epoch: int, *, is_backend: bool
    ) -> tuple[dict[str, Any], bool]:
        gen_fn = (self._official if is_backend else self._model).streaming_generate
        # delta tracking: snapshot total_ids length right before generate
        prev_len = sess.prev_total_len
        _LOG.info(
            "model generate started session=%s backend=%s", sess.session_id, is_backend
        )
        result = await self._model_call(
            sess, "generate", gen_fn,
            prompt_wav_path=None,
            decode_mode="sampling",
            max_new_speak_tokens_per_chunk=self._max_speak,
            **self._generation_options,
        )
        ids = list(self._official.total_ids)
        delta = tuple(ids[prev_len:])
        if any(
            self._special[token] in delta for token in ("<delegate>", "</delegate>")
        ):
            _LOG.info(
                "model emitted delegate control tokens session=%s generation=%s ids=%s",
                sess.session_id,
                gen_id,
                delta,
            )
        if not result.get("is_listen"):
            _LOG.info(
                "model speak session=%s backend=%s end=%s ids=%s",
                sess.session_id,
                is_backend,
                result.get("end_of_turn"),
                delta,
            )
        sess.prev_total_len = len(ids)
        wave = None if result.get("is_listen") else result.get("audio_waveform")
        audio_dict = sw.generated_audio_dict(wave, 24_000)
        _LOG.info(
            "model output session=%s listen=%s end=%s tokens=%d audio=%s",
            sess.session_id, result.get("is_listen"), result.get("end_of_turn"), len(delta), audio_dict is not None,
        )
        audio_chunk_seq = None
        if audio_dict is not None:
            sess.audio_seq += 1
            audio_chunk_seq = sess.audio_seq
        sess.step_seq += 1
        # Foreground listen units leave an idle boundary for backend prefill.
        # After prefill, listen is only a pause before the spoken answer; its
        # silence must not complete delivery or detach the answer from its Work.
        finished = bool(
            result.get("end_of_turn") or (result.get("is_listen") and not is_backend)
        )
        finish_reason = "turn_eos" if finished else None
        step = sw.model_output_step(
            session_id=sess.session_id,
            incarnation=sess.incarnation,
            generation_id=gen_id,
            generation_epoch=gen_epoch,
            step_seq=sess.step_seq,
            input_seq_cutoff=sess.consumed_input_seq,
            at_ms=sw.now_ms(),
            total_token_ids=delta,
            audio=audio_dict,
            audio_chunk_seq=audio_chunk_seq,
            turn_finished=finished,
            finish_reason=finish_reason,
        )
        return step, finished

    def _collect_bucket(self, sess: _Session) -> Any | None:
        if sess.event_seq == 0:
            return None
        # bucket is pure-python; nothing here needs to run in an await context
        if sess.audio_samples < self._chunk_samples and not sess.frames and not sess.texts:
            return None
        if sess.audio_samples < self._chunk_samples and not sess.texts:
            # frame(s) but no enough audio yet -> wait for audio (v1 needs audio to drive)
            return None
        # consume up to one chunk of audio + any pending frames
        take_samples = min(sess.audio_samples, self._chunk_samples)
        take_bytes = take_samples * 2
        pcm = bytes(sess.audio_buf[:take_bytes])
        del sess.audio_buf[:take_bytes]
        sess.audio_samples -= take_samples
        import numpy as np

        audio = sw.pcm16le_to_float32(pcm) if pcm else np.zeros(0, dtype=np.float32)
        # decode wire image bytes -> PIL RGB for the real model; invalid image
        # bytes fall back raw (see serving_wire.decode_frames).
        if sess.texts and len(audio) < self._chunk_samples:
            audio = np.pad(audio, (0, self._chunk_samples - len(audio)))
        texts = list(sess.texts)
        sess.texts.clear()
        frames = sw.decode_frames(list(sess.frames)) if sess.frames else []
        sess.frames.clear()
        sess.consumed_input_seq = sess.input_seq
        return _Bucket(audio=audio, frames=frames, texts=texts)

    def _allocate_generation(self, sess: _Session, *, prefix: str) -> tuple[str, int]:
        sess.gen_epoch += 1
        gen_id = f"{prefix}-{sess.session_id}:{sess.gen_epoch}"
        return gen_id, sess.gen_epoch

    # -- helpers ------------------------------------------------------------

    def _require(self, req: dict[str, Any], *, allow_closed: bool = False) -> _Session:
        return self._require_by_id(
            str(req.get("session_id", "")),
            int(req.get("incarnation", 0)),
            allow_closed=allow_closed,
        )

    def _require_by_id(
        self, session_id: str, incarnation: int, *, allow_closed: bool = False
    ) -> _Session:
        sess = self._session
        if sess is None or sess.session_id != session_id:
            raise UnknownSession(f"unknown session: {session_id}")
        if sess.incarnation != incarnation:
            raise IncarnationMismatch("session incarnation does not match")
        if sess.closed and not allow_closed:
            raise SessionClosedError(f"session closed: {session_id}")
        return sess

    @staticmethod
    def _ensure_not_closed(sess: _Session) -> None:
        if sess.closed:
            raise SessionClosedError(f"session closed: {sess.session_id}")

    @staticmethod
    def _fence_event(sess: _Session, event_seq: int) -> None:
        expected = sess.event_seq + 1
        if event_seq != expected:
            raise EventSeqMismatch(
                f"event_seq must be contiguous; expected {expected}, got {event_seq}"
            )
        sess.event_seq = event_seq


class _Bucket:
    __slots__ = ("audio", "frames", "texts")

    def __init__(self, audio: Any, frames: list[bytes], texts: list[str] | None = None) -> None:
        self.audio = audio
        self.frames = frames
        self.texts = texts or []


# ---------------------------------------------------------------------------
# Real model loading (heavy, runs in a worker thread)
# ---------------------------------------------------------------------------


def _load_real_blocking(
    model_path: str, memory_minutes: int, ref_audio_path: str | None, model_type: str = "omni"
):
    import librosa
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    if model_type not in {"omni", "audio"}:
        raise ValueError("Unknown model type")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        **({"init_vision": False, "init_audio": True, "init_tts": True} if model_type == "audio" else {}),
    )
    model.eval().cuda()
    if model_type == "omni":
        model.use_memory(memory_minutes=memory_minutes)
    model = model.as_duplex(generate_audio=True) if model_type == "audio" else model.as_duplex()
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
    ref_audio = None
    if ref_audio_path:
        ref_audio, _ = librosa.load(ref_audio_path, sr=16_000, mono=True)
        ref_audio = ref_audio.astype(np.float32)
    return model, tokenizer, ref_audio
