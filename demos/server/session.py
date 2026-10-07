"""One browser, one model session, one model-owner output loop."""

import asyncio
import logging
import secrets
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path

from harness.core.speech import limit_spoken_sentences

import anyio

from harness.bridge.host import AudioDisposition, VenusOmniServingHost
from harness.core.models import AudioChunk, VideoFrame

from .media import video_buckets, audio_buckets
from .protocol import decode_media, playback_message

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
VIDEO_TAIL_SECONDS = 8


class BrowserPlayback:
    def __init__(self, send):
        self.send = send

    async def enqueue(self, request):
        await self.send(playback_message(request))


class WebAgentSession:
    def __init__(
        self,
        websocket,
        factory,
        *,
        mode="omni",
        settings_path=None,
        input_source="live",
    ):
        self.websocket = websocket
        self.factory = factory
        self.mode = mode
        self.settings_path = settings_path
        self.input_source = input_source
        self.session_id = f"web-{uuid.uuid4().hex}"
        self.upload_token = secrets.token_urlsafe(32)
        self._video_task = None
        self._video_idle_task = None
        self._upload_busy = False
        self._closing = False
        self.resources = None
        self.host = None
        self._send_lock = asyncio.Lock()
        self._last_works = None
        self._audio_end = 0
        self._last_video_ms = 0
        self.started_at_ms = int(time.time() * 1000)

    async def send(self, message):
        async with self._send_lock:
            await self.websocket.send_json(message)

    async def run(self):
        await self.websocket.accept()
        tasks = []
        try:
            if self.mode not in {"omni", "audio"}:
                raise ValueError("unsupported input mode")
            if self.input_source not in ({"live", "audio_file", "text"} if self.mode == "audio" else {"live", "video", "audio_file", "text"}):
                raise ValueError("unsupported media source")
            self.resources = await self.factory(
                mode=self.mode, settings_path=self.settings_path
            )
            if not self.resources.model.lower().startswith("realtime-venus-"):
                raise ValueError("frontend model identifier must use the Venus series")
            self.started_at_ms = int(time.time() * 1000)
            self.host = await VenusOmniServingHost.open(
                agent=self.resources.agent,
                serving=self.resources.serving,
                playback=BrowserPlayback(self.send),
                session_id=self.session_id,
                model=self.resources.model,
                tokenizer=self.resources.tokenizer,
                mute_delegate_audio=True,
            )
            await self.send({"type": "ready", **self.ui_state()})
            tasks = [
                asyncio.create_task(fn())
                for fn in (self.receive, self.output, self.works)
            ]
            if self.input_source == "text":
                tasks.append(asyncio.create_task(self._video_idle()))
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except Exception as exc:
            if type(exc).__name__ not in {"WebSocketDisconnect", "ClientDisconnected"}:
                logger.exception("Venus Web session failed")
                with suppress(Exception):
                    await self.send(
                        {
                            "type": "fatal_error",
                            "error": "Venus 会话启动或运行失败，请检查服务端日志。",
                        }
                    )
        finally:
            self._closing = True
            with anyio.CancelScope(shield=True):
                if self._video_task:
                    tasks.append(self._video_task)
                if self._video_idle_task:
                    tasks.append(self._video_idle_task)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if self.host:
                    with suppress(Exception):
                        await self.host.close()
                if self.resources:
                    with suppress(Exception):
                        await self.resources.close()
                with suppress(Exception):
                    await self.websocket.close()

    def ui_state(self):
        return {
            "model": self.resources.model,
            "harness": "Realtime-Venus-Harness",
            "session_id": self.session_id,
            "session_started_at_ms": self.started_at_ms,
            "upload_token": self.upload_token,
            "input_source": self.input_source,
            "delegate_audio_filtered": self.host.opened.capabilities.delegate_audio_filtered,
            "delegate_audio_policy": "mute_until_turn_end",
        }

    def work_snapshot(self):
        # Only session-local public status; private model/tool traces stay server-side.
        snapshots = []
        for work in reversed(self.host.session.list_works()):
            item = {
                key: work.get(key)
                for key in (
                    "work_id",
                    "objective",
                    "capability",
                    "state",
                    "phase",
                    "created_at",
                    "started_at",
                    "completed_at",
                    "updated_at",
                    "queue_position",
                    "execution_outcome",
                    "delivery_status",
                )
            }
            item["progress_text"] = (work.get("progress") or {}).get("text", "")
            item["error_message"] = (work.get("fault") or {}).get("message", "")
            item["feedback"] = [
                {key: record.get(key) for key in ("text", "terminal", "at", "kind")}
                for record in work.get("feedback_records", [])[-20:]
            ]
            result = self.host.session.get_work_result(item["work_id"]) or {}
            item["result_text"] = next(
                (
                    result[key]
                    for key in ("speech", "text", "summary", "full_result")
                    if isinstance(result.get(key), str)
                ),
                "",
            )
            final_feedback = next(
                (record.get("text", "") for record in reversed(work.get("feedback_records", []))
                 if record.get("terminal") and record.get("text")),
                "",
            )
            item["result_text"] = limit_spoken_sentences(final_feedback or item["result_text"], 3)
            item["artifacts"] = [
                {
                    "index": index,
                    "name": Path(artifact["path"]).name,
                    "mime_type": artifact.get("mime_type", "application/octet-stream"),
                }
                for index, artifact in enumerate(result.get("artifacts", []))
            ]
            snapshots.append(item)
        return snapshots

    async def works(self):
        while True:
            snapshot = self.work_snapshot()
            if snapshot != self._last_works:
                await self.send({"type": "works", "works": snapshot})
                self._last_works = snapshot
            await asyncio.sleep(0.25)

    async def output(self):
        finished = True
        while True:
            if (
                finished
                and self.host.queued_backend_results
                and not self.host.active_backend_turn
            ):
                try:
                    await self.host.begin_backend_turn(timeout_s=0)
                    finished = False
                except (TimeoutError, asyncio.TimeoutError):
                    pass
            try:
                processed = await self.host.next_output(timeout_s=1.0)
            except (TimeoutError, asyncio.TimeoutError):
                continue
            finished = processed.output.turn_finished
            logger.info(
                "web output session=%s audio=%s disposition=%s text_chars=%d finished=%s",
                self.session_id, processed.output.audio is not None,
                processed.audio_disposition, len(processed.model_step.visible_text or ""), finished,
            )
            if finished:
                # The audio sink has already enqueued this turn's final chunk.
                # Let the client flush a short buffered tail without guessing
                # whether a network pause means the model has finished.
                await self.send(
                    {"type": "playback_end", "utterance_id": processed.utterance_id}
                )
            # Text attached to audio is emitted by BrowserPlayback, only after the gate.
            if (
                processed.model_step.visible_text
                and processed.audio_disposition is not AudioDisposition.QUEUED
            ):
                await self.send(
                    {
                        "type": "text",
                        "text": processed.model_step.visible_text,
                        "utterance_id": processed.utterance_id,
                    }
                )

    async def receive(self):
        while True:
            message = await self.websocket.receive_json()
            command = message.get("type", "") if isinstance(message, dict) else ""
            if command == "stop_session":
                return
            try:
                if not isinstance(message, dict):
                    raise TypeError("消息必须是 JSON object")
                await self.handle(command, message)
            except (TypeError, ValueError, RuntimeError, KeyError) as exc:
                await self.send(
                    {"type": "command_error", "command": command, "error": str(exc)}
                )

    async def handle(self, command, message):
        now = int(time.time() * 1000)
        if command in {"audio", "video_frame"} and self.input_source != "live":
            raise ValueError("视频模式请上传文件；实时采集需重新开始实时会话")
        if command == "audio":
            if message.get("format") != "pcm16" or message.get("sample_rate") != 16000:
                raise ValueError("需要 16000Hz 单声道 PCM16")
            data = decode_media(message, kind="audio")
            duration = max(1, round(len(data) / 32))
            start = max(self._audio_end, now - duration)
            await self.host.append_audio(
                AudioChunk(data=data, start_ms=start, end_ms=start + duration)
            )
            self._audio_end = start + duration
        elif command == "text":
            text = message.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                raise ValueError("Text must contain 1 to 4000 characters")
            await self.host.append_text(text.strip())
            await self.send({"type": "user_text", "text": text.strip()})
        elif command == "video_frame":
            if self.mode != "omni":
                raise ValueError("纯语音模式不接收画面")
            await self.host.append_video_frame(
                VideoFrame(
                    data=decode_media(message, kind="video_frame"), captured_at_ms=now
                )
            )
            self._last_video_ms = now
        elif command == "playback_ack":
            count = message.get("chunks")
            if type(count) is not int or count < 1:
                raise ValueError("播放确认必须包含正整数 chunks")
            await self.host.acknowledge_playback(
                str(message.get("utterance_id", "")), count
            )
        elif command == "cancel_work":
            result = await self.host.session.cancel_work(
                str(message.get("work_id", ""))
            )
            await self.send({"type": "cancel_result", "result": result})
        elif command == "get_works":
            await self.send({"type": "works", "works": self.work_snapshot()})
        elif command == "get_ui_state":
            await self.send({"type": "ready", **self.ui_state()})
        elif command == "ping":
            await self.send({"type": "pong"})
        else:
            raise ValueError("未知 Web 命令")

    async def upload_video(self, stream):
        if self.input_source != "video" or self.mode != "omni":
            raise ValueError("请先开始视频会话")
        await self._upload_media(stream)

    async def upload_audio(self, stream):
        if self.input_source != "audio_file" or self.mode not in {"audio", "omni"}:
            raise ValueError("请先开始音频文件会话")
        await self._upload_media(stream)

    async def _upload_media(self, stream):
        if self._closing or not self.host or self.input_source == "live":
            raise ValueError("请先开始文件会话")
        if self._upload_busy:
            raise ValueError("请等待当前文件处理完成")
        self._upload_busy = True
        temp = tempfile.TemporaryDirectory(prefix="venus-upload-")
        path = Path(temp.name) / "input.media"
        transferred = False
        try:
            if self._video_idle_task:
                self._video_idle_task.cancel()
                await asyncio.gather(self._video_idle_task, return_exceptions=True)
                self._video_idle_task = None
            size = 0
            with path.open("wb") as output:
                async for chunk in stream:
                    if self._closing:
                        raise ValueError("会话已结束")
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise ValueError("文件大小不能超过 200 MB")
                    await asyncio.to_thread(output.write, chunk)
            if not size or self._closing:
                raise ValueError("文件为空或会话已结束")
            self._video_task = asyncio.create_task(self._feed_video(path, temp))
            transferred = True
        finally:
            if not transferred:
                temp.cleanup()
                self._upload_busy = False

    async def _feed_video(self, path, temp):
        iterator = audio_buckets(path) if self.input_source == "audio_file" else video_buckets(path)
        sent = 0
        try:
            await self.send(
                {"type": "media_status", "state": "processing", "seconds": 0}
            )
            started = asyncio.get_running_loop().time()
            while True:
                # Await an in-flight decoder even on cancellation before closing
                # its generator; Python generators cannot be closed while running.
                decode = asyncio.create_task(asyncio.to_thread(next, iterator, None))
                try:
                    bucket = await asyncio.shield(decode)
                except asyncio.CancelledError:
                    await decode
                    raise
                if bucket is None:
                    break
                now = int(time.time() * 1000)
                start = max(self._audio_end, now)
                if self.mode == "omni" and bucket.jpeg:
                    await self.host.append_video_frame(VideoFrame(bucket.jpeg, start))
                await self.host.append_audio(
                    AudioChunk(bucket.pcm, start, start + 1000)
                )
                self._audio_end = start + 1000
                sent += 1
                await self.send(
                    {"type": "media_status", "state": "feeding", "seconds": sent}
                )
                await asyncio.sleep(
                    max(0, started + sent - asyncio.get_running_loop().time())
                )
            if not sent:
                raise ValueError("文件没有可解码内容")
            # Keep the model clock moving briefly after EOF so the last spoken
            # question has time to receive a reply.
            for _ in range(VIDEO_TAIL_SECONDS):
                start = max(self._audio_end, int(time.time() * 1000))
                await self.host.append_audio(
                    AudioChunk(b"\0\0" * 16000, start, start + 1000)
                )
                self._audio_end = start + 1000
                await asyncio.sleep(1)
            await self.send(
                {"type": "media_status", "state": "complete", "seconds": sent}
            )
            # Duplex generation needs a new input unit for every output unit,
            # including replies and backend work that finish after the file.
            self._video_idle_task = asyncio.create_task(self._video_idle())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Uploaded media failed")
            with suppress(Exception):
                await self.send(
                    {
                        "type": "media_status",
                        "state": "error",
                        "seconds": sent,
                        "error": "媒体处理失败，请检查文件格式和服务日志",
                    }
                )
        finally:
            iterator.close()
            temp.cleanup()
            self._upload_busy = False

    async def _video_idle(self):
        while not self._closing:
            await asyncio.sleep(1)
            start = max(self._audio_end, int(time.time() * 1000))
            await self.host.append_audio(
                AudioChunk(b"\0\0" * 16000, start, start + 1000)
            )
            self._audio_end = start + 1000
