"""Native llama.cpp Chat Completions transport, without Codex or credentials."""
import asyncio
import json
import time

import httpx

from harness.core.models import BackendResponse


class LlamaCppBackend:
    provider_name = "llamacpp"

    def __init__(self, config, *, model=None, timeout_s=180, transport=None):
        self.config = config
        self.model_name = model or config.model
        self.timeout_s = timeout_s
        self._transport = transport

    async def complete(self, instructions, context):
        return await self.chat([
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ])

    async def chat(self, messages, schema=None):
        body = {
            "model": self.model_name, "messages": messages, "stream": False,
            "temperature": 0.2, "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if schema is not None:
            body["response_format"]["schema"] = schema
        try:
            async with asyncio.timeout(self.timeout_s):
                async with httpx.AsyncClient(timeout=self.timeout_s, trust_env=False,
                                             transport=self._transport) as client:
                    response = await client.post(
                        self.config.base_url.rstrip("/") + "/chat/completions", json=body)
        except (TimeoutError, httpx.TimeoutException):
            raise RuntimeError("llama.cpp request timed out") from None
        except httpx.HTTPError:
            raise RuntimeError("Cannot connect to llama.cpp; check the server URL") from None
        if not response.is_success:
            raise RuntimeError(f"llama.cpp HTTP {response.status_code}; check model availability and context size")
        try:
            choice = response.json()["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("incomplete response")
            text = choice["message"]["content"]
            if not isinstance(json.loads(text), dict):
                raise ValueError("expected JSON object")
        except (ValueError, KeyError, IndexError, TypeError):
            raise RuntimeError("llama.cpp returned incomplete or invalid JSON; check output/context limits") from None
        return text


def check_server(config):
    """Check router availability and the configured model inventory without inference."""
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = client.get(config.base_url.rstrip("/") + "/models")
            response.raise_for_status()
            models = response.json()["data"]
            if not any(item.get("id") == config.model for item in models):
                return {"state": "error", "message": "llama.cpp configured model is not available; check /v1/models"}
        return {"state": "ready", "message": "llama.cpp is reachable and the configured model is available"}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return {"state": "error", "message": "Cannot reach llama.cpp; check the server URL and /v1/models"}


class LlamaCppDirect:
    def __init__(self, backend):
        self.backend = backend

    async def execute(self, request, context):
        if (context.audio or context.video or context.snapshot.audio_chunks or
                any(f.mime_type.startswith(("image/", "audio/", "video/"))
                    for f in context.snapshot.input_files)):
            raise ValueError("This llama.cpp backend is text-only; provide a text description or select a multimodal provider")
        started = time.monotonic()
        raw = await self.backend.complete(
            'Answer using the supplied text only. Return JSON {"speech":"answer"} in the requested language.',
            {"query": request.query, "language": request.language})
        speech = json.loads(raw).get("speech")
        if not isinstance(speech, str) or not speech.strip():
            raise ValueError("llama.cpp returned empty speech")
        return BackendResponse(speech, self.backend.provider_name, self.backend.model_name,
                               int((time.monotonic() - started) * 1000))

    async def aclose(self):
        pass
