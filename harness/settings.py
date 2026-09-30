"""Portable configuration for standalone Harness applications; no Demo dependency."""

import json
import os
import shutil
import tempfile
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path

from harness.agents.config import GeneralAgentConfig
from harness.config import GeminiConfig, LlamaCppConfig, ModelCallConfig, RoutingConfig
from harness.core.config import HarnessConfig
from harness.jobs.feedback import FeedbackConfig
from harness.bridge.config import venus_harness_config


def template():
    harness = venus_harness_config(HarnessConfig())
    harness = replace(harness, delegate=replace(harness.delegate, request_timeout_s=180, oralization_timeout_s=180, result_ttl_ms=600_000))
    return {
        "version": 1,
        "general_provider": "codex",
        "llamacpp": asdict(LlamaCppConfig()),
        "language": "zh",
        "workspace": "",
        "codex": {
            key: value
            for key, value in asdict(GeneralAgentConfig(effort="low")).items()
            if key not in {"workspace", "approval_policy", "sandbox", "queue_timeout_s"}
        },
        "routing": asdict(RoutingConfig()),
        "responses": asdict(ModelCallConfig()),
        "multimodal": asdict(ModelCallConfig()),
        "gemini": asdict(GeminiConfig()),
        "feedback": {**asdict(FeedbackConfig()), "progress_timeout_s": 180},
        "harness": {
            name: asdict(getattr(harness, name))
            for name in (
                "buffer",
                "workspace",
                "delegate",
                "runtime",
                "result_management",
            )
        },
    }


def config_path(path=None):
    return (
        Path(path or os.getenv("HARNESS_CONFIG", "runtime/harness.json"))
        .expanduser()
        .resolve()
    )


def merge(default, supplied):
    if isinstance(default, dict) and isinstance(supplied, dict):
        return {
            **default,
            **{key: merge(default.get(key), value) for key, value in supplied.items()},
        }
    return supplied


def settings(instance, values):
    if not isinstance(values, dict) or values.keys() - {
        field.name for field in fields(instance)
    }:
        raise ValueError("Unknown or invalid settings")
    updates = {
        key: settings(getattr(instance, key), value)
        if is_dataclass(getattr(instance, key))
        else value
        for key, value in values.items()
    }
    return replace(instance, **updates)


class HarnessSetup:
    def __init__(self, data, path):
        if not isinstance(data, dict) or data.keys() - template().keys():
            raise ValueError("Unknown configuration sections")
        # Previously Direct/Multimodal shared responses; preserve existing overrides.
        if "multimodal" not in data:
            legacy_response = data.get("responses", {})
            if not isinstance(legacy_response, dict):
                raise ValueError("Invalid response settings")
            inherited = legacy_response if legacy_response.get("provider", "codex") == "codex" else {}
            data = {**data, "multimodal": {
                key: value for key, value in inherited.items()
                if key in {"model", "effort", "timeout_s"}
            }}
        self.data = merge(template(), data)
        self.path = Path(path).resolve()
        self.general_provider = self.data["general_provider"]
        if self.general_provider not in {"codex", "llamacpp"}:
            raise ValueError("General provider must be codex or llamacpp")
        self.llamacpp = settings(LlamaCppConfig(), self.data["llamacpp"])
        self.language = self.data["language"]
        if self.data["version"] != 1 or self.language not in {"zh", "en"}:
            raise ValueError("Invalid configuration version or language")
        workspace = self.data["workspace"]
        if not isinstance(workspace, str):
            raise ValueError("Task workspace must be a path")
        self.data["workspace"] = workspace.strip()
        # An internal placeholder permits opening Settings before setup. The
        # session readiness check prevents using it until a path is supplied.
        self.workspace = (
            self.path.parent / Path(self.data["workspace"] or "./workspace").expanduser()
        ).resolve()
        if self.workspace == Path(self.workspace.anchor):
            raise ValueError("Choose a task workspace, not the filesystem root")
        codex = dict(self.data["codex"])
        if {
            "workspace",
            "approval_policy",
            "sandbox",
            "queue_timeout_s",
        } & codex.keys():
            raise ValueError("Workspace permissions are managed by the application")
        if not isinstance(codex.get("command"), (list, tuple)):
            raise ValueError("Codex command must be an argument list")
        codex["command"] = tuple(codex["command"])
        self.general = settings(
            GeneralAgentConfig(workspace=str(self.workspace)), codex
        )
        self.routing = settings(RoutingConfig(), self.data["routing"])
        self.responses = settings(ModelCallConfig(), self.data["responses"])
        self.multimodal = settings(ModelCallConfig(), self.data["multimodal"])
        self.gemini = settings(GeminiConfig(), self.data["gemini"])
        self.feedback = settings(FeedbackConfig(), self.data["feedback"])
        if self.feedback.journal_path and self.feedback.journal_path != ":memory:":
            self.feedback = replace(
                self.feedback,
                journal_path=str(
                    (
                        self.path.parent / Path(self.feedback.journal_path).expanduser()
                    ).resolve()
                ),
            )
        base = HarnessConfig(language=self.language)
        values = self.data["harness"]
        allowed = {"buffer", "workspace", "delegate", "runtime", "result_management"}
        if not isinstance(values, dict) or values.keys() - allowed:
            raise ValueError("Unknown Harness configuration")
        self.harness = replace(
            base,
            **{
                key: settings(getattr(base, key), value)
                for key, value in values.items()
            },
        )
        document_root = Path(self.harness.workspace.root or self.workspace).expanduser()
        self.harness = replace(
            self.harness,
            workspace=replace(
                self.harness.workspace,
                root=str((self.path.parent / document_root).resolve()),
            ),
        )

    @property
    def active_providers(self):
        providers = {self.general_provider, self.responses.provider, self.multimodal.provider}
        if self.routing.mode == "auto":
            providers.add(self.routing.provider)
        return providers

    @property
    def uses_codex(self):
        return "codex" in self.active_providers

    def check(self, *, web=True):
        problems = []
        if web and not self.data["workspace"]:
            problems.append("Configure Task workspace / 请填写任务工作目录")
        if self.uses_codex and not shutil.which(self.general.command[0]):
            problems.append("Codex executable not found / 找不到 Codex 程序")
        calls = [self.responses, self.multimodal]
        if self.routing.mode == "auto":
            calls.append(self.routing)
        if any(call.provider == "gemini" for call in calls) and not self.gemini.resolved_key():
            problems.append("Configure an official Gemini API key / 请配置官方 Gemini API Key")
        return {"ok": not problems, "problems": problems}


def load_setup(path=None):
    target = config_path(path)
    if path is not None and not target.is_file():
        raise ValueError(f"Harness configuration does not exist: {target}")
    return (
        HarnessSetup(json.loads(target.read_text()), target)
        if target.exists()
        else HarnessSetup(template(), target)
    )


def save_setup(data, path):
    target = config_path(path)
    setup = HarnessSetup(data, target)
    write_json(target, setup.data)
    return setup


def write_json(target, data):
    """Atomically replace a local JSON file, with owner-only permissions."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="harness-settings-", dir=target.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
