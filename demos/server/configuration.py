"""Browser access to local task settings; no credentials are returned to the UI."""

import copy
import hashlib
import json
import secrets
import tempfile
from dataclasses import asdict
from pathlib import Path
from harness.settings import save_setup as save_harness_setup, write_json
from harness.instructions import read_instructions, instructions_revision
from demos.deployment import frontend_settings, updated_frontend_document

from demos.settings import UserSetup, config_path, load_setup, save_setup
from .codex_login import CodexLoginCheck


class WebConfiguration:
    def __init__(self, path, demo_path=None):
        self.path = config_path(path)
        self.demo_path = Path(demo_path).resolve() if demo_path else None
        self.token = secrets.token_urlsafe(32)
        self.login = CodexLoginCheck()

    def revision(self):
        paths = (self.path, self.demo_path) if self.demo_path else (self.path,)
        parts = [p.read_bytes() if p.exists() else b"new" for p in paths]
        parts.append(instructions_revision().encode("ascii"))
        return hashlib.sha256(b"\0".join(parts)).hexdigest()

    def load(self):
        setup = load_setup(self.path)
        if self.demo_path:
            setup.duplex, _ = frontend_settings(self.demo_path, setup.duplex)
            setup.data["duplex"] = asdict(setup.duplex)
        return setup

    def save_startup(self, data):
        if self.demo_path:
            save_harness_setup({k: v for k, v in data.items() if k != "duplex"}, self.path)
        else:
            save_setup(data, self.path)

    def check(self, setup, *, force_login=False):
        result = setup.check()
        login = (self.login.check(setup.general.command[0], force=force_login)
                 if setup.uses_codex else {"state": "not_required", "message": "Codex login is not required"})
        if "llamacpp" in setup.active_providers:
            from harness.llm.llamacpp import check_server
            local = check_server(setup.llamacpp)
            if local["state"] != "ready":
                result["problems"].append(local["message"])
            if not setup.uses_codex:
                login = local
        if login["state"] not in {"logged_in", "ready", "not_required"} and login["message"] not in result["problems"]:
            result["problems"].append(login["message"])
        return {"ok": not result["problems"], "problems": result["problems"], "codex_login": login}

    def status(self, *, force_login=False):
        try:
            return self.path.exists() and self.check(self.load(), force_login=force_login)["ok"]
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def public(self):
        setup = self.load()
        result = self.check(setup, force_login=True)
        public_data = copy.deepcopy(setup.data)
        public_data["gemini"]["api_key"] = ""
        return {
            "data": public_data,
            "gemini_key_configured": bool(setup.gemini.resolved_key()),
            "revision": self.revision(),
            "token": self.token,
            "instructions": read_instructions(),
            "instructions_revision": instructions_revision(),
            "configured": self.path.exists() and result["ok"],
            "check": result,
            "codex_login": result["codex_login"],
        }

    def save(self, body):
        if not isinstance(body, dict) or body.get("revision") != self.revision():
            raise RuntimeError(
                "Settings changed; reopen the panel / 配置已变化，请重新打开"
            )
        data = copy.deepcopy(body.get("data"))
        if not isinstance(data, dict):
            raise ValueError("Invalid configuration")
        current = self.load()
        gemini = data.setdefault("gemini", {})
        if not isinstance(gemini, dict):
            raise ValueError("Invalid Gemini configuration")
        # A blank write-only field means retain the stored key; never return it.
        if gemini.get("api_key", "") == "":
            gemini["api_key"] = current.gemini.api_key
        setup = UserSetup(data, self.path)
        result = setup.check()
        if not result["ok"]:
            return result
        setup.workspace.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=setup.workspace):
            pass
        if self.demo_path:
            frontend_document = updated_frontend_document(self.demo_path, setup.duplex)
            original_document = self.demo_path.read_bytes()
            write_json(self.demo_path, frontend_document)
            try:
                self.save_startup(setup.data)
            except Exception:
                write_json(self.demo_path, json.loads(original_document))
                raise
        else:
            self.save_startup(setup.data)
        return result
