"""Own two child processes and clean up both on failure or shutdown."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import LaunchConfig


def process_start(pid: int) -> str | None:
    """Linux process start ticks prevent acting on a recycled PID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def read_state(path: Path) -> dict:
    try:
        state = json.loads(path.read_text())
        state["alive"] = (
            bool(state.get("start_ticks"))
            and process_start(state["pid"]) == state["start_ticks"]
        )
        return state
    except (OSError, ValueError, KeyError, TypeError):
        return {"alive": False, "status": "stopped"}


def write_state(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.chmod(0o600)
    temporary.replace(path)


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            body = json.load(response)
            return response.status == 200 and (
                body.get("ready") is True or body.get("status") == "ok"
            )
    except (OSError, ValueError, urllib.error.URLError):
        return False


class StackSupervisor:
    def __init__(
        self, config: LaunchConfig, model_command: list[str], web_command: list[str]
    ):
        self.config = config
        self.commands = {"model": model_command, "web": web_command}
        self.log_stdout = os.getenv("VENUS_LOG_STDOUT") == "1"
        self.children: dict[str, subprocess.Popen] = {}
        self.stopping = False
        self.state = {
            "pid": os.getpid(),
            "start_ticks": process_start(os.getpid()),
            "status": "starting",
            "model_type": config.model_type,
            "model_path": str(config.model_path),
            "model_port": config.model_port,
            "web_port": config.web_port,
        }

    def spawn(self, name: str) -> None:
        env = dict(os.environ)
        env["PATH"] = (
            str(self.config.runtime / "bin") + os.pathsep + env.get("PATH", "")
        )
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("MAX_NUM_FRAMES", "100000")
        if self.log_stdout:
            env["PYTHONUNBUFFERED"] = "1"
        output = (
            contextlib.nullcontext(None)
            if self.log_stdout
            else (self.config.runtime / "logs" / f"{name}.log").open("ab", buffering=0)
        )
        with output as log:
            child = subprocess.Popen(
                self.commands[name],
                cwd=self.config.root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.children[name] = child
        self.state[f"{name}_pid"] = child.pid
        self.save()

    def save(self) -> None:
        write_state(self.config.runtime / "stack.json", self.state)

    def check_children(self) -> None:
        for name, child in self.children.items():
            if child.poll() is not None:
                location = (
                    "see service output above"
                    if self.log_stdout
                    else f"see runtime/logs/{name}.log"
                )
                raise RuntimeError(
                    f"{name} exited with code {child.returncode}; {location}"
                )

    def wait_ready(self, url: str) -> None:
        deadline = time.monotonic() + self.config.startup_timeout
        while not self.stopping:
            self.check_children()
            if healthy(url):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Service startup timed out: {url}")
            time.sleep(0.25)
        raise InterruptedError("Startup cancelled")

    def shutdown(self) -> None:
        for child in reversed(list(self.children.values())):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)

    def run(self, ready_callback=None) -> None:
        def stop(_signal, _frame):
            self.stopping = True

        previous = {
            sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)
        }
        (self.config.runtime / "logs").mkdir(parents=True, exist_ok=True)
        self.save()
        try:
            self.spawn("model")
            self.wait_ready(f"http://127.0.0.1:{self.config.model_port}/healthz")
            self.spawn("web")
            self.wait_ready(f"http://127.0.0.1:{self.config.web_port}/health")
            self.state["status"] = "ready"
            self.save()
            if ready_callback:
                ready_callback()
            while not self.stopping:
                self.check_children()
                time.sleep(0.25)
        except BaseException as exc:
            self.state["error"] = str(exc)
            if not isinstance(exc, InterruptedError):
                raise
        finally:
            self.shutdown()
            self.state["status"] = "stopped"
            self.save()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
