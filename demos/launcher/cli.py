"""One-command launch, health checks and shutdown for Realtime-Venus."""

from __future__ import annotations

import fcntl
import getpass
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import (
    LaunchConfig,
    checked_file,
    ensure_ports_available,
)
from .supervisor import StackSupervisor, read_state
from .variants import select_checkpoint
from .options import parse_launch_options
from demos.variants import model_name


def setup_backend(config: LaunchConfig, allow_login: bool) -> None:
    from demos.settings import UserSetup
    from demos.server.configuration import WebConfiguration

    configuration = WebConfiguration(config.settings, demo_path=getattr(config, "demo_path", None))
    setup = configuration.load()
    settings_changed = False
    if not config.settings.exists():
        setup.data["workspace"] = ""
        settings_changed = True

    if setup.uses_codex:
        configured_binary = setup.general.command[0]
        candidates = (
            os.getenv("GENERAL_CODEX_BINARY"),
            configured_binary,
            shutil.which("codex"),
            str(config.runtime / "bin/codex"),
        )
        binary = next(
            (resolved for candidate in candidates if candidate and (resolved := shutil.which(candidate))),
            None,
        )
        if not binary:
            raise RuntimeError("Codex executable is missing. Run bash install.sh.")
        if binary != configured_binary:
            setup.data["codex"]["command"] = [binary, *setup.general.command[1:]]
            settings_changed = True

        if settings_changed:
            # Reconstruct derived settings after filling or repairing startup defaults.
            setup = UserSetup(setup.data, config.settings)
        check = setup.check(web=False)
        if not check["ok"]:
            raise RuntimeError("; ".join(check["problems"]))
        binary = setup.general.command[0]
        status = subprocess.run(
            [binary, "login", "status"], capture_output=True, timeout=30
        )
        if status.returncode:
            if not allow_login or not sys.stdin.isatty():
                raise RuntimeError(
                    f"Codex login required. Run {binary} login --device-auth, then retry."
                )
            print("Complete Codex device login in your local browser.", flush=True)
            subprocess.run([binary, "login", "--device-auth"], check=True)
    else:
        check = setup.check(web=False)
        if not check["ok"]:
            raise RuntimeError("; ".join(check["problems"]))
    if "llamacpp" in setup.active_providers:
        from harness.llm.llamacpp import check_server
        status = check_server(setup.llamacpp)
        if status["state"] != "ready":
            raise RuntimeError(status["message"])
        print(status["message"], flush=True)
    if setup.data["workspace"]:
        setup.workspace.mkdir(parents=True, exist_ok=True)
    if settings_changed:
        configuration.save_startup(setup.data)


def show_access(config: LaunchConfig) -> None:
    print(
        f"\n{model_name(config.model_type)} API: http://127.0.0.1:{config.model_port}", flush=True
    )
    print(f"Realtime-Venus-Harness: http://localhost:{config.web_port}", flush=True)
    ssh = os.getenv("SSH_CONNECTION", "").split()
    server = ssh[2] if len(ssh) == 4 else "<server>"
    print("From your local computer, keep this SSH tunnel open:", flush=True)
    print(
        f"  ssh -N -L {config.web_port}:127.0.0.1:{config.web_port} {getpass.getuser()}@{server}",
        flush=True,
    )
    print("Choose microphone or audio upload." if config.model_type == "audio" else "Choose camera with microphone or video upload.", flush=True)
    print(f"Logs: {config.runtime}/logs; stop: bash start.sh --stop", flush=True)


def _run(argv=None):
    root = Path(os.getenv("VENUS_ROOT", Path.cwd())).expanduser().resolve()
    parser, args, options, launch_argv = parse_launch_options(root, argv)
    runtime = root / "runtime"
    runtime.mkdir(exist_ok=True)
    state = read_state(runtime / "stack.json")
    if args.status:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    if args.stop:
        if state["alive"]:
            os.kill(state["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 50
            while (
                read_state(runtime / "stack.json")["alive"]
                and time.monotonic() < deadline
            ):
                time.sleep(0.2)
            if read_state(runtime / "stack.json")["alive"]:
                raise RuntimeError(
                    "Shutdown is still in progress; inspect runtime/logs before retrying."
                )
        print("Realtime-Venus stack stopped.")
        return
    if state["alive"]:
        parser.error(
            "This checkout already owns a running stack. Use --status or --stop."
        )
    model = select_checkpoint(
        root, args.model_type,
        explicit=args.model_path,
        configured=None,
    )
    ref = (
        checked_file(root / Path(args.ref_audio))
        if args.ref_audio
        else checked_file(model / "assets/HT_ref_audio.wav")
    )
    config = LaunchConfig(
        root,
        model,
        ref,
        os.getenv("VENUS_MODEL_PYTHON", sys.executable),
        sys.executable,
        args.host,
        args.model_port,
        args.web_port,
        args.memory_minutes,
        args.startup_timeout,
        args.model_type,
        harness_path=Path(args.harness_config) if args.harness_config else None,
        demo_path=args.demo_config,
    )
    if config.memory_minutes <= 0 or config.startup_timeout <= 0:
        parser.error("Memory duration and startup timeout must be positive")
    ensure_ports_available("127.0.0.1", (config.model_port, config.web_port))
    if config.host != "127.0.0.1":
        ensure_ports_available(config.host, (config.web_port,))
    subprocess.run(
        [
            config.model_python,
            "-c",
            'import torch; assert torch.cuda.is_available(), "No CUDA GPU available"; print("GPU:", torch.cuda.get_device_name(0))',
        ],
        check=True,
        timeout=60,
    )
    setup_backend(config, allow_login=not args.no_login)
    if args.check:
        print("Checkpoint, speech assets, ports, CUDA and backend: OK")
        return
    if args.detach:
        child_args = [
            a for a in launch_argv if a != "--detach"
        ]
        (runtime / "logs").mkdir(exist_ok=True)
        with (runtime / "logs/stack.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(
                [sys.executable, "-m", "demos", *child_args, "--no-login"],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + config.startup_timeout * 2 + 90
        try:
            while child.poll() is None and time.monotonic() < deadline:
                current = read_state(runtime / "stack.json")
                if current.get("pid") == child.pid and current.get("status") == "ready":
                    show_access(config)
                    return
                time.sleep(0.25)
            raise RuntimeError("Stack startup failed; see runtime/logs/stack.log")
        except BaseException:
            child.terminate()
            child.wait(timeout=50)
            raise
    with (runtime / "stack.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another launcher owns this checkout")
        model_command = [
            config.model_python,
            "-m",
            "demos.model.server",
            "--model-type",
            config.model_type,
            "--model-path",
            str(model),
            "--ref-audio",
            str(ref),
            "--memory-minutes",
            str(config.memory_minutes),
            "--host",
            "127.0.0.1",
            "--port",
            str(config.model_port),
        ]
        web_command = [
            config.web_python,
            "-m",
            "demos.server.cli",
            "--model-type",
            config.model_type,
            "--model-server",
            f"http://127.0.0.1:{config.model_port}",
            "--tokenizer-path",
            str(model),
            "--config",
            str(config.settings),
            "--host",
            config.host,
            "--port",
            str(config.web_port),
        ]
        if config.demo_path:
            web_command += ["--demo-config", str(config.demo_path)]
        print(
            f"Loading {model_name(config.model_type)}; the web application starts after the model is ready.",
            flush=True,
        )
        StackSupervisor(config, model_command, web_command).run(
            lambda: show_access(config)
        )


def main(argv=None):
    try:
        return _run(argv)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
