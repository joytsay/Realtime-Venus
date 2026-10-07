"""User-authored conversation instructions stored verbatim in prompt.csv."""

import hashlib
import os
import tempfile
from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt.csv"
MAX_PROMPT_BYTES = 64 * 1024


def read_instructions():
    if not PROMPT_PATH.exists():
        return ""
    data = PROMPT_PATH.read_bytes()
    if len(data) > MAX_PROMPT_BYTES:
        raise ValueError("prompt.csv must be smaller than 64 KiB")
    return data.decode("utf-8")


def instructions_revision():
    return hashlib.sha256(read_instructions().encode("utf-8")).hexdigest()


def save_instructions(text, revision):
    if revision != instructions_revision():
        raise RuntimeError("prompt.csv changed; reopen Settings before saving")
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError("Instructions must be text smaller than 64 KiB")
    descriptor, temporary = tempfile.mkstemp(prefix="venus-prompt-", dir=PROMPT_PATH.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, PROMPT_PATH)
    finally:
        Path(temporary).unlink(missing_ok=True)
