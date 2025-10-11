from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_file(path: str | None = None, override: bool = True) -> None:
    """
    Minimal .env loader without external deps.

    - Reads KEY=VALUE pairs from .env in project root (default) or given path.
    - Ignores blank lines and lines starting with '#'.
    - Strips surrounding quotes from values.
    - If override=False, keeps existing env vars untouched.
    """
    env_path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not override and key in os.environ:
            continue
        os.environ[key] = val