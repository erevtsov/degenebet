"""Environment-based settings: cache directory, SharpAPI key."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)  # shell env vars take precedence over .env

_DEFAULT_CACHE_ROOT = Path.home() / ".degenebet" / "cache"


def cache_dir() -> Path:
    """Return the local cache root, overridable via ``DEGENEBET_CACHE_DIR``."""
    env = os.environ.get("DEGENEBET_CACHE_DIR")
    return Path(env) if env else _DEFAULT_CACHE_ROOT


def sharpapi_key() -> str:
    """Return the SharpAPI key from the environment.

    Raises
    ------
    RuntimeError
        If ``SHARPAPI_KEY`` is not set.
    """
    key = os.environ.get("SHARPAPI_KEY")
    if not key:
        raise RuntimeError(
            "SHARPAPI_KEY is not set. Add it to .env or export it in your shell."
        )
    return key
