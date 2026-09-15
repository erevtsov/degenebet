from __future__ import annotations

from pathlib import Path

import pytest

from degenebet import config


def test_cache_dir_defaults_to_home_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEGENEBET_CACHE_DIR", raising=False)
    assert config.cache_dir() == Path.home() / ".degenebet" / "cache"


def test_cache_dir_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEGENEBET_CACHE_DIR", "/tmp/custom-cache")
    assert config.cache_dir() == Path("/tmp/custom-cache")


def test_sharpapi_key_returns_value_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARPAPI_KEY", "abc123")
    assert config.sharpapi_key() == "abc123"


def test_sharpapi_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHARPAPI_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SHARPAPI_KEY"):
        config.sharpapi_key()
