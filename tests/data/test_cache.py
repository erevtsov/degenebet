from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from degenebet.data import cache


class FakeProvider:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls = 0

    def fetch_raw(self) -> pl.DataFrame:
        self.calls += 1
        return self.frame


def _frame(pulled_at: datetime) -> pl.DataFrame:
    return pl.DataFrame({"value": [1, 2], "pulled_at": [pulled_at, pulled_at]})


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEGENEBET_CACHE_DIR", str(tmp_path))


def test_load_or_fetch_calls_provider_when_no_cache() -> None:
    provider = FakeProvider(_frame(datetime.now(UTC)))
    result = cache.load_or_fetch(provider, "testsource")
    assert provider.calls == 1
    assert result.equals(provider.frame)


def test_load_or_fetch_reuses_fresh_cache() -> None:
    provider = FakeProvider(_frame(datetime.now(UTC)))
    cache.load_or_fetch(provider, "testsource")
    cache.load_or_fetch(provider, "testsource")
    assert provider.calls == 1


def test_load_or_fetch_refetches_when_stale() -> None:
    old = datetime.now(UTC) - timedelta(hours=1)
    provider = FakeProvider(_frame(old))
    cache.load_or_fetch(provider, "testsource", max_age=timedelta(minutes=15))
    provider.frame = _frame(datetime.now(UTC))
    cache.load_or_fetch(provider, "testsource", max_age=timedelta(minutes=15))
    assert provider.calls == 2


def test_load_or_fetch_force_refresh_bypasses_fresh_cache() -> None:
    provider = FakeProvider(_frame(datetime.now(UTC)))
    cache.load_or_fetch(provider, "testsource")
    cache.load_or_fetch(provider, "testsource", force_refresh=True)
    assert provider.calls == 2


def test_load_or_fetch_writes_a_parquet_snapshot(tmp_path: object) -> None:
    provider = FakeProvider(_frame(datetime.now(UTC)))
    cache.load_or_fetch(provider, "testsource")
    snapshots = list((Path(str(tmp_path)) / "testsource").glob("testsource_*.parquet"))
    assert len(snapshots) == 1
