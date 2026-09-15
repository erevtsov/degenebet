"""Snapshot parquet cache for point-in-time data sources (e.g. current odds).

Unlike a historical time-series cache, this does not gap-fill a date range —
each write is a full-replacement snapshot, timestamped by when it was pulled.
Freshness is judged from the snapshot's own ``pulled_at`` column (not file
mtime), so it stays correct even if files are copied or touched externally.

Cache layout: ``{cache_dir()}/{source}/{source}_{pulled_at}.parquet``
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from degenebet.config import cache_dir
from degenebet.data.providers.base import Provider

_TS_FORMAT = "%Y%m%dT%H%M%SZ"


def _source_dir(source: str) -> Path:
    return cache_dir() / source


def _snapshot_path(source: str, pulled_at: datetime) -> Path:
    ts = pulled_at.astimezone(UTC).strftime(_TS_FORMAT)
    return _source_dir(source) / f"{source}_{ts}.parquet"


def _latest_snapshot(source: str) -> Path | None:
    source_dir = _source_dir(source)
    if not source_dir.exists():
        return None
    snapshots = sorted(source_dir.glob(f"{source}_*.parquet"))
    return snapshots[-1] if snapshots else None


def load_or_fetch(
    provider: Provider,
    source: str,
    *,
    max_age: timedelta = timedelta(minutes=15),
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Return a cached snapshot for *source*, fetching a fresh one if needed.

    If the most recent cached snapshot's ``pulled_at`` is within *max_age*
    of now and *force_refresh* is False, returns it without calling
    ``provider.fetch_raw()``. Otherwise fetches, writes a new timestamped
    snapshot, and returns the fresh data.

    Parameters
    ----------
    provider:
        Object implementing ``Provider``.
    source:
        Cache namespace, e.g. ``"sharpapi"``.
    max_age:
        How old a cached snapshot may be before it's considered stale.
    force_refresh:
        If True, always fetch fresh data regardless of cache age.
    """
    latest_path = None if force_refresh else _latest_snapshot(source)

    if latest_path is not None:
        cached = pl.read_parquet(latest_path)
        pulled_at: datetime = cached["pulled_at"].max()  # type: ignore[assignment]
        if datetime.now(UTC) - pulled_at <= max_age:
            return cached

    fresh = provider.fetch_raw()
    pulled_at = fresh["pulled_at"].max()  # type: ignore[assignment]
    path = _snapshot_path(source, pulled_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_parquet(path)
    return fresh
