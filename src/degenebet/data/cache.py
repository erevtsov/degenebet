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
        cached_pulled_at: datetime | None = cached["pulled_at"].max()  # type: ignore[assignment]
        # A 0-height snapshot (e.g. a bye-week empty odds response) has no
        # rows to take a max() over, so treat it as maximally stale rather
        # than crashing on a None comparison.
        if cached_pulled_at is not None and datetime.now(UTC) - cached_pulled_at <= max_age:
            return cached

    fresh = provider.fetch_raw()
    pulled_at: datetime | None = fresh["pulled_at"].max()  # type: ignore[assignment]
    if pulled_at is None:
        pulled_at = datetime.now(UTC)
    path = _snapshot_path(source, pulled_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_parquet(path)
    return fresh


def load_all_snapshots(source: str) -> pl.DataFrame:
    """Return every retained snapshot for *source*, concatenated (each row
    keeps its own ``pulled_at``). Empty (zero rows) if nothing's cached yet.
    """
    source_dir = _source_dir(source)
    if not source_dir.exists():
        return pl.DataFrame()
    snapshots = sorted(source_dir.glob(f"{source}_*.parquet"))
    if not snapshots:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(p) for p in snapshots], how="vertical")
