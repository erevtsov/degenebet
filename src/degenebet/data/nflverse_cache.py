"""Merge/upsert cache for nflreadpy-derived tables (schedules, team_stats).

Unlike data/cache.py's append-only SharpAPI snapshots, this cache is a
single persisted, monotonically-growing store per table: each fetch is
merged into what's already there rather than trusting the raw fetch as a
full replacement. See docs/superpowers/specs/2026-09-17-data-access-design.md
("Storage semantics") for why -- a naive overwrite would silently commit
data loss if the upstream API ever starts limiting the historical range it
returns. The persisted store is always in canonical form: team codes are
lowercased and a `gameweek` int column (season * 100 + week, matching
Gameweek.as_int()) is added at write time, so readers never have to
normalize either.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from degenebet.config import cache_dir
from degenebet.data import nflverse

_MERGED_DIRNAME = "merged"


def merge_frames(
    existing: pl.DataFrame | None, new: pl.DataFrame, key_columns: list[str]
) -> pl.DataFrame:
    """Union `new` into `existing`, keyed by `key_columns`.

    A key present in both takes `new`'s row (so legitimate corrections
    propagate); a key present only in `existing` survives even if `new`
    doesn't have it (so a `new` fetch covering less history than before
    can't silently erase rows already cached).
    """
    if existing is None or existing.height == 0:
        return new
    carried_over = existing.join(new.select(key_columns), on=key_columns, how="anti")
    return pl.concat([new, carried_over.select(new.columns)], how="vertical")


def _merged_path(name: str) -> Path:
    return cache_dir() / _MERGED_DIRNAME / f"{name}.parquet"


def read_merged(name: str) -> pl.DataFrame | None:
    """Return the persisted merged store for `name`, or None if it's never
    been synced."""
    path = _merged_path(name)
    return pl.read_parquet(path) if path.exists() else None


def load_or_merge(
    new: pl.DataFrame, *, name: str, key_columns: list[str], group_column: str
) -> pl.DataFrame:
    """Merge `new` into the persisted store at cache_dir()/merged/{name}.parquet,
    write the result back, and return it.

    Raises ValueError if the merge would drop `group_column`'s row count for
    any group already present in the existing store -- defense in depth on
    top of merge_frames's own no-loss guarantee, to surface a shrinking raw
    fetch rather than let it pass unnoticed.
    """
    existing = read_merged(name)
    if existing is not None and existing.height > 0:
        missing = set(new.columns) - set(existing.columns)
        if missing:
            raise ValueError(
                f"The persisted '{name}' store's schema predates a schema change "
                f"(missing columns: {sorted(missing)}) -- delete "
                f"cache_dir()/merged/{name}.parquet and re-sync."
            )

    merged = merge_frames(existing, new, key_columns)

    if existing is not None and existing.height > 0:
        old_counts = existing.group_by(group_column).len()
        new_counts = merged.group_by(group_column).len()
        comparison = old_counts.join(new_counts, on=group_column, how="left", suffix="_new")
        shrunk = comparison.filter(
            pl.col("len_new").is_null() | (pl.col("len_new") < pl.col("len"))
        )
        if shrunk.height > 0:
            groups = shrunk[group_column].to_list()
            raise ValueError(
                f"Merging '{name}' would shrink {group_column}(s) {groups} -- "
                "refusing to write; investigate the upstream fetch before retrying."
            )

    path = _merged_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(path)
    return merged


def sync_schedules(seasons: list[int] | None = None) -> pl.DataFrame:
    """Fetch schedules from nflreadpy, normalize team codes to lowercase,
    add the `gameweek` sort/filter column, and merge into the persisted
    store."""
    fresh = nflverse.load_schedules(seasons).with_columns(
        pl.col("home_team").str.to_lowercase(),
        pl.col("away_team").str.to_lowercase(),
        (pl.col("season") * 100 + pl.col("week")).alias("gameweek"),
    )
    return load_or_merge(fresh, name="schedules", key_columns=["game_id"], group_column="season")


def sync_team_stats(seasons: list[int] | None = None) -> pl.DataFrame:
    """Fetch team stats from nflreadpy, normalize team codes to lowercase,
    add the `gameweek` sort/filter column, and merge into the persisted
    store."""
    fresh = nflverse.load_team_stats(seasons).with_columns(
        pl.col("team").str.to_lowercase(),
        pl.col("opponent_team").str.to_lowercase(),
        (pl.col("season") * 100 + pl.col("week")).alias("gameweek"),
    )
    return load_or_merge(
        fresh, name="team_stats", key_columns=["game_id", "team"], group_column="season"
    )
