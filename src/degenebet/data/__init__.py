"""Public data-loading API for degenebet.

Historical NFL data (schedules, stats, rosters) is unaffected by the cache
in this package — ``nflreadpy`` handles its own filesystem cache. Only the
current-odds pull goes through ``data/cache.py``, since it's a point-in-time
snapshot rather than a stable historical series.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from degenebet.data import cache, nflverse
from degenebet.data.providers.sharpapi import SharpAPIProvider

load_schedules = nflverse.load_schedules
load_player_stats = nflverse.load_player_stats
load_team_stats = nflverse.load_team_stats
load_rosters = nflverse.load_rosters


def load_odds(
    *,
    max_age: timedelta = timedelta(minutes=15),
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Return current NFL odds (moneyline, spread, total), using the cache.

    See ``degenebet.data.cache.load_or_fetch`` for freshness semantics.
    """
    return cache.load_or_fetch(
        SharpAPIProvider(), "sharpapi", max_age=max_age, force_refresh=force_refresh
    )


__all__ = [
    "load_schedules",
    "load_player_stats",
    "load_team_stats",
    "load_rosters",
    "load_odds",
]
