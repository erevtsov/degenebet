"""Gameweek: the (season, week) granularity DataAccess indexes "which
games" queries by, instead of a raw calendar-date range. See
docs/superpowers/specs/2026-09-18-data-access-redesign.md, Decision 4 --
NFL's Thursday/Sunday/Monday week structure makes date-range filtering a
footgun (a naively-computed range can silently drop the Thursday or Monday
game of a week), and nflreadpy's own season/week columns already group
every game correctly regardless of which day it's played.
"""

from __future__ import annotations

from typing import NamedTuple


class Gameweek(NamedTuple):
    """A single NFL week within a season. Sorts correctly as a plain tuple
    (season compares first, then week) -- as_int() exists only for
    polars' convenience as a single sortable column. Its formula
    (season * 100 + week) must stay identical to the `gameweek` column
    nflverse_cache.py's sync_schedules/sync_team_stats compute -- see this
    plan's Global Constraints.
    """

    season: int
    week: int

    def as_int(self) -> int:
        return self.season * 100 + self.week
