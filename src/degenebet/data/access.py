"""DataAccess: point-in-time stitching of historical (nflreadpy) and current
(SharpAPI) schedule data. See
docs/superpowers/specs/2026-09-17-data-access-design.md.

Cross-vendor join contract: every DataSource implementation must return
`home_team`/`away_team` as codes from `teams.CANONICAL_TEAMS`, `gameday` as
an ISO 8601 date string (YYYY-MM-DD), and `spread_line` (where present) with
positive meaning home favored. These three are the actual join/comparison
surface DataAccess relies on to stitch sources together -- the rest of each
source's columns can differ; a historical source and an odds source are
fundamentally different shapes, and full column parity between them isn't
useful or required.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import polars as pl

from degenebet.data import cache, teams

# SharpAPI's real full-name format, confirmed against live data 2026-09-17
# (e.g. "Buffalo Bills", "Chicago Bears") -- not the abbreviated-city guess
# in the provider's own test fixtures, which was never verified live.
# Validated against teams.CANONICAL_TEAMS below, not just by this module's
# own tests happening to exercise every team.
_SHARPAPI_TEAM_CROSSWALK: dict[str, str] = {
    "Arizona Cardinals": "ari",
    "Atlanta Falcons": "atl",
    "Baltimore Ravens": "bal",
    "Buffalo Bills": "buf",
    "Carolina Panthers": "car",
    "Chicago Bears": "chi",
    "Cincinnati Bengals": "cin",
    "Cleveland Browns": "cle",
    "Dallas Cowboys": "dal",
    "Denver Broncos": "den",
    "Detroit Lions": "det",
    "Green Bay Packers": "gb",
    "Houston Texans": "hou",
    "Indianapolis Colts": "ind",
    "Jacksonville Jaguars": "jax",
    "Kansas City Chiefs": "kc",
    "Las Vegas Raiders": "lv",
    "Los Angeles Chargers": "lac",
    "Los Angeles Rams": "la",
    "Miami Dolphins": "mia",
    "Minnesota Vikings": "min",
    "New England Patriots": "ne",
    "New Orleans Saints": "no",
    "New York Giants": "nyg",
    "New York Jets": "nyj",
    "Philadelphia Eagles": "phi",
    "Pittsburgh Steelers": "pit",
    "San Francisco 49ers": "sf",
    "Seattle Seahawks": "sea",
    "Tampa Bay Buccaneers": "tb",
    "Tennessee Titans": "ten",
    "Washington Commanders": "was",
}

teams.assert_maps_to_canonical_teams(_SHARPAPI_TEAM_CROSSWALK)


class DataSource(Protocol):
    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame: ...


class SharpApiSource:
    """Current-odds source: one row per (home_team, away_team, gameday) with
    `spread_line` in nflreadpy's sign convention (positive = home favored),
    the median main-line home spread across sportsbooks and retained
    snapshots as of the latest snapshot in cache."""

    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        raw = cache.load_all_snapshots("sharpapi")
        if raw.height == 0:
            return pl.DataFrame(
                schema={
                    "home_team": pl.Utf8,
                    "away_team": pl.Utf8,
                    "gameday": pl.Utf8,
                    "spread_line": pl.Float64,
                    "pulled_at": pl.Datetime(time_zone="UTC"),
                }
            )

        home_spreads = raw.filter(
            (pl.col("market_type") == "spread") & (pl.col("selection_type") == "home")
        ).with_columns(
            pl.col("home_team").replace_strict(_SHARPAPI_TEAM_CROSSWALK),
            pl.col("away_team").replace_strict(_SHARPAPI_TEAM_CROSSWALK),
            (-pl.col("line")).alias("spread_line"),
            pl.col("event_start_time").str.slice(0, 10).alias("gameday"),
        )

        in_range = home_spreads.filter(
            (pl.col("gameday") >= start_date.isoformat())
            & (pl.col("gameday") <= end_date.isoformat())
        )

        return in_range.group_by(["home_team", "away_team", "gameday"]).agg(
            pl.col("spread_line").median(), pl.col("pulled_at").max()
        )
