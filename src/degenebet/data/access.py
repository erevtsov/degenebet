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

import warnings
from datetime import date
from typing import Protocol

import polars as pl

from degenebet.data import cache, nflverse_cache, teams
from degenebet.data.gameweek import Gameweek

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


class HistoricalDataSource(Protocol):
    def fetch(self, start_week: Gameweek, end_week: Gameweek) -> pl.DataFrame: ...


class CurrentDataSource(Protocol):
    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame: ...


class SharpApiSource:
    """Current-odds source, implements CurrentDataSource: one row per
    (home_team, away_team, gameday, pulled_at) with `spread_line` in
    nflreadpy's sign convention (positive = home favored) -- the median
    main-line home spread across sportsbooks *within* a single snapshot.
    Snapshots are never collapsed across `pulled_at` here: DataAccess.get_team_data
    is what picks the as-of-correct snapshot per game, since only it knows
    the query's as_of_date."""

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
            # event_start_time is UTC; nflreadpy's gameday is the game's
            # Eastern-local calendar date, so an evening kickoff (~20% of
            # the weekly slate) needs the timezone conversion before
            # slicing off the date, or it lands on the wrong gameday and
            # silently misses the join in DataAccess.get_team_data.
            pl.col("event_start_time")
            .str.to_datetime(time_zone="UTC")
            .dt.convert_time_zone("America/New_York")
            .dt.date()
            .cast(pl.Utf8)
            .alias("gameday"),
        )

        in_range = home_spreads.filter(
            (pl.col("gameday") >= start_date.isoformat())
            & (pl.col("gameday") <= end_date.isoformat())
        )

        return in_range.group_by(["home_team", "away_team", "gameday", "pulled_at"]).agg(
            pl.col("spread_line").median()
        )


class NflverseSource:
    """Historical schedule source: reads the persisted merged schedules
    store (nflverse_cache.py), never the network. Team codes and the
    `gameweek` column are already normalized at the merge-cache write
    boundary (nflverse_cache.sync_schedules) -- this is a pure read, no
    transformation of its own."""

    def fetch(self, start_week: Gameweek, end_week: Gameweek) -> pl.DataFrame:
        merged = nflverse_cache.read_merged("schedules")
        if merged is None:
            raise RuntimeError(
                "No schedules have ever been synced -- run `degenebet fetch schedules` "
                "or `degenebet sync` first."
            )
        return merged.filter(
            (pl.col("gameweek") >= start_week.as_int()) & (pl.col("gameweek") <= end_week.as_int())
        )


class DataAccess:
    """Stitches historical and current schedule sources into one
    point-in-time-correct view. See
    docs/superpowers/specs/2026-09-17-data-access-design.md."""

    def __init__(self, historical: HistoricalDataSource, current: CurrentDataSource) -> None:
        self.historical = historical
        self.current = current

    def get_team_data(self, start_date: date, end_date: date, as_of_date: date) -> pl.DataFrame:
        historical = self.historical.fetch(start_date, end_date)
        current = self.current.fetch(start_date, end_date)

        if historical.height > 0:
            earliest = str(historical["gameday"].min())
            if as_of_date.isoformat() < earliest:
                # Informational only -- do NOT clamp as_of_date forward.
                # Clamping would admit SharpAPI snapshots pulled after the
                # true requested as_of_date (a look-ahead leak); the honest
                # behavior for "as_of_date predates any cached history" is
                # to proceed with the original as_of_date, which naturally
                # yields an empty/limited `current` contribution below.
                warnings.warn(
                    f"as_of_date {as_of_date} predates earliest cached history {earliest}.",
                    stacklevel=2,
                )

        current_asof = (
            current.filter(pl.col("pulled_at").dt.date() <= as_of_date)
            if current.height > 0
            else current
        )
        if current_asof.height > 0:
            # Multiple snapshots may survive the as-of cutoff for the same
            # game (SharpAPI is polled repeatedly); keep only the one
            # closest to (but not after) as_of_date -- the latest pulled_at
            # per game among those already filtered above.
            current_asof = (
                current_asof.sort("pulled_at")
                .group_by(["home_team", "away_team", "gameday"], maintain_order=True)
                .last()
            )

        unresolved = historical.filter(pl.col("result").is_null()).select(
            "game_id", "home_team", "away_team", "gameday"
        )
        current_for_unresolved = current_asof.join(
            unresolved, on=["home_team", "away_team", "gameday"], how="inner"
        ).select("game_id", pl.col("spread_line").alias("current_spread_line"))

        return (
            historical.join(current_for_unresolved, on="game_id", how="left")
            .with_columns(pl.coalesce(["current_spread_line", "spread_line"]).alias("spread_line"))
            .drop("current_spread_line")
        )
