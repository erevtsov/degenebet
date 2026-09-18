"""DataAccess: point-in-time stitching of historical (nflreadpy) and current
(SharpAPI) schedule data. See
docs/superpowers/specs/2026-09-18-data-access-redesign.md.

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

# get_team_data's own columns that are just a team-indexed restatement of
# information get_game_data's base table already has (season/week/gameweek/
# gameday, the spread/result pair re-signed per team, home/away, opponent).
# _widen_with_team_data excludes these so widening never produces columns
# like home_team_spread_line duplicating spread_line.
_TEAM_DATA_CONTEXT_COLUMNS = frozenset(
    {
        "game_id",
        "team",
        "season",
        "week",
        "gameweek",
        "gameday",
        "opponent",
        "is_home",
        "team_spread_line",
        "team_margin",
        "opponent_team",
    }
)


class HistoricalDataSource(Protocol):
    def fetch(self, start_week: Gameweek, end_week: Gameweek) -> pl.DataFrame: ...


class CurrentDataSource(Protocol):
    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame: ...


class SharpApiSource:
    """Current-odds source, implements CurrentDataSource: one row per
    (home_team, away_team, gameday, pulled_at) with `spread_line` in
    nflreadpy's sign convention (positive = home favored) -- the median
    main-line home spread across sportsbooks *within* a single snapshot.
    Snapshots are never collapsed across `pulled_at` here:
    DataAccess._get_stitched_schedule is what picks the as-of-correct
    snapshot per game, since only it knows the query's as_of_date."""

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
            # silently misses the join in DataAccess._get_stitched_schedule.
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
    """Fetches NFL data at whatever granularity a caller needs -- see
    docs/superpowers/specs/2026-09-18-data-access-redesign.md."""

    def __init__(self, historical: HistoricalDataSource, current: CurrentDataSource) -> None:
        self.historical = historical
        self.current = current

    def _get_stitched_schedule(
        self, start_week: Gameweek, end_week: Gameweek, as_of_date: date
    ) -> pl.DataFrame:
        """Game-indexed, point-in-time-correct: historical's own spread_line
        for a settled game; current's (SharpAPI's) for an unplayed one,
        whenever current covers it, even if historical already has an early
        line for that same game; historical's line as the fallback when
        current doesn't cover it. Shared by get_game_data and
        get_team_data."""
        historical = self.historical.fetch(start_week, end_week)
        if historical.height == 0:
            return historical

        earliest = str(historical["gameday"].min())

        # The warning compares against the full persisted store's true
        # earliest gameday, not `historical`'s own min -- `historical` is
        # narrowed to the requested Gameweek range, so its min is just
        # "earliest date in the games asked about." Querying an upcoming
        # week (this system's central use case) would otherwise always
        # false-positive, since today's date naturally precedes an upcoming
        # game's date. Falls back to `historical`'s own min if the full
        # store can't be read, so behavior degrades sensibly rather than
        # crashing.
        full_store = nflverse_cache.read_merged("schedules")
        true_earliest = (
            str(full_store["gameday"].min())
            if full_store is not None and full_store.height > 0
            else earliest
        )
        if as_of_date.isoformat() < true_earliest:
            # Informational only -- do NOT clamp as_of_date forward.
            # Clamping would admit current snapshots pulled after the true
            # requested as_of_date (a look-ahead leak); the honest behavior
            # for "as_of_date predates any cached history" is to proceed
            # with the original as_of_date.
            warnings.warn(
                f"as_of_date {as_of_date} predates earliest cached history {true_earliest}.",
                stacklevel=2,
            )

        min_date = date.fromisoformat(earliest)
        max_date = date.fromisoformat(str(historical["gameday"].max()))
        current = self.current.fetch(min_date, max_date)

        current_asof = (
            current.filter(pl.col("pulled_at").dt.date() <= as_of_date)
            if current.height > 0
            else current
        )

        if current_asof.height > 0:
            # current has no season/week of its own -- resolve each row's
            # (season, week) by matching (home_team, away_team) against
            # historical and taking the closest gameday, disambiguating the
            # rare case where a Gameweek range spans multiple seasons and
            # the same team pairing appears more than once.
            historical_games = historical.select(
                "home_team",
                "away_team",
                pl.col("gameday").alias("historical_gameday"),
                "season",
                "week",
            )
            current_asof = (
                current_asof.join(historical_games, on=["home_team", "away_team"], how="inner")
                .with_columns(
                    (pl.col("gameday").str.to_date() - pl.col("historical_gameday").str.to_date())
                    .dt.total_days()
                    .abs()
                    .alias("_date_distance")
                )
                .sort("_date_distance")
                .group_by(["home_team", "away_team", "gameday", "pulled_at"], maintain_order=True)
                .first()
                .drop("historical_gameday", "_date_distance")
            )
            # Multiple snapshots may survive the as-of cutoff for the same
            # game; keep only the one closest to (but not after) as_of_date.
            current_asof = (
                current_asof.sort("pulled_at")
                .group_by(["season", "week", "home_team", "away_team"], maintain_order=True)
                .last()
            )

        unresolved = historical.filter(pl.col("result").is_null()).select(
            "game_id", "home_team", "away_team", "season", "week"
        )
        if current_asof.height > 0:
            current_for_unresolved = current_asof.join(
                unresolved, on=["home_team", "away_team", "season", "week"], how="inner"
            ).select("game_id", pl.col("spread_line").alias("current_spread_line"))
        else:
            current_for_unresolved = pl.DataFrame(
                schema={"game_id": pl.Utf8, "current_spread_line": pl.Float64}
            )

        return (
            historical.join(current_for_unresolved, on="game_id", how="left")
            .with_columns(pl.coalesce(["current_spread_line", "spread_line"]).alias("spread_line"))
            .drop("current_spread_line")
        )

    def _to_team_indexed(self, game_table: pl.DataFrame) -> pl.DataFrame:
        """One game row -> two team rows (home's perspective, away's),
        team_spread_line/team_margin re-signed per team."""
        # No empty-frame early return: the .select(...) chain below produces
        # the correct team-indexed schema even on a zero-row input, and an
        # early return of `game_table` as-is would leak the game-indexed
        # schema (home_team/away_team, no team/opponent/is_home) instead.
        # Cast before negating: a table where every row's result/spread_line
        # is null (e.g. an unplayed game, as in a single-row test fixture)
        # infers a Null dtype from nflreadpy/polars, and `neg` isn't defined
        # for Null; casting to Float64 first keeps both sides' dtypes
        # identical too, which pl.concat(how="vertical") requires.
        home_side = game_table.select(
            "game_id",
            "season",
            "week",
            "gameweek",
            "gameday",
            pl.col("home_team").alias("team"),
            pl.col("away_team").alias("opponent"),
            pl.lit(True).alias("is_home"),
            pl.col("spread_line").cast(pl.Float64).alias("team_spread_line"),
            pl.col("result").cast(pl.Float64).alias("team_margin"),
        )
        away_side = game_table.select(
            "game_id",
            "season",
            "week",
            "gameweek",
            "gameday",
            pl.col("away_team").alias("team"),
            pl.col("home_team").alias("opponent"),
            pl.lit(False).alias("is_home"),
            (-pl.col("spread_line").cast(pl.Float64)).alias("team_spread_line"),
            (-pl.col("result").cast(pl.Float64)).alias("team_margin"),
        )
        return pl.concat([home_side, away_side], how="vertical")

    def _widen_with_team_data(
        self, game_table: pl.DataFrame, team_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Join `team_data` (keyed on game_id, team) onto `game_table` twice
        -- home perspective and away perspective -- prefixing every
        non-context column home_/away_. Generic: works whether `team_data`
        is get_team_data()'s own output or compute_rolling_features()'s."""
        payload_columns = [c for c in team_data.columns if c not in _TEAM_DATA_CONTEXT_COLUMNS]
        home_payload = team_data.select(
            "game_id",
            "team",
            *[pl.col(c).alias(f"home_{c}") for c in payload_columns],
        )
        away_payload = team_data.select(
            "game_id",
            "team",
            *[pl.col(c).alias(f"away_{c}") for c in payload_columns],
        )
        return game_table.join(
            home_payload,
            left_on=["game_id", "home_team"],
            right_on=["game_id", "team"],
            how="left",
        ).join(
            away_payload,
            left_on=["game_id", "away_team"],
            right_on=["game_id", "team"],
            how="left",
        )

    def get_game_data(
        self,
        start_week: Gameweek,
        end_week: Gameweek,
        as_of_date: date,
        *,
        team_data: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """One row per game: game_id, season, week, gameweek, gameday,
        home_team, away_team, result, spread_line. If `team_data` is given
        (any team-indexed table keyed on game_id, team), its payload
        columns are joined in twice, prefixed home_/away_."""
        game_table = self._get_stitched_schedule(start_week, end_week, as_of_date)
        if team_data is None or game_table.height == 0:
            return game_table
        return self._widen_with_team_data(game_table, team_data)

    def get_team_data(
        self, start_week: Gameweek, end_week: Gameweek, as_of_date: date
    ) -> pl.DataFrame:
        """One row per (game_id, team): schedule/spread context from that
        team's own perspective, plus the full raw team_stats row for
        (game_id, team) left-joined -- null for a game with no stats yet
        (it hasn't been played)."""
        game_table = self._get_stitched_schedule(start_week, end_week, as_of_date)
        long_table = self._to_team_indexed(game_table)
        team_stats = nflverse_cache.read_merged("team_stats")
        if team_stats is None or long_table.height == 0:
            return long_table
        # season/week/gameweek are already on long_table from
        # _to_team_indexed; dropping team_stats's own copies before the join
        # avoids polars auto-suffixing them to season_right/week_right/
        # gameweek_right on collision.
        team_stats = team_stats.drop(
            [c for c in ("season", "week", "gameweek") if c in team_stats.columns]
        )
        return long_table.join(team_stats, on=["game_id", "team"], how="left")
