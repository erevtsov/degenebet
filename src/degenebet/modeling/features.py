"""No-lookahead rolling team efficiency features for the spread model.

Offensive EPA/play uses only passing_epa + rushing_epa — adding
receiving_epa would double-count the same pass plays from the receiver's
side. Defensive EPA allowed has no direct column in team_stats; it's the
opponent's offensive EPA/play in the same game, found via a join.
"""

from __future__ import annotations

import polars as pl

# Maps each per-game metric to its rolling-output column name. Output names
# match SpreadModel._FEATURE_COLUMNS (minus the home_/away_ prefix DataAccess's
# team_data widening adds), so the widened output needs no further renaming.
_ROLLING_METRICS = {
    "offense_epa_per_play": "offense_epa",
    "defense_epa_allowed_per_play": "defense_epa_allowed",
    "turnover_margin": "turnover_margin",
}


def compute_rolling_features(
    team_stats: pl.DataFrame,
    *,
    window: int = 8,
    min_history: int = 4,
) -> pl.DataFrame:
    """One row per (team, game_id): trailing rolling averages, no lookahead.

    Each row's rolling values are computed strictly from that team's games
    before this one (`.shift(1)` before the rolling mean) — window may span
    a season boundary. Rows with fewer than `min_history` prior games are
    dropped, not padded.
    """
    per_game = team_stats.select(
        "season",
        "week",
        "team",
        "opponent_team",
        "game_id",
        (
            (pl.col("passing_epa") + pl.col("rushing_epa"))
            / (pl.col("attempts") + pl.col("carries"))
        ).alias("offense_epa_per_play"),
        (
            (pl.col("def_interceptions") + pl.col("fumble_recovery_opp"))
            - (pl.col("passing_interceptions") + pl.col("fumbles_lost_total"))
        ).alias("turnover_margin"),
    )

    opponent_offense = per_game.select(
        pl.col("game_id"),
        pl.col("team"),
        pl.col("offense_epa_per_play").alias("defense_epa_allowed_per_play"),
    )
    per_game = per_game.join(
        opponent_offense,
        left_on=["game_id", "opponent_team"],
        right_on=["game_id", "team"],
        how="left",
    )

    per_game = per_game.sort(["team", "season", "week"])
    rolling = per_game.with_columns(
        [
            pl.col(c)
            .shift(1)
            .rolling_mean(window_size=window, min_samples=min_history)
            .over("team")
            .alias(out_name)
            for c, out_name in _ROLLING_METRICS.items()
        ]
    )

    rolling_cols = list(_ROLLING_METRICS.values())
    not_null = pl.lit(True)
    for c in rolling_cols:
        not_null = not_null & pl.col(c).is_not_null()

    return rolling.filter(not_null).select("season", "week", "team", "game_id", *rolling_cols)
