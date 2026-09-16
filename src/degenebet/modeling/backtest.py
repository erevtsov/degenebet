"""Backtest harness: evaluates a SpreadModel against real historical closing lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import polars as pl

from degenebet.modeling.spread_model import SpreadModel

_UNITS_RISKED_PER_BET = 1.1  # -110 pricing: risk 1.1 units to win 1.0


@dataclass(frozen=True)
class BacktestResult:
    bets: pl.DataFrame
    bets_placed: int
    ats_win_rate: float
    units_won: float
    roi_pct: float


def run_backtest(
    model_table: pl.DataFrame,
    *,
    train_seasons: list[int],
    test_seasons: list[int],
    edge_threshold: float = 1.0,
) -> BacktestResult:
    """Fits a fresh SpreadModel on train_seasons, predicts on test_seasons.

    Bets 1 unit on the side (home/away) whose edge exceeds edge_threshold;
    skips games without sufficient edge. Scores each bet against the real
    `result` via the cover-margin formula (result - spread_line) at -110
    pricing (push refunds the bet, excluded from win rate).
    """
    train = model_table.filter(pl.col("season").is_in(train_seasons))
    test = model_table.filter(pl.col("season").is_in(test_seasons))

    model = SpreadModel()
    model.fit(train)
    predicted = model.predict(test)

    predicted = predicted.with_columns(
        (pl.col("predicted_result") - pl.col("spread_line")).alias("edge")
    ).with_columns(
        pl.when(pl.col("edge") > edge_threshold)
        .then(pl.lit("home"))
        .when(pl.col("edge") < -edge_threshold)
        .then(pl.lit("away"))
        .otherwise(pl.lit("none"))
        .alias("side")
    )

    bets_df = (
        predicted.filter(pl.col("side") != "none")
        .with_columns((pl.col("result") - pl.col("spread_line")).alias("home_cover_margin"))
        .with_columns(
            (pl.col("home_cover_margin") == 0).alias("push"),
            pl.when(pl.col("side") == "home")
            .then(pl.col("home_cover_margin") > 0)
            .otherwise(pl.col("home_cover_margin") < 0)
            .alias("won"),
        )
        .with_columns(
            pl.when(pl.col("push"))
            .then(0.0)
            .when(pl.col("won"))
            .then(1.0)
            .otherwise(-_UNITS_RISKED_PER_BET)
            .alias("units")
        )
    )

    decided = bets_df.filter(~pl.col("push"))
    bets_placed = bets_df.height
    ats_win_rate = float(cast(float, decided["won"].mean())) if decided.height > 0 else 0.0
    units_won = float(cast(float, bets_df["units"].sum()))
    roi_pct = (units_won / (bets_placed * _UNITS_RISKED_PER_BET) * 100) if bets_placed > 0 else 0.0

    return BacktestResult(
        bets=bets_df,
        bets_placed=bets_placed,
        ats_win_rate=ats_win_rate,
        units_won=units_won,
        roi_pct=roi_pct,
    )
