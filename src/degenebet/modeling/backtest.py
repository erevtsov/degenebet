"""Backtest: market-aware, sizing-aware scoring of a Model's predictions
against real historical closing lines and real per-bet American odds. See
docs/superpowers/specs/2026-09-18-backtest-rework-design.md.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

import polars as pl

from degenebet.modeling.splits import FoldPredictions


class _BetsSummary(TypedDict):
    bets_placed: int
    ats_win_rate: float
    units_won: float
    roi_pct: float


def _win_multiplier(odds: int) -> float:
    """American odds -> profit per unit risked. Negative odds (e.g. -110,
    risk $1.10 to win $1.00): 100/abs(odds). Positive odds (e.g. +120,
    risk $100 to win $120): odds/100."""
    return 100.0 / abs(odds) if odds < 0 else odds / 100.0


class SizingStrategy(Protocol):
    """predictions already has edge, side, and win_multiplier computed
    (win_multiplier so a future odds-aware strategy, e.g. Kelly, can size
    against the real payout, not just the edge). Adds a stake column."""

    def size(self, predictions: pl.DataFrame) -> pl.DataFrame: ...


class FlatSizing:
    """Divides a normalized pool of 1.0 evenly across every bet in this
    call -- stake_i = 1.0 / N for N placed bets, regardless of edge or
    odds. This is "flat" in the sense of splitting a period's betting
    pool evenly across that period's bets, not a constant per-bet dollar
    amount. The backtest still scores each bet at its own real payout
    odds; FlatSizing only decides how the pool is split, never the
    payout."""

    def size(self, predictions: pl.DataFrame) -> pl.DataFrame:
        n = predictions.height
        return predictions.with_columns(pl.lit(1.0 / n if n > 0 else 0.0).alias("stake"))


def _decide_bets(predictions: pl.DataFrame, edge_threshold: float) -> pl.DataFrame:
    """Guard 1 (predicted_result/result/spread_line non-null), compute
    edge = predicted_result - spread_line, decide side (home if
    edge > edge_threshold, away if edge < -edge_threshold, else none),
    filter to placed bets, resolve bet_odds and win_multiplier. Guard 2
    (bet_odds non-null among placed bets). Backtest never filters
    unplayed games itself -- see the module docstring; the caller opts in
    explicitly before this is ever called."""
    required_columns = ("predicted_result", "result", "spread_line")
    missing_columns = [c for c in required_columns if c not in predictions.columns]
    if missing_columns:
        raise ValueError(
            f"predictions is missing required column(s) {missing_columns} -- Backtest "
            "needs predicted_result, result, and spread_line to decide and score bets."
        )

    null_counts = predictions.select("predicted_result", "result", "spread_line").null_count()
    null_columns = [
        c for c in ("predicted_result", "result", "spread_line") if null_counts[c][0] > 0
    ]
    if null_columns:
        raise ValueError(
            f"predictions has null values in {null_columns} -- Backtest never filters "
            "unplayed games itself. Filter them out explicitly (e.g. "
            "predictions.filter(pl.col('result').is_not_null() & "
            "pl.col('spread_line').is_not_null())) before calling run()/run_folds()."
        )

    decided = predictions.with_columns(
        (pl.col("predicted_result") - pl.col("spread_line")).alias("edge")
    ).with_columns(
        pl.when(pl.col("edge") > edge_threshold)
        .then(pl.lit("home"))
        .when(pl.col("edge") < -edge_threshold)
        .then(pl.lit("away"))
        .otherwise(pl.lit("none"))
        .alias("side")
    )

    placed = decided.filter(pl.col("side") != "none").with_columns(
        pl.when(pl.col("side") == "home")
        .then(pl.col("home_spread_odds"))
        .otherwise(pl.col("away_spread_odds"))
        .alias("bet_odds")
    )

    if placed.height > 0 and placed["bet_odds"].null_count() > 0:
        raise ValueError(
            "predictions has null bet_odds (home_spread_odds/away_spread_odds) among "
            "placed bets -- Backtest never filters unplayed/unpriced games itself. "
            "Filter them out explicitly before calling run()/run_folds()."
        )

    return placed.with_columns(
        # map_elements (not a vectorized pl.when expression) so _win_multiplier
        # has exactly one implementation -- directly unit-tested and directly
        # used, never duplicated between a scalar function and an inline
        # expression that could drift from it. Placed-bet counts per call are
        # small (one gameweek's qualifying games at most), so the per-row
        # Python call overhead is not a real cost here.
        pl.col("bet_odds")
        .map_elements(_win_multiplier, return_dtype=pl.Float64)
        .alias("win_multiplier")
    )


def _score_bets(sized: pl.DataFrame) -> pl.DataFrame:
    """sized already has stake (from SizingStrategy.size()). Computes
    home_cover_margin = result - spread_line, push/won, and units =
    stake * win_multiplier if won, -stake if lost, 0 if push."""
    return (
        sized.with_columns((pl.col("result") - pl.col("spread_line")).alias("home_cover_margin"))
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
            .then(pl.col("stake") * pl.col("win_multiplier"))
            .otherwise(-pl.col("stake"))
            .alias("units")
        )
    )


def _summarize_bets(bets_df: pl.DataFrame) -> _BetsSummary:
    """bets_placed, ats_win_rate, units_won, roi_pct. Returns zeroed
    values if bets_df is empty (0 rows) -- not an error, see the
    null-handling contract above. Shared by run() and both the per-fold
    and pooled calls inside run_folds() -- same DRY principle as
    Efficacy's _compute_metrics. roi_pct's denominator (sum of stake)
    excludes a 0-bet fold entirely, unlike compute_bankroll_trajectory,
    which treats a 0-bet fold as a legitimate 0%-return period."""
    if bets_df.height == 0:
        return {"bets_placed": 0, "ats_win_rate": 0.0, "units_won": 0.0, "roi_pct": 0.0}

    decided = bets_df.filter(~pl.col("push"))
    bets_placed = bets_df.height
    ats_win_rate = float(cast(float, decided["won"].mean())) if decided.height > 0 else 0.0
    units_won = float(cast(float, bets_df["units"].sum()))
    total_staked = float(cast(float, bets_df["stake"].sum()))
    roi_pct = (units_won / total_staked * 100) if total_staked > 0 else 0.0

    return {
        "bets_placed": bets_placed,
        "ats_win_rate": ats_win_rate,
        "units_won": units_won,
        "roi_pct": roi_pct,
    }


@dataclass(frozen=True)
class BacktestResult:
    """Result of scoring a set of predictions as bets; by_fold is None
    from run(), or a per-fold breakdown from run_folds()."""

    bets: pl.DataFrame
    bets_placed: int
    ats_win_rate: float
    units_won: float
    roi_pct: float
    by_fold: pl.DataFrame | None


class Backtest:
    """Market-aware, sizing-aware scorer -- mirrors Efficacy's
    evaluate()/evaluate_folds() shape. ats_win_rate is NOT the same
    concept as Efficacy's directional_accuracy -- see
    architecture-notes.md's explicit warning against conflating the two
    "win rate"s."""

    def __init__(self, sizing_strategy: SizingStrategy, edge_threshold: float = 1.0) -> None:
        self.sizing_strategy = sizing_strategy
        self.edge_threshold = edge_threshold

    def run(self, predictions: pl.DataFrame) -> BacktestResult:
        """Scores one frame. by_fold is always None here -- a single
        frame has no fold concept, so there's nothing to build a
        breakdown from."""
        decided = _decide_bets(predictions, self.edge_threshold)
        bets_df = _score_bets(self.sizing_strategy.size(decided))
        return BacktestResult(bets=bets_df, by_fold=None, **_summarize_bets(bets_df))

    def run_folds(self, folds: Iterable[FoldPredictions]) -> BacktestResult:
        """Scores fold.out_of_sample only -- never in_sample; you don't
        bet on training data. Raises ValueError on an empty folds stream
        (zero folds total -- distinct from a fold with zero placed bets,
        which is legitimate). Pooled stats computed by concatenating
        every fold's scored bets and summarizing once -- never by
        averaging each fold's already-computed stats, same principle
        Efficacy.evaluate_folds established."""
        materialized = list(folds)
        if not materialized:
            raise ValueError("Cannot run_folds on an empty folds stream.")

        by_fold_rows: list[dict[str, object]] = []
        all_bets: list[pl.DataFrame] = []
        for fold_index, fold in enumerate(materialized):
            decided = _decide_bets(fold.out_of_sample, self.edge_threshold)
            bets_df = _score_bets(self.sizing_strategy.size(decided))
            by_fold_rows.append({"fold": fold_index, **_summarize_bets(bets_df)})
            all_bets.append(bets_df)

        pooled = pl.concat(all_bets, how="vertical")
        return BacktestResult(
            bets=pooled, by_fold=pl.DataFrame(by_fold_rows), **_summarize_bets(pooled)
        )


@dataclass(frozen=True)
class BankrollTrajectory:
    by_fold: pl.DataFrame
    starting_bankroll: float
    ending_bankroll: float
    total_pnl: float


def compute_bankroll_trajectory(
    by_fold: pl.DataFrame | None, starting_bankroll: float
) -> BankrollTrajectory:
    """Sequentially compounds BacktestResult.by_fold's per-fold units_won
    (a normalized fractional return under FlatSizing's evenly-divided
    pool) into a dollar-denominated trajectory:
    bankroll_after = bankroll_before * (1 + units_won). Requires by_fold
    to already be in time order -- WalkForwardSplit's `fold` index (0, 1,
    2, ... in gameweek order) guarantees this as long as by_fold isn't
    re-sorted before calling this. Raises ValueError if by_fold is None
    (i.e. it came from Backtest.run() rather than run_folds()). Only
    correct for a SizingStrategy whose allocation is linear in the pool
    size (FlatSizing qualifies; see the design spec's Non-goals for what
    doesn't)."""
    if by_fold is None:
        raise ValueError(
            "by_fold is None -- call run_folds(), not run(), to get a by-fold "
            "breakdown for compute_bankroll_trajectory"
        )

    rows = by_fold.sort("fold").iter_rows(named=True)
    bankroll = starting_bankroll
    trajectory_rows: list[dict[str, float | int]] = []
    for row in rows:
        bankroll_before = bankroll
        pnl = bankroll_before * row["units_won"]
        bankroll = bankroll_before + pnl
        trajectory_rows.append(
            {
                "fold": row["fold"],
                "bankroll_before": bankroll_before,
                "pnl": pnl,
                "bankroll_after": bankroll,
            }
        )

    return BankrollTrajectory(
        by_fold=pl.DataFrame(trajectory_rows),
        starting_bankroll=starting_bankroll,
        ending_bankroll=bankroll,
        total_pnl=bankroll - starting_bankroll,
    )
