from __future__ import annotations

import polars as pl
import pytest

from degenebet.modeling.backtest import (
    Backtest,
    FlatSizing,
    _win_multiplier,
    compute_bankroll_trajectory,
)
from degenebet.modeling.splits import FoldPredictions
from degenebet.modeling.spread_model import SpreadModel


def test_win_multiplier_negative_odds() -> None:
    assert _win_multiplier(-140) == pytest.approx(0.7142857142857143)
    assert _win_multiplier(-110) == pytest.approx(0.9090909090909091)


def test_win_multiplier_positive_odds() -> None:
    assert _win_multiplier(120) == pytest.approx(1.2)


def test_flat_sizing_divides_pool_evenly() -> None:
    predictions = pl.DataFrame({"x": [1, 2, 3, 4]})

    sized = FlatSizing().size(predictions)

    assert sized["stake"].to_list() == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert sized["stake"].sum() == pytest.approx(1.0)


def test_flat_sizing_zero_rows_gives_zero_stake_not_a_division_error() -> None:
    predictions = pl.DataFrame({"x": []}, schema={"x": pl.Int64})

    sized = FlatSizing().size(predictions)

    assert sized.height == 0
    assert sized.schema["stake"] == pl.Float64


def _decided_predictions_fixture() -> pl.DataFrame:
    # A: home bet, wins, odds -140 (favorite). B: away bet, wins, odds
    # +120 (underdog). C: home bet, push. Deliberately asymmetric odds
    # (not -110 on both sides) so the test can't pass by accident under
    # the old hardcoded -110 assumption.
    return pl.DataFrame(
        {
            "game_id": ["A", "B", "C"],
            "predicted_result": [10.0, -10.0, 10.0],
            "result": [10.0, -10.0, 3.0],
            "spread_line": [3.0, -3.0, 3.0],
            "home_spread_odds": [-140, -140, -110],
            "away_spread_odds": [120, 120, 100],
        }
    )


def test_backtest_run_scores_win_loss_push_with_real_asymmetric_odds() -> None:
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)

    result = backtest.run(_decided_predictions_fixture())

    assert result.bets_placed == 3
    assert result.ats_win_rate == pytest.approx(1.0)  # 2/2 decided bets won, 1 push excluded
    assert result.units_won == pytest.approx(0.638095238095238)
    assert result.roi_pct == pytest.approx(63.8095238095238)
    assert result.by_fold is None

    push_row = result.bets.filter(pl.col("game_id") == "C")
    assert push_row["units"][0] == pytest.approx(0.0)
    assert push_row["push"][0] is True


def test_backtest_run_edge_threshold_filters_games() -> None:
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=100.0)

    result = backtest.run(_decided_predictions_fixture())

    assert result.bets_placed == 0
    assert result.ats_win_rate == 0.0
    assert result.units_won == 0.0
    assert result.roi_pct == 0.0


def test_backtest_run_empty_predictions_returns_zeroed_result_not_an_error() -> None:
    empty = pl.DataFrame(
        {
            "predicted_result": [],
            "result": [],
            "spread_line": [],
            "home_spread_odds": [],
            "away_spread_odds": [],
        },
        schema={
            "predicted_result": pl.Float64,
            "result": pl.Float64,
            "spread_line": pl.Float64,
            "home_spread_odds": pl.Int64,
            "away_spread_odds": pl.Int64,
        },
    )
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)

    result = backtest.run(empty)

    assert result.bets_placed == 0
    assert result.ats_win_rate == 0.0
    assert result.units_won == 0.0
    assert result.roi_pct == 0.0


def test_decide_bets_raises_value_error_on_missing_required_column() -> None:
    predictions = _decided_predictions_fixture().drop("spread_line")
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)

    with pytest.raises(ValueError, match=r"missing required column.*spread_line"):
        backtest.run(predictions)


def test_backtest_run_raises_on_null_predicted_result_or_result_or_spread_line() -> None:
    predictions = pl.DataFrame(
        {
            "predicted_result": [10.0],
            "result": [None],
            "spread_line": [3.0],
            "home_spread_odds": [-140],
            "away_spread_odds": [120],
        },
        schema={
            "predicted_result": pl.Float64,
            "result": pl.Float64,
            "spread_line": pl.Float64,
            "home_spread_odds": pl.Int64,
            "away_spread_odds": pl.Int64,
        },
    )
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)

    with pytest.raises(ValueError, match=r"null values in \['result'\]"):
        backtest.run(predictions)


def test_backtest_run_raises_on_null_bet_odds_for_the_side_actually_bet() -> None:
    predictions = pl.DataFrame(
        {
            "predicted_result": [10.0],
            "result": [10.0],
            "spread_line": [3.0],
            "home_spread_odds": [None],  # side will be "home" -- this is the relevant odds
            "away_spread_odds": [120],
        },
        schema={
            "predicted_result": pl.Float64,
            "result": pl.Float64,
            "spread_line": pl.Float64,
            "home_spread_odds": pl.Int64,
            "away_spread_odds": pl.Int64,
        },
    )
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)

    with pytest.raises(ValueError, match="null bet_odds"):
        backtest.run(predictions)


def test_backtest_run_does_not_raise_on_null_odds_for_the_side_not_bet() -> None:
    predictions = pl.DataFrame(
        {
            "predicted_result": [10.0],
            "result": [10.0],
            "spread_line": [3.0],
            "home_spread_odds": [-140],
            "away_spread_odds": [None],  # side is "home" -- irrelevant that away is null
        },
        schema={
            "predicted_result": pl.Float64,
            "result": pl.Float64,
            "spread_line": pl.Float64,
            "home_spread_odds": pl.Int64,
            "away_spread_odds": pl.Int64,
        },
    )
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)

    result = backtest.run(predictions)

    assert result.bets_placed == 1
    assert result.units_won == pytest.approx(0.7142857142857143)


def _fold(predictions: pl.DataFrame) -> FoldPredictions:
    # SpreadModel() here is an unfitted placeholder -- Backtest.run_folds
    # never reads FoldPredictions.model or .in_sample, only .out_of_sample.
    return FoldPredictions(model=SpreadModel(), in_sample=predictions, out_of_sample=predictions)


def test_backtest_run_folds_pools_bet_count_weighted_not_averaged_across_folds() -> None:
    # Fold 1: 1 bet, home, wins. Fold 2: 3 bets, home, 1 win + 2 losses (no
    # pushes). Naive average of each fold's own ats_win_rate would be
    # (1.0 + 1/3) / 2 = 0.667; the true bet-count-weighted pooled rate is
    # 2 wins / 4 decided bets = 0.5. FlatSizing normalizes every fold's
    # total stake to 1.0 regardless of bet count, so ats_win_rate (not
    # stake-weighted at all) is what actually proves pooling counts real
    # bets rather than averaging fold-level rates.
    fold_1 = _fold(
        pl.DataFrame(
            {
                "predicted_result": [10.0],
                "result": [10.0],
                "spread_line": [3.0],
                "home_spread_odds": [-110],
                "away_spread_odds": [-110],
            }
        )
    )
    fold_2 = _fold(
        pl.DataFrame(
            {
                "predicted_result": [10.0, 10.0, 10.0],
                "result": [10.0, 0.0, 0.0],
                "spread_line": [3.0, 3.0, 3.0],
                "home_spread_odds": [-110, -110, -110],
                "away_spread_odds": [-110, -110, -110],
            }
        )
    )
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)

    result = backtest.run_folds([fold_1, fold_2])

    assert result.ats_win_rate == pytest.approx(0.5)
    assert result.by_fold is not None
    assert result.by_fold.height == 2
    assert result.by_fold["fold"].to_list() == [0, 1]
    assert result.by_fold["ats_win_rate"].to_list() == pytest.approx([1.0, 1.0 / 3.0])


def test_backtest_run_folds_raises_on_empty_folds_stream() -> None:
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)

    with pytest.raises(ValueError, match="empty folds stream"):
        backtest.run_folds([])


def test_backtest_run_folds_scores_out_of_sample_only_never_in_sample() -> None:
    # in_sample deliberately has a shape that would raise (null result) if
    # run_folds ever touched it -- proving it doesn't.
    in_sample = pl.DataFrame(
        {
            "predicted_result": [10.0],
            "result": [None],
            "spread_line": [3.0],
            "home_spread_odds": [-140],
            "away_spread_odds": [120],
        },
        schema={
            "predicted_result": pl.Float64,
            "result": pl.Float64,
            "spread_line": pl.Float64,
            "home_spread_odds": pl.Int64,
            "away_spread_odds": pl.Int64,
        },
    )
    out_of_sample = pl.DataFrame(
        {
            "game_id": ["A", "B", "C"],
            "predicted_result": [10.0, -10.0, 10.0],
            "result": [10.0, -10.0, 3.0],
            "spread_line": [3.0, -3.0, 3.0],
            "home_spread_odds": [-140, -140, -110],
            "away_spread_odds": [120, 120, 100],
        }
    )
    fold = FoldPredictions(model=SpreadModel(), in_sample=in_sample, out_of_sample=out_of_sample)
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)

    result = backtest.run_folds([fold])

    assert result.bets_placed == 3


def test_compute_bankroll_trajectory_matches_motivating_example() -> None:
    # Start with $50, lose $30 (a -60% fold return) -> $20 left.
    by_fold = pl.DataFrame(
        {"fold": [0], "bets_placed": [1], "ats_win_rate": [0.0], "units_won": [-0.6]}
    )

    trajectory = compute_bankroll_trajectory(by_fold, starting_bankroll=50.0)

    assert trajectory.ending_bankroll == pytest.approx(20.0)
    assert trajectory.total_pnl == pytest.approx(-30.0)
    assert trajectory.by_fold["bankroll_before"].to_list() == pytest.approx([50.0])
    assert trajectory.by_fold["pnl"].to_list() == pytest.approx([-30.0])
    assert trajectory.by_fold["bankroll_after"].to_list() == pytest.approx([20.0])


def test_compute_bankroll_trajectory_compounds_across_multiple_folds() -> None:
    # Fold 0: -60% of $50 -> $20. Fold 1: +50% of $20 -> $30.
    by_fold = pl.DataFrame({"fold": [0, 1], "units_won": [-0.6, 0.5]})

    trajectory = compute_bankroll_trajectory(by_fold, starting_bankroll=50.0)

    assert trajectory.by_fold["bankroll_before"].to_list() == pytest.approx([50.0, 20.0])
    assert trajectory.by_fold["bankroll_after"].to_list() == pytest.approx([20.0, 30.0])
    assert trajectory.ending_bankroll == pytest.approx(30.0)
    assert trajectory.total_pnl == pytest.approx(-20.0)


def test_compute_bankroll_trajectory_zero_return_fold_leaves_bankroll_unchanged() -> None:
    by_fold = pl.DataFrame({"fold": [0], "units_won": [0.0]})

    trajectory = compute_bankroll_trajectory(by_fold, starting_bankroll=100.0)

    assert trajectory.ending_bankroll == pytest.approx(100.0)
    assert trajectory.total_pnl == pytest.approx(0.0)


def test_compute_bankroll_trajectory_raises_on_none_by_fold() -> None:
    with pytest.raises(ValueError, match="by_fold is None"):
        compute_bankroll_trajectory(None, 1000.0)


def test_compute_bankroll_trajectory_total_wipeout_stays_at_zero() -> None:
    # A fold where every bet loses has units_won == -1.0 exactly (100% of
    # that week's staked pool lost) -- full-reinvestment sizing means this
    # wipes the bankroll to $0, and it correctly stays $0 for every later
    # fold regardless of that fold's own return (a real occurrence in the
    # 2022-2024 real-data walk-forward run, not just a theoretical case).
    by_fold = pl.DataFrame({"fold": [0, 1], "units_won": [-1.0, 0.9]})

    trajectory = compute_bankroll_trajectory(by_fold, starting_bankroll=1000.0)

    assert trajectory.by_fold["bankroll_after"].to_list() == pytest.approx([0.0, 0.0])
    assert trajectory.ending_bankroll == pytest.approx(0.0)
    assert trajectory.total_pnl == pytest.approx(-1000.0)
