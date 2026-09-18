from __future__ import annotations

import polars as pl
import pytest

from degenebet.modeling.backtest import Backtest, FlatSizing, _win_multiplier


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
