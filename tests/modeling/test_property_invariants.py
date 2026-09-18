"""Property-based tests for invariants that would be expensive to get
silently wrong: cover probability bounds, backtest P&L reconciliation
under re-aggregation, and Efficacy's directional_accuracy/r_squared bounds
(per AGENTS.md's Automation & Verification mandate).

All three tests below exercise the real production code paths
(`SpreadModel.cover_probability`, `Backtest.run`, and `Efficacy.evaluate`)
rather than re-testing the scipy/polars primitives they're built on.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, reject
from hypothesis import strategies as st

from degenebet.modeling.backtest import Backtest, FlatSizing
from degenebet.modeling.efficacy import Efficacy
from degenebet.modeling.splits import SingleSplit, iterate_folds
from degenebet.modeling.spread_model import SpreadModel

_FEATURE_COLUMNS = [
    "home_offense_epa",
    "home_defense_epa_allowed",
    "home_turnover_margin",
    "away_offense_epa",
    "away_defense_epa_allowed",
    "away_turnover_margin",
]

_finite_float = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)


def _fitted_model() -> SpreadModel:
    # A small, fixed, well-conditioned model_table (6 features + result),
    # fit once — fitting per hypothesis example would be slow and isn't
    # what this property is about (the property is cover_probability's
    # boundedness, not fit's behavior).
    table = pl.DataFrame(
        {
            "home_offense_epa": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            "home_defense_epa_allowed": [0.2, 0.1, -0.1, 0.3, -0.2, 0.0],
            "home_turnover_margin": [1.0, -1.0, 0.0, 2.0, -2.0, 1.0],
            "away_offense_epa": [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6],
            "away_defense_epa_allowed": [0.0, 0.3, -0.2, 0.1, 0.2, -0.1],
            "away_turnover_margin": [-1.0, 1.0, 0.0, -2.0, 2.0, -1.0],
            "result": [3.0, -5.0, 7.0, -2.0, 4.0, -1.0],
        }
    )
    model = SpreadModel()
    model.fit(table)
    return model


_MODEL = _fitted_model()


@given(predicted_result=_finite_float, spread_line=_finite_float)
def test_cover_probability_always_in_unit_interval(
    predicted_result: float, spread_line: float
) -> None:
    # cover_probability skips re-predicting when `predicted_result` is
    # already present, so a 2-column frame exercises the real transform
    # (norm_cdf((predicted_result - spread_line) / residual_std)) without
    # needing the full 6-feature model_table per example.
    table = pl.DataFrame({"predicted_result": [predicted_result], "spread_line": [spread_line]})

    result = _MODEL.cover_probability(table)

    probability = result["home_cover_probability"][0]
    assert 0.0 <= probability <= 1.0


_small_float = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
# American odds never fall strictly between -100 and +100 -- generate from
# the two realistic ranges rather than an arbitrary nonzero integer.
_odds_int = st.one_of(
    st.integers(min_value=-500, max_value=-100), st.integers(min_value=100, max_value=500)
)
_game_row = st.tuples(*([_small_float] * 8), _odds_int, _odds_int)
# 6 features + result + spread_line + home_spread_odds + away_spread_odds


_test_row = st.tuples(_game_row, st.integers(min_value=1, max_value=4))


@given(
    train_rows=st.lists(_game_row, min_size=2, max_size=6),
    test_rows=st.lists(_test_row, min_size=1, max_size=8),
)
def test_backtest_pnl_reconciles_under_reaggregation(
    train_rows: list[tuple[float, ...]],
    test_rows: list[tuple[tuple[float, ...], int]],
) -> None:
    n_train = len(train_rows)
    train_df = pl.DataFrame(
        {
            "game_id": [f"train_{i}" for i in range(n_train)],
            "season": [2000] * n_train,
            "week": list(range(1, n_train + 1)),
            **{col: [row[i] for row in train_rows] for i, col in enumerate(_FEATURE_COLUMNS)},
            "result": [row[6] for row in train_rows],
            "spread_line": [row[7] for row in train_rows],
            "home_spread_odds": [int(row[8]) for row in train_rows],
            "away_spread_odds": [int(row[9]) for row in train_rows],
        }
    )

    n_test = len(test_rows)
    test_df = pl.DataFrame(
        {
            "game_id": [f"test_{i}" for i in range(n_test)],
            "season": [2001] * n_test,
            "week": [week for _, week in test_rows],
            **{col: [row[i] for row, _ in test_rows] for i, col in enumerate(_FEATURE_COLUMNS)},
            "result": [row[6] for row, _ in test_rows],
            "spread_line": [row[7] for row, _ in test_rows],
            "home_spread_odds": [int(row[8]) for row, _ in test_rows],
            "away_spread_odds": [int(row[9]) for row, _ in test_rows],
        }
    )

    model_table = pl.concat([train_df, test_df])
    split_strategy = SingleSplit(train_seasons=[2000], test_seasons=[2001])
    try:
        folds = list(iterate_folds(model_table, split_strategy, SpreadModel))
    except ValueError:
        # A degenerate train draw (e.g. residual_std ~0 or NaN) is rejected
        # by SpreadModel.fit's own guard — not what this property is about
        # (P&L re-aggregation), so discard the example rather than fail.
        reject()

    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)
    full_result = backtest.run(folds[0].out_of_sample)

    weekly_sum = (
        full_result.bets.group_by("week")
        .agg(pl.col("units").sum())
        .select(pl.col("units").sum())
        .item()
        if full_result.bets.height > 0
        else 0.0
    )

    assert weekly_sum == pytest.approx(full_result.units_won, abs=1e-9)


_prediction_pair = st.tuples(_finite_float, _finite_float)


@given(pairs=st.lists(_prediction_pair, min_size=2, max_size=20))
def test_efficacy_directional_accuracy_and_r_squared_are_bounded(
    pairs: list[tuple[float, float]],
) -> None:
    predicted = [p for p, _ in pairs]
    actual = [a for _, a in pairs]
    # A near-constant `actual` column makes r2_score's denominator (the sum
    # of squared deviations from the mean) ~0, producing a huge or NaN R²
    # from an essentially unrelated numerical fluke -- not what this
    # property is about (boundedness under real variation), so discard
    # those examples. A near-constant `predicted` column triggers scipy's
    # ConstantInputWarning in pearsonr (undefined correlation) -- not a
    # failure, but noise this property test doesn't need either.
    if np.std(actual) < 1e-6 or np.std(predicted) < 1e-6:
        reject()

    predictions = pl.DataFrame({"predicted_result": predicted, "result": actual})

    result = Efficacy().evaluate(predictions)

    assert 0.0 <= result.metrics["directional_accuracy"] <= 1.0
    assert result.metrics["r_squared"] <= 1.0 + 1e-9
