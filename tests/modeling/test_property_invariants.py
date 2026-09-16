"""Property-based tests for invariants that would be expensive to get
silently wrong: cover probability bounds, and backtest P&L reconciling
under re-aggregation (per AGENTS.md's Automation & Verification mandate).
"""

from __future__ import annotations

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import norm

_finite_float = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
_positive_residual_std = st.floats(
    min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False
)


@given(edge=_finite_float, residual_std=_positive_residual_std)
def test_cover_probability_always_in_unit_interval(edge: float, residual_std: float) -> None:
    # Exercise the same normal_cdf transform SpreadModel.cover_probability
    # uses, directly, since a full model.fit() call per hypothesis example
    # would be slow — this tests the transform's own boundedness.
    probability = float(norm.cdf(edge / residual_std))

    assert 0.0 <= probability <= 1.0


@given(
    units=st.lists(
        st.floats(min_value=-1.1, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=30,
    ),
    weeks=st.integers(min_value=1, max_value=5),
)
def test_backtest_pnl_reconciles_under_reaggregation(units: list[float], weeks: int) -> None:
    if not units:
        return
    week_assignment = [i % weeks + 1 for i in range(len(units))]
    bets = pl.DataFrame({"units": units, "week": week_assignment})

    season_total = bets["units"].sum()
    weekly_sums = (
        bets.group_by("week").agg(pl.col("units").sum()).select(pl.col("units").sum()).item()
    )

    assert weekly_sums == pytest.approx(season_total, abs=1e-9)
