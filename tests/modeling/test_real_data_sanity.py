"""Real-data sanity check: the full pipeline against a frozen real slice.

Every other test in this package uses small synthetic fixtures with
hand-computed expected values — deliberately, since that's fast and
deterministic. But synthetic fixtures only prove the code does what a test
*expects*; they can't catch a semantic misunderstanding of the real data
(e.g. the spread_line sign convention bug this test exists to guard
against — the original bug produced a 76% ATS win rate, which four
task-scoped reviews and a synthetic test suite all missed, because no
synthetic fixture happened to probe an asymmetric spread/margin case).

To keep this out of "no real network calls in tests" territory, the real
data (2022-2024 schedules + team_stats, trimmed to only the columns the
pipeline needs) is pulled once and frozen as parquet fixtures under
_fixtures/ rather than fetched live. Regenerate them only if the pipeline's
input column requirements change — see the module docstring commands
below.

To regenerate, run this from the repo root (only needed if the pipeline's
input column requirements change):

    uv run python - <<'EOF'
    import degenebet.data as data

    team_stats_cols = [
        "season", "week", "team", "opponent_team", "game_id",
        "passing_epa", "rushing_epa", "attempts", "carries",
        "def_interceptions", "fumble_recovery_opp",
        "passing_interceptions", "fumbles_lost_total",
    ]
    schedules_cols = [
        "game_id", "season", "week", "home_team", "away_team",
        "result", "spread_line", "home_spread_odds", "away_spread_odds",
    ]

    (
        data.load_team_stats([2022, 2023, 2024])
        .select(team_stats_cols)
        .write_parquet("tests/modeling/_fixtures/real_team_stats_2022_2024.parquet")
    )
    (
        data.load_schedules([2022, 2023, 2024])
        .select(schedules_cols)
        .write_parquet("tests/modeling/_fixtures/real_schedules_2022_2024.parquet")
    )
    EOF
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from degenebet.data.access import _widen_with_team_data
from degenebet.modeling.backtest import Backtest, FlatSizing, compute_bankroll_trajectory
from degenebet.modeling.features import compute_rolling_features
from degenebet.modeling.splits import SingleSplit, WalkForwardSplit, iterate_folds
from degenebet.modeling.spread_model import SpreadModel

_FIXTURES_DIR = Path(__file__).parent / "_fixtures"


def _real_model_table() -> pl.DataFrame:
    team_stats = pl.read_parquet(_FIXTURES_DIR / "real_team_stats_2022_2024.parquet")
    schedules = pl.read_parquet(_FIXTURES_DIR / "real_schedules_2022_2024.parquet")

    rolling_features = compute_rolling_features(team_stats)
    # Same drop_nulls/filter contract as the sanity check below -- see its
    # docstring for why _widen_with_team_data's left join needs this.
    return (
        _widen_with_team_data(schedules, rolling_features)
        .drop_nulls(
            subset=[
                "home_offense_epa",
                "home_defense_epa_allowed",
                "home_turnover_margin",
                "away_offense_epa",
                "away_defense_epa_allowed",
                "away_turnover_margin",
            ]
        )
        .filter(
            pl.col("result").is_not_null()
            & pl.col("spread_line").is_not_null()
            & pl.col("home_spread_odds").is_not_null()
            & pl.col("away_spread_odds").is_not_null()
        )
    )


def test_backtest_ats_win_rate_is_plausible_against_real_data() -> None:
    """A naive linear baseline's ATS win rate should land near 50%, not in
    the 70s or 20s — either extreme is almost always a bug (e.g. a sign
    error), not a surprisingly good or bad model. This band is deliberately
    generous (not a claim this model is calibrated) — it exists to catch
    the class of bug that produces an implausible result, not to validate
    the model's actual edge.
    """
    model_table = _real_model_table()

    split_strategy = SingleSplit(train_seasons=[2022, 2023], test_seasons=[2024])
    folds = list(iterate_folds(model_table, split_strategy, SpreadModel))
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)
    result = backtest.run(folds[0].out_of_sample)

    assert result.bets_placed > 50, "too few bets placed to be a meaningful sanity check"
    assert 0.35 <= result.ats_win_rate <= 0.65, (
        f"ATS win rate {result.ats_win_rate:.3f} is outside the plausible band for a naive "
        "baseline — check for a sign error or other systematic scoring bug before trusting "
        "this number"
    )


def test_walk_forward_backtest_matches_golden_fixture_on_real_data() -> None:
    """Pins WalkForwardSplit + Backtest.run_folds + compute_bankroll_trajectory
    against the same frozen real 2022-2024 slice, at tight tolerance (not the
    plausibility-band check above). Unlike the SingleSplit sanity check, the
    walk-forward + run_folds + compounding-bankroll path was previously only
    exercised by notebooks/03_backtest.py -- outside `just check` -- even
    though it includes two real, load-bearing behaviors from that same data:
    a fold with zero placed bets (a bye-week-like gap), and a fold where every
    placed bet loses (units_won == -1.0 exactly), which permanently zeroes a
    full-reinvestment bankroll for every later fold. A regression in
    WalkForwardSplit's fold boundaries or run_folds' pooling could otherwise
    ship green until someone happened to re-run that notebook by hand.

    These expected values were independently reproduced twice already (once
    by this branch's implementer, once by its reviewer) via headless
    `marimo export html` against this exact fixture -- see
    docs/superpowers/plans/2026-09-18-backtest-rework.md's Task 6 and its
    final review. Regenerate them deliberately (re-run the walk-forward
    backtest and update the asserted numbers) if the fixture or the pipeline
    upstream of it ever changes -- never as a reflexive fix for a failing
    test.
    """
    model_table = _real_model_table().with_columns(
        (pl.col("season") * 100 + pl.col("week")).alias("gameweek")
    )

    split_strategy = WalkForwardSplit(warmup_seasons=1)
    folds = iterate_folds(model_table, split_strategy, SpreadModel)
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)
    result = backtest.run_folds(folds)

    assert result.bets_placed == 438
    assert result.ats_win_rate == pytest.approx(0.481043, abs=1e-5)
    assert result.units_won == pytest.approx(-2.624233, abs=1e-5)
    assert result.roi_pct == pytest.approx(-6.1029, abs=1e-3)
    assert result.by_fold is not None
    assert result.by_fold.height == 44

    # The real zero-bet fold and the real total-wipeout fold both exist in
    # this data (see the docstring above) -- confirm both are still present
    # rather than only checking the pooled top-line numbers, which could stay
    # unchanged even if a regression shifted which folds they land in.
    assert (result.by_fold["bets_placed"] == 0).sum() == 1
    assert any(units == pytest.approx(-1.0) for units in result.by_fold["units_won"])

    trajectory = compute_bankroll_trajectory(result.by_fold, starting_bankroll=1000.0)

    assert trajectory.starting_bankroll == pytest.approx(1000.0)
    assert trajectory.ending_bankroll == pytest.approx(0.0)
    assert trajectory.total_pnl == pytest.approx(-1000.0)
