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
        "result", "spread_line",
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

from degenebet.modeling.backtest import run_backtest
from degenebet.modeling.features import build_model_table, compute_rolling_features

_FIXTURES_DIR = Path(__file__).parent / "_fixtures"


def test_backtest_ats_win_rate_is_plausible_against_real_data() -> None:
    """A naive linear baseline's ATS win rate should land near 50%, not in
    the 70s or 20s — either extreme is almost always a bug (e.g. a sign
    error), not a surprisingly good or bad model. This band is deliberately
    generous (not a claim this model is calibrated) — it exists to catch
    the class of bug that produces an implausible result, not to validate
    the model's actual edge.
    """
    team_stats = pl.read_parquet(_FIXTURES_DIR / "real_team_stats_2022_2024.parquet")
    schedules = pl.read_parquet(_FIXTURES_DIR / "real_schedules_2022_2024.parquet")

    rolling_features = compute_rolling_features(team_stats)
    model_table = build_model_table(schedules, rolling_features)

    result = run_backtest(model_table, train_seasons=[2022, 2023], test_seasons=[2024])

    assert result.bets_placed > 50, "too few bets placed to be a meaningful sanity check"
    assert 0.35 <= result.ats_win_rate <= 0.65, (
        f"ATS win rate {result.ats_win_rate:.3f} is outside the plausible band for a naive "
        "baseline — check for a sign error or other systematic scoring bug before trusting "
        "this number"
    )
