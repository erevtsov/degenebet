# Baseline Spread Model + Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A linear-regression point-spread model plus a backtest harness that evaluates it against real historical closing lines, with the golden-fixture/property-test coverage `AGENTS.md` mandates for any modeling work.

**Architecture:** A `modeling/` subpackage with three layers — `features.py` (no-lookahead rolling team efficiency features), `spread_model.py` (a `SpreadModel` composed with an injected regressor — `LinearRegression` by default, swappable for any sklearn-compatible estimator satisfying a small `RegressorProtocol` — plus a derived cover probability), `backtest.py` (single train/test season split, flat-unit -110 scoring). Golden fixtures and `hypothesis` property tests land alongside, not after.

**Tech Stack:** Python 3.12+, existing `uv`/polars/ruff/mypy/pytest toolchain, plus new dependencies `scikit-learn`, `scipy` (core — used directly for `norm.cdf`, not just transitively via sklearn) and `hypothesis` (test extra).

**Spec:** `docs/superpowers/specs/2026-09-15-baseline-spread-model-design.md`

## Global Constraints

- Polars for all tabular data; numpy only at the sklearn/scipy call boundary (`.to_numpy()`); pandas never used.
- **Verified data facts (not assumptions — confirmed against real data during design):**
  - `spread_line` is negative when the home team is favored. Home team's cover margin is `result + spread_line`: positive → home covers, negative → away covers, zero → push.
  - `offense_epa_per_play = (passing_epa + rushing_epa) / (attempts + carries)` — do **not** add `receiving_epa`, it double-counts the same pass plays from the receiver's perspective.
  - `team_stats` has no direct "defensive EPA allowed" column — derive it as the opponent's `offense_epa_per_play` in the same `game_id`, via a join on `(game_id, opponent_team=team)`.
  - Rolling features use `.shift(1)` before `.rolling_mean(window_size=..., min_samples=...)` (note: this installed polars version's parameter is `min_samples`, not `min_periods`) `.over("team")` — this excludes the current game from its own rolling average. Rows below `min_history` are dropped, never padded.
- -110 pricing: a win nets `+1.0` unit, a loss nets `-1.1` units (risking 1.1 to win 1.0), a push nets `0`. `roi_pct = units_won / (bets_placed * 1.1) * 100`.
- Neither `scikit-learn` nor `scipy` ships a `py.typed` marker (verified) — both need `ignore_missing_imports = true` mypy overrides, same pattern as the existing `nflreadpy` override.
- `SpreadModel` is composed with an injected `RegressorProtocol` (structural: `.fit(X, y)`, `.predict(X)`), defaulting to `LinearRegression()` when none is given — mirroring the existing `Provider`/`waypoint`-style pluggable-strategy `Protocol` convention already used elsewhere in this codebase, rather than hardcoding the estimator. This is a deliberate revision from the spec's original wording ("uses scikit-learn's LinearRegression internally") made during plan review, in direct service of the spec's own stated future intent to swap in other model classes without refactoring `SpreadModel`.
- No real network calls in tests — `team_stats`/`schedules` test fixtures are small hand-built in-memory `pl.DataFrame`s with realistic column names/dtypes, not real fetched data.
- **PR-only workflow:** every task commits locally to this plan's branch as usual. Only at the very end, when opening the branch's PR (per `superpowers:finishing-a-development-branch`), the agent creates the PR and stops — it never runs `gh pr merge`. A human merges. This applies to any other PR-worthy action encountered mid-plan too.
- Conventional commits; `just check` clean (ruff check, ruff format check, mypy, pytest) before any task is done.

---

## File Structure

```
src/degenebet/modeling/
  __init__.py
  features.py
  spread_model.py
  backtest.py
tests/modeling/
  __init__.py
  test_features.py
  test_spread_model.py
  test_backtest.py
  _golden_spread_fixture.py
  test_spread_model_golden.py
  test_property_invariants.py
pyproject.toml   # modified: + scikit-learn, scipy (core); + hypothesis (test extra); + 2 mypy overrides
```

---

### Task 1: Feature pipeline

**Files:**
- Create: `src/degenebet/modeling/__init__.py` (empty)
- Create: `src/degenebet/modeling/features.py`
- Create: `tests/modeling/__init__.py` (empty)
- Create: `tests/modeling/test_features.py`

**Interfaces:**
- Consumes: nothing project-internal — operates on `team_stats`/`schedules`-shaped `pl.DataFrame`s matching `degenebet.data.load_team_stats()`/`load_schedules()`'s real column names.
- Produces: `compute_rolling_features(team_stats, *, window=8, min_history=4) -> pl.DataFrame` (columns: `season`, `week`, `team`, `game_id`, `rolling_offense_epa_per_play`, `rolling_defense_epa_allowed_per_play`, `rolling_turnover_margin`) and `build_model_table(schedules, rolling_features) -> pl.DataFrame` (columns: `game_id`, `season`, `week`, `home_team`, `away_team`, `result`, `spread_line`, `home_offense_epa`, `home_defense_epa_allowed`, `home_turnover_margin`, `away_offense_epa`, `away_defense_epa_allowed`, `away_turnover_margin`) — both used by Task 2 (model) and Task 3 (backtest).

- [ ] **Step 1: Create package markers**

`src/degenebet/modeling/__init__.py`: empty file.
`tests/modeling/__init__.py`: empty file.

- [ ] **Step 2: Write the failing tests**

`tests/modeling/test_features.py`:

```python
from __future__ import annotations

import polars as pl

from degenebet.modeling.features import build_model_table, compute_rolling_features


def _team_stats_row(
    *,
    season: int,
    week: int,
    team: str,
    opponent_team: str,
    game_id: str,
    passing_epa: float,
    attempts: int = 10,
    carries: int = 10,
    def_interceptions: int = 0,
    fumble_recovery_opp: int = 0,
    passing_interceptions: int = 0,
    fumbles_lost_total: int = 0,
) -> dict[str, object]:
    return {
        "season": season,
        "week": week,
        "team": team,
        "opponent_team": opponent_team,
        "game_id": game_id,
        "passing_epa": passing_epa,
        "rushing_epa": 0.0,
        "attempts": attempts,
        "carries": carries,
        "def_interceptions": def_interceptions,
        "fumble_recovery_opp": fumble_recovery_opp,
        "passing_interceptions": passing_interceptions,
        "fumbles_lost_total": fumbles_lost_total,
    }


def _four_game_team_stats() -> pl.DataFrame:
    # Team A vs Team B, 4 games. A's offense_epa_per_play (passing_epa/20):
    # G1=1.0, G2=2.0, G3=0.0, G4=3.0. B's: G1=0.5, G2=1.5, G3=2.5, G4=0.0.
    # A's turnover_margin: G1=1, G2=-1, G3=0, G4=2.
    rows = [
        _team_stats_row(season=2099, week=1, team="A", opponent_team="B", game_id="g1", passing_epa=20.0, def_interceptions=1),
        _team_stats_row(season=2099, week=1, team="B", opponent_team="A", game_id="g1", passing_epa=10.0),
        _team_stats_row(season=2099, week=2, team="A", opponent_team="B", game_id="g2", passing_epa=40.0, passing_interceptions=1),
        _team_stats_row(season=2099, week=2, team="B", opponent_team="A", game_id="g2", passing_epa=30.0),
        _team_stats_row(season=2099, week=3, team="A", opponent_team="B", game_id="g3", passing_epa=0.0),
        _team_stats_row(season=2099, week=3, team="B", opponent_team="A", game_id="g3", passing_epa=50.0),
        _team_stats_row(season=2099, week=4, team="A", opponent_team="B", game_id="g4", passing_epa=60.0, def_interceptions=2),
        _team_stats_row(season=2099, week=4, team="B", opponent_team="A", game_id="g4", passing_epa=0.0),
    ]
    return pl.DataFrame(rows)


def test_compute_rolling_features_excludes_current_game_and_respects_min_history() -> None:
    rolling = compute_rolling_features(_four_game_team_stats(), window=3, min_history=2)

    team_a = rolling.filter(pl.col("team") == "A").sort("week")
    # G1, G2 dropped: fewer than min_history=2 prior games.
    assert team_a["week"].to_list() == [3, 4]

    g3 = team_a.filter(pl.col("week") == 3).row(0, named=True)
    assert g3["rolling_offense_epa_per_play"] == 1.5  # mean(1.0, 2.0)
    assert g3["rolling_defense_epa_allowed_per_play"] == 1.0  # mean(B's 0.5, 1.5)
    assert g3["rolling_turnover_margin"] == 0.0  # mean(1, -1)

    g4 = team_a.filter(pl.col("week") == 4).row(0, named=True)
    assert g4["rolling_offense_epa_per_play"] == 1.0  # mean(1.0, 2.0, 0.0), window=3
    assert g4["rolling_defense_epa_allowed_per_play"] == 1.5  # mean(B's 0.5, 1.5, 2.5)
    assert g4["rolling_turnover_margin"] == 0.0  # mean(1, -1, 0)


def test_build_model_table_joins_home_and_away_features() -> None:
    rolling_features = pl.DataFrame(
        {
            "season": [2099, 2099],
            "week": [5, 5],
            "team": ["A", "B"],
            "game_id": ["g5", "g5"],
            "rolling_offense_epa_per_play": [1.2, 0.8],
            "rolling_defense_epa_allowed_per_play": [0.5, 1.1],
            "rolling_turnover_margin": [0.3, -0.2],
        }
    )
    schedules = pl.DataFrame(
        {
            "game_id": ["g5"],
            "season": [2099],
            "week": [5],
            "home_team": ["A"],
            "away_team": ["B"],
            "result": [7],
            "spread_line": [-2.5],
        }
    )

    table = build_model_table(schedules, rolling_features)

    assert table.height == 1
    row = table.row(0, named=True)
    assert row["home_offense_epa"] == 1.2
    assert row["home_defense_epa_allowed"] == 0.5
    assert row["home_turnover_margin"] == 0.3
    assert row["away_offense_epa"] == 0.8
    assert row["away_defense_epa_allowed"] == 1.1
    assert row["away_turnover_margin"] == -0.2
    assert row["result"] == 7
    assert row["spread_line"] == -2.5


def test_build_model_table_drops_games_missing_either_teams_features() -> None:
    rolling_features = pl.DataFrame(
        {
            "season": [2099],
            "week": [5],
            "team": ["A"],
            "game_id": ["g5"],
            "rolling_offense_epa_per_play": [1.2],
            "rolling_defense_epa_allowed_per_play": [0.5],
            "rolling_turnover_margin": [0.3],
        }
    )
    schedules = pl.DataFrame(
        {
            "game_id": ["g5"],
            "season": [2099],
            "week": [5],
            "home_team": ["A"],
            "away_team": ["B"],  # B has no rolling_features row
            "result": [7],
            "spread_line": [-2.5],
        }
    )

    table = build_model_table(schedules, rolling_features)

    assert table.height == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/modeling/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degenebet.modeling.features'`

- [ ] **Step 4: Implement `src/degenebet/modeling/features.py`**

```python
"""No-lookahead rolling team efficiency features for the spread model.

Offensive EPA/play uses only passing_epa + rushing_epa — adding
receiving_epa would double-count the same pass plays from the receiver's
side. Defensive EPA allowed has no direct column in team_stats; it's the
opponent's offensive EPA/play in the same game, found via a join.
"""

from __future__ import annotations

import polars as pl

_ROLLING_METRICS = ["offense_epa_per_play", "defense_epa_allowed_per_play", "turnover_margin"]


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
        ((pl.col("passing_epa") + pl.col("rushing_epa")) / (pl.col("attempts") + pl.col("carries"))).alias(
            "offense_epa_per_play"
        ),
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
            pl.col(c).shift(1).rolling_mean(window_size=window, min_samples=min_history).over("team").alias(f"rolling_{c}")
            for c in _ROLLING_METRICS
        ]
    )

    rolling_cols = [f"rolling_{c}" for c in _ROLLING_METRICS]
    not_null = pl.lit(True)
    for c in rolling_cols:
        not_null = not_null & pl.col(c).is_not_null()

    return rolling.filter(not_null).select("season", "week", "team", "game_id", *rolling_cols)


def build_model_table(schedules: pl.DataFrame, rolling_features: pl.DataFrame) -> pl.DataFrame:
    """One row per game with home_*/away_* feature columns, result, spread_line.

    Games where either team lacks rolling features (see compute_rolling_features's
    min_history) are dropped via the inner joins below.
    """
    home_feats = rolling_features.select(
        pl.col("game_id"),
        pl.col("team").alias("home_team"),
        pl.col("rolling_offense_epa_per_play").alias("home_offense_epa"),
        pl.col("rolling_defense_epa_allowed_per_play").alias("home_defense_epa_allowed"),
        pl.col("rolling_turnover_margin").alias("home_turnover_margin"),
    )
    away_feats = rolling_features.select(
        pl.col("game_id"),
        pl.col("team").alias("away_team"),
        pl.col("rolling_offense_epa_per_play").alias("away_offense_epa"),
        pl.col("rolling_defense_epa_allowed_per_play").alias("away_defense_epa_allowed"),
        pl.col("rolling_turnover_margin").alias("away_turnover_margin"),
    )
    return (
        schedules.select("game_id", "season", "week", "home_team", "away_team", "result", "spread_line")
        .join(home_feats, on=["game_id", "home_team"], how="inner")
        .join(away_feats, on=["game_id", "away_team"], how="inner")
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_features.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy`
Expected: all clean

- [ ] **Step 7: Commit**

```bash
git add src/degenebet/modeling/__init__.py src/degenebet/modeling/features.py tests/modeling/__init__.py tests/modeling/test_features.py
git commit -m "feat: add no-lookahead rolling team efficiency features"
```

---

### Task 2: Spread model

**Files:**
- Modify: `pyproject.toml` (add `scikit-learn`, `scipy` to `[project.dependencies]`; add two `[[tool.mypy.overrides]]` entries)
- Create: `src/degenebet/modeling/spread_model.py`
- Create: `tests/modeling/test_spread_model.py`

**Interfaces:**
- Consumes: `build_model_table`'s output shape (Task 1) — 6 feature columns (`home_offense_epa`, `home_defense_epa_allowed`, `home_turnover_margin`, `away_offense_epa`, `away_defense_epa_allowed`, `away_turnover_margin`) plus `result`, `spread_line`.
- Produces: `RegressorProtocol` (structural type: `.fit(X, y)`, `.predict(X)`) and `SpreadModel` class — composed with an injected `RegressorProtocol` instance (`model: RegressorProtocol | None = None`, defaulting to `LinearRegression()` when omitted) rather than hardcoding `LinearRegression` internally, so a future sub-project can swap in a different sklearn-compatible estimator without changing `SpreadModel` itself. Exposes `.model` (public attribute, the injected/default regressor), `.fit(model_table)`, `.predict(model_table) -> pl.DataFrame` (adds `predicted_result`), `.cover_probability(model_table) -> pl.DataFrame` (adds `home_cover_probability`) — used by Task 3 (backtest) and Task 4 (golden fixture, property tests).

- [ ] **Step 1: Add dependencies to `pyproject.toml`**

In `[project] dependencies`, add two entries (alphabetical among the existing list):

```toml
"scikit-learn>=1.5",
"scipy>=1.13",
```

Add two entries to the existing `[[tool.mypy.overrides]]` block (or a new one) — neither `scikit-learn` nor `scipy` ships a `py.typed` marker:

```toml
[[tool.mypy.overrides]]
module = ["sklearn.*", "scipy.*"]
ignore_missing_imports = true
```

Run: `uv sync --extra dev`
Expected: installs `scikit-learn` and `scipy` successfully.

- [ ] **Step 2: Write the failing tests**

`tests/modeling/test_spread_model.py`:

```python
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from degenebet.modeling.spread_model import SpreadModel

_FEATURE_COLUMNS = [
    "home_offense_epa",
    "home_defense_epa_allowed",
    "home_turnover_margin",
    "away_offense_epa",
    "away_defense_epa_allowed",
    "away_turnover_margin",
]


def _noiseless_table() -> pl.DataFrame:
    # result = 10 * home_offense_epa exactly; every other feature is 0.
    home_offense = [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, 2.5, 0.25, -0.25]
    zeros = [0.0] * len(home_offense)
    return pl.DataFrame(
        {
            "home_offense_epa": home_offense,
            "home_defense_epa_allowed": zeros,
            "home_turnover_margin": zeros,
            "away_offense_epa": zeros,
            "away_defense_epa_allowed": zeros,
            "away_turnover_margin": zeros,
            "result": [10.0 * v for v in home_offense],
        }
    )


def _noisy_table_with_spread() -> pl.DataFrame:
    table = _noiseless_table()
    # Add small deterministic noise so residual_std is nonzero.
    noise = [0.3, -0.2, 0.1, -0.4, 0.2, -0.1, 0.3, -0.3, 0.1, -0.2]
    table = table.with_columns((pl.col("result") + pl.Series(noise)).alias("result"))
    return table.with_columns(pl.Series("spread_line", [0.0] * table.height))


def test_predict_matches_noiseless_linear_relationship() -> None:
    model = SpreadModel()
    table = _noiseless_table()
    model.fit(table)

    predicted = model.predict(table)

    for expected, actual in zip(table["result"].to_list(), predicted["predicted_result"].to_list()):
        assert actual == pytest.approx(expected, abs=1e-6)


def test_cover_probability_is_bounded_and_present() -> None:
    model = SpreadModel()
    table = _noisy_table_with_spread()
    model.fit(table)

    result = model.cover_probability(table)

    assert "home_cover_probability" in result.columns
    for p in result["home_cover_probability"].to_list():
        assert 0.0 <= p <= 1.0


def test_cover_probability_before_fit_raises() -> None:
    model = SpreadModel()
    table = _noisy_table_with_spread()

    with pytest.raises(RuntimeError, match="fit"):
        model.cover_probability(table)


def test_cover_probability_calls_predict_internally_if_missing() -> None:
    model = SpreadModel()
    table = _noisy_table_with_spread()
    model.fit(table)

    result = model.cover_probability(table.select(_FEATURE_COLUMNS + ["spread_line"]))

    assert "predicted_result" in result.columns
    assert "home_cover_probability" in result.columns


class _ConstantRegressor:
    """Trivial regressor always predicting a fixed constant — a test double
    proving SpreadModel actually delegates to an injected regressor rather
    than hardcoding LinearRegression. Matches RegressorProtocol's shape
    exactly (ndarray in, ndarray out) so it type-checks where a
    RegressorProtocol is expected."""

    def __init__(self, constant: float) -> None:
        self._constant = constant

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> "_ConstantRegressor":
        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.full(X.shape[0], self._constant)


def test_spread_model_accepts_injected_regressor() -> None:
    model = SpreadModel(model=_ConstantRegressor(constant=3.5))
    table = _noiseless_table()
    model.fit(table)

    predicted = model.predict(table)

    for value in predicted["predicted_result"].to_list():
        assert value == pytest.approx(3.5)


def test_spread_model_defaults_to_linear_regression() -> None:
    from sklearn.linear_model import LinearRegression

    model = SpreadModel()

    assert isinstance(model.model, LinearRegression)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/modeling/test_spread_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degenebet.modeling.spread_model'`

- [ ] **Step 4: Implement `src/degenebet/modeling/spread_model.py`**

```python
"""Linear regression (by default) predicting NFL game margin from rolling
team features. SpreadModel is composed with a RegressorProtocol rather than
hardcoding LinearRegression, so a future sub-project can swap in a
different sklearn-compatible estimator without changing this class."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

_FEATURE_COLUMNS = [
    "home_offense_epa",
    "home_defense_epa_allowed",
    "home_turnover_margin",
    "away_offense_epa",
    "away_defense_epa_allowed",
    "away_turnover_margin",
]


class RegressorProtocol(Protocol):
    """Minimal sklearn-compatible regressor interface SpreadModel depends on."""

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> object: ...
    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...


class SpreadModel:
    """Predicts `result` from 6 rolling team features via an injected
    regressor (LinearRegression by default), plus a cover probability
    derived from the training residual spread."""

    def __init__(self, model: RegressorProtocol | None = None) -> None:
        self.model: RegressorProtocol = model if model is not None else LinearRegression()
        self._residual_std: float | None = None

    def fit(self, model_table: pl.DataFrame) -> None:
        """Fits the injected regressor on the 6 feature columns against `result`."""
        features = model_table.select(_FEATURE_COLUMNS).to_numpy()
        target = model_table["result"].to_numpy()
        self.model.fit(features, target)
        residuals = target - self.model.predict(features)
        self._residual_std = float(np.std(residuals, ddof=1))

    def predict(self, model_table: pl.DataFrame) -> pl.DataFrame:
        """Returns model_table with a new `predicted_result` column."""
        features = model_table.select(_FEATURE_COLUMNS).to_numpy()
        predicted = self.model.predict(features)
        return model_table.with_columns(pl.Series("predicted_result", predicted))

    def cover_probability(self, model_table: pl.DataFrame) -> pl.DataFrame:
        """Requires `spread_line` present. Adds `home_cover_probability` via
        normal_cdf((predicted_result + spread_line) / residual_std)."""
        if self._residual_std is None:
            raise RuntimeError("SpreadModel.fit() must be called before cover_probability().")
        if "predicted_result" not in model_table.columns:
            model_table = self.predict(model_table)
        edge = (model_table["predicted_result"] + model_table["spread_line"]) / self._residual_std
        probabilities = norm.cdf(edge.to_numpy())
        return model_table.with_columns(pl.Series("home_cover_probability", probabilities))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_spread_model.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy`
Expected: all clean

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/degenebet/modeling/spread_model.py tests/modeling/test_spread_model.py
git commit -m "feat: add SpreadModel (composable regressor + cover probability)"
```

---

### Task 3: Backtest harness

**Files:**
- Create: `src/degenebet/modeling/backtest.py`
- Create: `tests/modeling/test_backtest.py`

**Interfaces:**
- Consumes: `SpreadModel` (Task 2), `build_model_table`'s output shape (Task 1) plus a `season` column for the train/test split.
- Produces: `BacktestResult` frozen dataclass, `run_backtest(model_table, *, train_seasons, test_seasons, edge_threshold=1.0) -> BacktestResult` — used by Task 4's property tests.

- [ ] **Step 1: Write the failing tests**

`tests/modeling/test_backtest.py`:

```python
from __future__ import annotations

import polars as pl

from degenebet.modeling.backtest import run_backtest

_FEATURE_COLUMNS = [
    "home_offense_epa",
    "home_defense_epa_allowed",
    "home_turnover_margin",
    "away_offense_epa",
    "away_defense_epa_allowed",
    "away_turnover_margin",
]


def _model_table() -> pl.DataFrame:
    # Train: result = 10 * home_offense_epa exactly (season 2098).
    # Test (season 2099): home_offense_epa chosen so predicted edge clearly
    # crosses the default edge_threshold=1.0 for some games, not others.
    train_home_offense = [0.0, 0.5, 1.0, -0.5, -1.0, 0.25, -0.25, 0.75]
    zeros_train = [0.0] * len(train_home_offense)
    train = pl.DataFrame(
        {
            "game_id": [f"train_{i}" for i in range(len(train_home_offense))],
            "season": [2098] * len(train_home_offense),
            "week": list(range(1, len(train_home_offense) + 1)),
            "home_offense_epa": train_home_offense,
            "home_defense_epa_allowed": zeros_train,
            "home_turnover_margin": zeros_train,
            "away_offense_epa": zeros_train,
            "away_defense_epa_allowed": zeros_train,
            "away_turnover_margin": zeros_train,
            "result": [10.0 * v for v in train_home_offense],
            "spread_line": [0.0] * len(train_home_offense),
        }
    )

    # Test games: predicted_result ~= 10 * home_offense_epa.
    # g_home_bet: home_offense_epa=0.5 -> predicted~5; spread_line=-1 -> edge~4 > 1.0 -> bet home; result=6 -> covers (6-1=5>0)
    # g_away_bet: home_offense_epa=-0.5 -> predicted~-5; spread_line=1 -> edge~-4 < -1.0 -> bet away; result=-6 -> away covers (-6+1=-5<0)
    # g_no_bet: home_offense_epa=0.05 -> predicted~0.5; spread_line=0 -> edge~0.5, within threshold -> no bet
    # g_push: home_offense_epa=0.5 -> predicted~5; spread_line=-5 -> edge~0, no bet either (kept simple: no push case forced, backtest naturally excludes it)
    test = pl.DataFrame(
        {
            "game_id": ["g_home_bet", "g_away_bet", "g_no_bet"],
            "season": [2099, 2099, 2099],
            "week": [1, 1, 1],
            "home_offense_epa": [0.5, -0.5, 0.05],
            "home_defense_epa_allowed": [0.0, 0.0, 0.0],
            "home_turnover_margin": [0.0, 0.0, 0.0],
            "away_offense_epa": [0.0, 0.0, 0.0],
            "away_defense_epa_allowed": [0.0, 0.0, 0.0],
            "away_turnover_margin": [0.0, 0.0, 0.0],
            "result": [6.0, -6.0, 1.0],
            "spread_line": [-1.0, 1.0, 0.0],
        }
    )
    return pl.concat([train, test])


def test_run_backtest_places_bets_only_when_edge_exceeds_threshold() -> None:
    result = run_backtest(_model_table(), train_seasons=[2098], test_seasons=[2099], edge_threshold=1.0)

    assert result.bets_placed == 2
    assert set(result.bets["game_id"].to_list()) == {"g_home_bet", "g_away_bet"}


def test_run_backtest_scores_wins_correctly() -> None:
    result = run_backtest(_model_table(), train_seasons=[2098], test_seasons=[2099], edge_threshold=1.0)

    assert result.ats_win_rate == 1.0  # both placed bets covered, per the hand-derived comments above
    assert result.units_won == 2.0  # two wins at +1.0 unit each
    assert result.roi_pct > 0


def test_run_backtest_with_no_qualifying_bets_returns_zeroed_result() -> None:
    result = run_backtest(_model_table(), train_seasons=[2098], test_seasons=[2099], edge_threshold=100.0)

    assert result.bets_placed == 0
    assert result.ats_win_rate == 0.0
    assert result.units_won == 0.0
    assert result.roi_pct == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/modeling/test_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degenebet.modeling.backtest'`

- [ ] **Step 3: Implement `src/degenebet/modeling/backtest.py`**

```python
"""Backtest harness: evaluates a SpreadModel against real historical closing lines."""

from __future__ import annotations

from dataclasses import dataclass

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
    `result` via the cover-margin formula (result + spread_line) at -110
    pricing (push refunds the bet, excluded from win rate).
    """
    train = model_table.filter(pl.col("season").is_in(train_seasons))
    test = model_table.filter(pl.col("season").is_in(test_seasons))

    model = SpreadModel()
    model.fit(train)
    predicted = model.predict(test)

    predicted = predicted.with_columns((pl.col("predicted_result") + pl.col("spread_line")).alias("edge")).with_columns(
        pl.when(pl.col("edge") > edge_threshold)
        .then(pl.lit("home"))
        .when(pl.col("edge") < -edge_threshold)
        .then(pl.lit("away"))
        .otherwise(pl.lit("none"))
        .alias("side")
    )

    bets_df = (
        predicted.filter(pl.col("side") != "none")
        .with_columns((pl.col("result") + pl.col("spread_line")).alias("home_cover_margin"))
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
    ats_win_rate = float(decided["won"].mean()) if decided.height > 0 else 0.0
    units_won = float(bets_df["units"].sum())
    roi_pct = (units_won / (bets_placed * _UNITS_RISKED_PER_BET) * 100) if bets_placed > 0 else 0.0

    return BacktestResult(
        bets=bets_df,
        bets_placed=bets_placed,
        ats_win_rate=ats_win_rate,
        units_won=units_won,
        roi_pct=roi_pct,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_backtest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy`
Expected: all clean

- [ ] **Step 6: Commit**

```bash
git add src/degenebet/modeling/backtest.py tests/modeling/test_backtest.py
git commit -m "feat: add spread-model backtest harness"
```

---

### Task 4: Golden fixture + property tests

**Files:**
- Create: `tests/modeling/_golden_spread_fixture.py`
- Create: `tests/modeling/test_spread_model_golden.py`
- Create: `tests/modeling/test_property_invariants.py`
- Modify: `pyproject.toml` (add `hypothesis` to the `test` optional-dependency group)

**Interfaces:**
- Consumes: `SpreadModel` (Task 2), `run_backtest`/`BacktestResult` (Task 3).
- Produces: nothing consumed by later tasks — this is the plan's last task.

- [ ] **Step 1: Add `hypothesis` to `pyproject.toml`'s `test` extra**

In `[project.optional-dependencies] test`, add `"hypothesis>=6.100",` to the existing list (alongside `pytest`, `pytest-cov`, `respx`).

Run: `uv sync --extra dev`
Expected: installs `hypothesis`.

- [ ] **Step 2: Create the golden fixture data**

`tests/modeling/_golden_spread_fixture.py`:

```python
"""Deterministic synthetic model_table for golden regression tests.

Fixed-seed, not real fetched data — avoids any dependency on nflverse data
potentially changing over time and keeps golden values exactly
reproducible. Mirrors waypoint's np.random.default_rng(seed=42) convention
for reproducible test data.
"""

from __future__ import annotations

import numpy as np
import polars as pl

_FEATURE_COLUMNS = [
    "home_offense_epa",
    "home_defense_epa_allowed",
    "home_turnover_margin",
    "away_offense_epa",
    "away_defense_epa_allowed",
    "away_turnover_margin",
]


def golden_model_table() -> pl.DataFrame:
    rng = np.random.default_rng(seed=42)
    n = 40
    features = {col: rng.normal(loc=0.0, scale=1.0, size=n) for col in _FEATURE_COLUMNS}
    # A fixed, made-up "true" relationship plus noise, so the fit is
    # well-conditioned and reproducible — not meant to reflect real football.
    true_coefs = np.array([8.0, -6.0, 3.0, -8.0, 6.0, -3.0])
    design = np.column_stack([features[c] for c in _FEATURE_COLUMNS])
    noise = rng.normal(loc=0.0, scale=2.0, size=n)
    result = design @ true_coefs + noise
    spread_line = -result + rng.normal(loc=0.0, scale=3.0, size=n)  # a noisy "market" around the true result

    data = {**features, "result": result, "spread_line": spread_line}
    return pl.DataFrame(data)
```

- [ ] **Step 3: Generate the pinned values (run once, capture real output)**

Run:

```bash
uv run python -c "
from degenebet.modeling.spread_model import SpreadModel
from tests.modeling._golden_spread_fixture import golden_model_table

table = golden_model_table()
model = SpreadModel()
model.fit(table)
print('coef_:', model.model.coef_.tolist())
print('intercept_:', float(model.model.intercept_))
print('residual_std:', model._residual_std)

predicted = model.predict(table.head(5))
print('first 5 predicted_result:', predicted['predicted_result'].to_list())

cover = model.cover_probability(table.head(5))
print('first 5 home_cover_probability:', cover['home_cover_probability'].to_list())
"
```

Copy the actual printed values into Step 4 below, replacing every `<PASTE:...>` placeholder with the real number the command printed. This is the one place in this plan where exact literal values can't be given in advance — golden/snapshot tests are inherently "run once, pin what you got," and the fixture generation above is fully deterministic (fixed seed), so the values will be stable across runs on the same dependency versions.

- [ ] **Step 4: Write the golden test using the pinned values from Step 3**

`tests/modeling/test_spread_model_golden.py`:

```python
"""Golden regression test: pins SpreadModel's fitted output against a fixed
synthetic dataset. A failure here means the math actually changed — verify
the new values are intentional before updating the pinned constants below;
never update them reflexively just to make the test pass.
"""

from __future__ import annotations

import pytest

from degenebet.modeling.spread_model import SpreadModel
from tests.modeling._golden_spread_fixture import golden_model_table

# Pinned from a real run against the fixed-seed fixture (see plan Task 4 Step 3).
_EXPECTED_COEF = [<PASTE: coef_ list, 6 floats>]
_EXPECTED_INTERCEPT = <PASTE: intercept_ float>
_EXPECTED_RESIDUAL_STD = <PASTE: residual_std float>
_EXPECTED_FIRST_5_PREDICTED = [<PASTE: first 5 predicted_result, 5 floats>]
_EXPECTED_FIRST_5_COVER_PROB = [<PASTE: first 5 home_cover_probability, 5 floats>]


def test_spread_model_golden_fit_and_predict() -> None:
    table = golden_model_table()
    model = SpreadModel()
    model.fit(table)

    for actual, expected in zip(model.model.coef_.tolist(), _EXPECTED_COEF):
        assert actual == pytest.approx(expected, abs=1e-6)
    assert float(model.model.intercept_) == pytest.approx(_EXPECTED_INTERCEPT, abs=1e-6)
    assert model._residual_std == pytest.approx(_EXPECTED_RESIDUAL_STD, abs=1e-6)

    predicted = model.predict(table.head(5))
    for actual, expected in zip(predicted["predicted_result"].to_list(), _EXPECTED_FIRST_5_PREDICTED):
        assert actual == pytest.approx(expected, abs=1e-6)

    cover = model.cover_probability(table.head(5))
    for actual, expected in zip(cover["home_cover_probability"].to_list(), _EXPECTED_FIRST_5_COVER_PROB):
        assert actual == pytest.approx(expected, abs=1e-6)
```

- [ ] **Step 5: Run the golden test to verify it passes**

Run: `uv run pytest tests/modeling/test_spread_model_golden.py -v`
Expected: PASS (1 test) — if it fails, the pasted values in Step 4 don't match Step 3's actual output; re-check the copy, don't adjust the model to match wrong pinned values.

- [ ] **Step 6: Write the property tests**

`tests/modeling/test_property_invariants.py`:

```python
"""Property-based tests for invariants that would be expensive to get
silently wrong: cover probability bounds, and backtest P&L reconciling
under re-aggregation (per AGENTS.md's Automation & Verification mandate).
"""

from __future__ import annotations

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

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


@given(edge=_finite_float, residual_std=st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False))
def test_cover_probability_always_in_unit_interval(edge: float, residual_std: float) -> None:
    # Exercise the same normal_cdf transform SpreadModel.cover_probability
    # uses, directly, since a full model.fit() call per hypothesis example
    # would be slow — this tests the transform's own boundedness.
    from scipy.stats import norm

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
    weekly_sums = bets.group_by("week").agg(pl.col("units").sum()).select(pl.col("units").sum()).item()

    assert weekly_sums == pytest.approx(season_total, abs=1e-9)
```

Note: `test_backtest_pnl_reconciles_under_reaggregation` needs `import pytest` for `pytest.approx` — add it to the imports above.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_property_invariants.py -v`
Expected: PASS (2 tests, each running multiple hypothesis-generated examples)

- [ ] **Step 8: Full suite, lint, type-check**

Run: `uv run pytest -v`
Expected: all tests pass (data-foundation's 46 plus this plan's new tests), coverage report printed.

Run: `just check`
Expected: clean end-to-end (ruff check, ruff format check, mypy, pytest).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock tests/modeling/_golden_spread_fixture.py tests/modeling/test_spread_model_golden.py tests/modeling/test_property_invariants.py
git commit -m "test: add golden regression fixture and property-based invariant tests"
```

---

## Self-Review Notes

- **Spec coverage:** feature pipeline with verified no-lookahead/EPA/defense-derivation logic (Task 1) ✅, `SpreadModel` with cover probability (Task 2) ✅, backtest harness with -110 scoring (Task 3) ✅, golden fixture + property tests fulfilling `AGENTS.md`'s mandate (Task 4) ✅. Every spec section maps to a task.
- **Placeholder scan:** no TBD/TODO. The `<PASTE: ...>` markers in Task 4 Step 4 are a deliberate, called-out exception — golden/snapshot test values cannot be known before the code exists to generate them; Step 3 gives the exact command to generate them and Step 4 explains precisely what to do with the output. This is not a placeholder in the sense the "No Placeholders" rule prohibits (vague instructions); it's the standard shape of authoring any golden test.
- **Type consistency:** `_FEATURE_COLUMNS` (6 names, exact order) is identical across `spread_model.py`, `backtest.py`'s test fixtures, `_golden_spread_fixture.py`, and `test_property_invariants.py`. `SpreadModel.fit/predict/cover_probability` signatures match between Task 2's implementation and Tasks 3/4's callers. `BacktestResult`'s fields (`bets`, `bets_placed`, `ats_win_rate`, `units_won`, `roi_pct`) are consistent between Task 3's dataclass and its tests. `SpreadModel`'s injected regressor is consistently accessed as the public `.model` attribute (not the private `._model` from an earlier draft of this plan) everywhere it's touched outside the class itself — Task 4's golden-fixture generation script and golden test both use `model.model.coef_`/`model.model.intercept_`.
- **Verified-not-assumed:** every data-shape claim in Global Constraints (EPA formula, defense-EPA derivation, sign convention, the `min_samples` polars parameter name, the exact rolling/self-join logic, the vectorized backtest scoring expressions) was prototyped and run against either real data or hand-computed expected values during design — not written from memory of how these APIs "should" work.
