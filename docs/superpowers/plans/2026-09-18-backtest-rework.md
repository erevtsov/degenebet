# Backtest Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `backtest.py`'s single-split `run_backtest` baseline with a market-aware, sizing-aware `Backtest` class scoring real per-bet American odds, plus `WalkForwardSplit` and a compounding dollar bankroll trajectory.

**Architecture:** `WalkForwardSplit` joins `SingleSplit` in `splits.py` as a second `SplitStrategy`. `backtest.py` is fully replaced: `SizingStrategy`/`FlatSizing` decide bet size (`stake`), `Backtest.run()`/`run_folds()` mirror `Efficacy.evaluate()`/`evaluate_folds()`'s shape, and `compute_bankroll_trajectory` is a small, separate, sequential pass that turns `run_folds()`'s already-computed per-fold results into a real dollar-denominated trajectory.

**Tech Stack:** Python 3.12, polars, pytest, `hypothesis`, mypy --strict, ruff, marimo/altair for the notebook.

**Spec:** `docs/superpowers/specs/2026-09-18-backtest-rework-design.md`

## Global Constraints

- Use polars for all tabular data; never pandas.
- `from collections.abc import Callable, Iterable, Iterator` — not `typing` — this repo's ruff config (`select = ["E", "F", "I", "UP"]`) rejects the `typing` versions when `from __future__ import annotations` is present. `Protocol`/`NamedTuple`/`TypedDict` still come from `typing`.
- `mypy --strict` only checks `src/degenebet` (`[tool.mypy] packages = ["degenebet"]`) — test files are not type-checked, but everything must still pass `ruff check`/`ruff format`.
- `ats_win_rate` (not `directional_accuracy`) is `Backtest`'s win-rate field name — it is NOT the same concept as `Efficacy.evaluate()`'s `directional_accuracy`; do not conflate the two "win rate"s (see `architecture-notes.md`'s explicit warning).
- `stake` means the amount *risked* on a bet (not, as today's `run_backtest` has it, an amount sized to win 1.0) — this is a deliberate convention change confirmed in the spec's "Terminology" section. `units = stake * win_multiplier` if won, `-stake` if lost, `0` if push.
- `FlatSizing` divides a normalized pool of `1.0` evenly across every bet in one `size()` call (`stake_i = 1.0 / N`) — not a constant per-bet stake. This is what makes per-fold computation independent of a live bankroll; see "Compounding bankroll" in the spec.
- `roi_pct = units_won / sum(stake) * 100` — simpler than today's `bets_placed * 1.1` denominator, since `stake` already means amount risked.
- `Backtest` never filters unplayed/unpriced games itself — the caller opts in explicitly before calling `run()`/`run_folds()` (mirrors `SpreadModel`'s null-feature guard and `Efficacy._compute_metrics`'s null-`result` guard). `Backtest` raises `ValueError` on nulls in rows that exist; an empty (0-row) input is legitimate and returns a zeroed result, not an error.
- `run_backtest` and the old `BacktestResult` are fully deleted, not kept alongside the new API — this project's established pattern (DataAccess redesign, `build_model_table`'s retirement) never keeps two parallel implementations of the same concept.
- **Task 2 intentionally breaks `tests/modeling/test_property_invariants.py` and `tests/modeling/test_real_data_sanity.py`** (both import `run_backtest`, deleted in Task 2) — this is expected, not a regression to fix in Task 2. Task 5 migrates both to the new API. Confirm after Task 2 that these two files fail with an import error specifically (not some other error), and do not attempt to fix them before Task 5.
- Test-first for every task: write the failing test, confirm it fails, then implement.
- Never mock what can be faked with real in-memory Polars DataFrames — every test in this plan uses small, hand-constructed real data, never a mock.
- `notebooks/03_backtest.py` is not part of `just check` (notebooks aren't CI-gated), but must pass `uv run marimo check notebooks/03_backtest.py` and a headless `uv run marimo export html notebooks/03_backtest.py -o <tmp>.html` with no error output before committing, per `CLAUDE.md`'s Notebooks convention. It must also pass `uv run ruff check`/`ruff format`.
- Reads for the notebook go through `DataAccess` only, against the local cache — never live `nflreadpy`/SharpAPI calls.

---

### Task 1: `WalkForwardSplit`

**Files:**
- Modify: `src/degenebet/modeling/splits.py`
- Modify: `tests/modeling/test_splits.py`

**Interfaces:**
- Consumes: `Split`, `SplitStrategy` (already exist in `splits.py`).
- Produces: `WalkForwardSplit` with `__init__(self, warmup_seasons: int) -> None` and `splits(self, data: pl.DataFrame) -> Iterator[Split]`.

This task also fixes a stale docstring: `SingleSplit`'s current docstring says `"Mirrors run_backtest's existing train_seasons/test_seasons filtering and leakage guards, extracted here rather than reimplemented."` — `run_backtest` is deleted in Task 2, making "mirrors run_backtest" inaccurate. Fix it now (independent of Task 2's timing) to describe `SingleSplit` on its own terms.

- [ ] **Step 1: Write the failing tests**

In `tests/modeling/test_splits.py`, change the import line to:

```python
from degenebet.modeling.splits import SingleSplit, Split, WalkForwardSplit, iterate_folds
```

Then append, after the existing `test_iterate_folds_uses_a_fresh_model_per_fold` test:

```python
def _walk_forward_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2020, 2020, 2020, 2021, 2021, 2021],
            "week": [1, 2, 3, 1, 2, 3],
            "gameweek": [202001, 202002, 202003, 202101, 202102, 202103],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )


def test_walk_forward_split_produces_one_fold_per_post_warmup_gameweek() -> None:
    strategy = WalkForwardSplit(warmup_seasons=1)

    splits = list(strategy.splits(_walk_forward_data()))

    assert [s.test["gameweek"].to_list() for s in splits] == [[202101], [202102], [202103]]


def test_walk_forward_split_train_expands_and_never_includes_current_or_future_gameweek() -> None:
    strategy = WalkForwardSplit(warmup_seasons=1)

    splits = list(strategy.splits(_walk_forward_data()))

    assert splits[0].train["gameweek"].to_list() == [202001, 202002, 202003]
    assert splits[1].train["gameweek"].to_list() == [202001, 202002, 202003, 202101]
    assert splits[2].train["gameweek"].to_list() == [202001, 202002, 202003, 202101, 202102]
    # Never includes the test gameweek itself or anything after it.
    for split in splits:
        assert split.train["gameweek"].max() < split.test["gameweek"][0]


def test_walk_forward_split_first_test_fold_is_first_gameweek_of_correct_season() -> None:
    strategy = WalkForwardSplit(warmup_seasons=1)

    first = next(iter(strategy.splits(_walk_forward_data())))

    assert first.test["gameweek"].to_list() == [202101]


def test_walk_forward_split_raises_when_warmup_leaves_no_train_data() -> None:
    strategy = WalkForwardSplit(warmup_seasons=0)

    with pytest.raises(ValueError, match="No rows in data before gameweek"):
        list(strategy.splits(_walk_forward_data()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/modeling/test_splits.py -v`
Expected: FAIL with `ImportError: cannot import name 'WalkForwardSplit' from 'degenebet.modeling.splits'`

- [ ] **Step 3: Write the implementation**

In `src/degenebet/modeling/splits.py`, replace `SingleSplit`'s docstring:

```python
class SingleSplit:
    """One (train, test) pair by season list, with a leakage guard
    against overlapping train/test seasons."""
```

Then add `WalkForwardSplit` immediately after `SingleSplit`'s `splits()` method (before the `Model` protocol):

```python
class WalkForwardSplit:
    """One (train, test) pair per gameweek from the first post-warmup
    season onward, expanding window (train grows every fold, never
    shrinks or slides). Does not filter unplayed rows -- same rule as
    SingleSplit; a fold that walks into a partially-played or future week
    just has null-result rows in its test, which Backtest's own guard
    catches if that fold ever reaches run()/run_folds() unfiltered."""

    def __init__(self, warmup_seasons: int) -> None:
        self.warmup_seasons = warmup_seasons

    def splits(self, data: pl.DataFrame) -> Iterator[Split]:
        seasons = sorted(data["season"].unique().to_list())
        first_test_season = seasons[0] + self.warmup_seasons
        gameweeks = sorted(
            data.filter(pl.col("season") >= first_test_season)["gameweek"].unique().to_list()
        )
        for gw in gameweeks:
            train = data.filter(pl.col("gameweek") < gw)
            test = data.filter(pl.col("gameweek") == gw)
            if train.height == 0:
                raise ValueError(
                    f"No rows in data before gameweek={gw} "
                    f"(warmup_seasons={self.warmup_seasons} too small?)"
                )
            yield Split(train=train, test=test)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_splits.py -v`
Expected: PASS (9 tests: 5 existing + 4 new)

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/degenebet/modeling/splits.py tests/modeling/test_splits.py && uv run ruff format src/degenebet/modeling/splits.py tests/modeling/test_splits.py && uv run mypy`
Expected: all clean

- [ ] **Step 6: Commit**

```bash
git add src/degenebet/modeling/splits.py tests/modeling/test_splits.py
git commit -m "feat: add WalkForwardSplit"
```

---

### Task 2: `backtest.py` full replace through `Backtest.run()`

**Files:**
- Modify: `src/degenebet/modeling/backtest.py` (full replace — deletes `run_backtest`, old `BacktestResult`, `_UNITS_RISKED_PER_BET`)
- Modify: `tests/modeling/test_backtest.py` (full replace)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_win_multiplier(odds: int) -> float`. `SizingStrategy` protocol (`size(self, predictions: pl.DataFrame) -> pl.DataFrame`). `FlatSizing` (implements `SizingStrategy`). `_decide_bets(predictions: pl.DataFrame, edge_threshold: float) -> pl.DataFrame` (private). `_score_bets(sized: pl.DataFrame) -> pl.DataFrame` (private). `_BetsSummary` (private `TypedDict`: `bets_placed: int`, `ats_win_rate: float`, `units_won: float`, `roi_pct: float`). `_summarize_bets(bets_df: pl.DataFrame) -> _BetsSummary` (private). `BacktestResult` (new shape: `bets`, `bets_placed`, `ats_win_rate`, `units_won`, `roi_pct`, `by_fold: pl.DataFrame | None`). `Backtest` with `__init__(self, sizing_strategy: SizingStrategy, edge_threshold: float = 1.0) -> None` and `run(self, predictions: pl.DataFrame) -> BacktestResult`. **`Backtest.run_folds` is added in Task 3, not here** — do not implement it in this task.

**This task deletes `run_backtest`.** Per the plan's Global Constraints, this intentionally breaks `tests/modeling/test_property_invariants.py` and `tests/modeling/test_real_data_sanity.py` until Task 5. After this task, run `uv run pytest tests/modeling/test_property_invariants.py tests/modeling/test_real_data_sanity.py` and confirm both fail with an import error naming `run_backtest` (not some other error) — do not fix them now.

- [ ] **Step 1: Write the failing tests**

Create `tests/modeling/test_backtest.py` (this replaces the file entirely — delete its current content first):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/modeling/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` (the old `backtest.py` has no `Backtest`/`FlatSizing`/`_win_multiplier`)

- [ ] **Step 3: Write the implementation**

Replace `src/degenebet/modeling/backtest.py` entirely with:

```python
"""Backtest: market-aware, sizing-aware scoring of a Model's predictions
against real historical closing lines and real per-bet American odds. See
docs/superpowers/specs/2026-09-18-backtest-rework-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

import polars as pl


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
    def size(self, predictions: pl.DataFrame) -> pl.DataFrame: ...

    """predictions already has edge, side, and win_multiplier computed
    (win_multiplier so a future odds-aware strategy, e.g. Kelly, can size
    against the real payout, not just the edge). Adds a stake column."""


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
    Efficacy's _compute_metrics."""
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
        decided = _decide_bets(predictions, self.edge_threshold)
        bets_df = _score_bets(self.sizing_strategy.size(decided))
        return BacktestResult(bets=bets_df, by_fold=None, **_summarize_bets(bets_df))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_backtest.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Confirm the two files this task intentionally breaks fail the expected way**

Run: `uv run pytest tests/modeling/test_property_invariants.py tests/modeling/test_real_data_sanity.py -v`
Expected: both FAIL with an error naming `run_backtest` (e.g. `ImportError: cannot import name 'run_backtest' from 'degenebet.modeling.backtest'`). This is expected per Global Constraints — do not attempt a fix here.

- [ ] **Step 6: Lint, format, type-check**

Run: `uv run ruff check src/degenebet/modeling/backtest.py tests/modeling/test_backtest.py && uv run ruff format src/degenebet/modeling/backtest.py tests/modeling/test_backtest.py && uv run mypy`
Expected: all clean (mypy checks `src/` only, so the two intentionally-broken test files don't affect it)

- [ ] **Step 7: Commit**

```bash
git add src/degenebet/modeling/backtest.py tests/modeling/test_backtest.py
git commit -m "feat: replace run_backtest with market-aware Backtest.run"
```

---

### Task 3: `Backtest.run_folds()`

**Files:**
- Modify: `src/degenebet/modeling/backtest.py`
- Modify: `tests/modeling/test_backtest.py`

**Interfaces:**
- Consumes: `FoldPredictions` from `src/degenebet/modeling/splits.py` (existing). `_decide_bets`, `_score_bets`, `_summarize_bets`, `BacktestResult` from Task 2 (same file).
- Produces: `Backtest.run_folds(self, folds: Iterable[FoldPredictions]) -> BacktestResult`.

**`by_fold` shape:** one row per fold — `fold: int`, `bets_placed`, `ats_win_rate`, `units_won`, `roi_pct` (no in-sample/out-of-sample sample tag, unlike `Efficacy`'s `by_fold` — `Backtest` never scores in-sample data at all). **Pooled stats:** computed by concatenating every fold's scored bets and calling `_summarize_bets` once on the pooled frame — never by averaging each fold's already-computed stats.

- [ ] **Step 1: Write the failing tests**

Replace `tests/modeling/test_backtest.py`'s top-of-file imports with:

```python
from __future__ import annotations

import polars as pl
import pytest

from degenebet.modeling.backtest import Backtest, FlatSizing, _win_multiplier
from degenebet.modeling.splits import FoldPredictions
from degenebet.modeling.spread_model import SpreadModel
```

Then append, after `test_backtest_run_does_not_raise_on_null_odds_for_the_side_not_bet`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/modeling/test_backtest.py -v`
Expected: FAIL with `AttributeError: 'Backtest' object has no attribute 'run_folds'`

- [ ] **Step 3: Write the implementation**

In `src/degenebet/modeling/backtest.py`, add `from collections.abc import Iterable` to the imports (before `from dataclasses import dataclass`, matching isort's stdlib-alphabetical order) and `from degenebet.modeling.splits import FoldPredictions` as a new first-party import group after the third-party imports:

```python
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

import polars as pl

from degenebet.modeling.splits import FoldPredictions
```

Then add this method to the `Backtest` class, after `run`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_backtest.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/degenebet/modeling/backtest.py tests/modeling/test_backtest.py && uv run ruff format src/degenebet/modeling/backtest.py tests/modeling/test_backtest.py && uv run mypy`
Expected: all clean. If mypy reports a `dict-item` error on the `BacktestResult(...)` call sites, `by_fold_rows`'s type annotation must be `list[dict[str, object]]`, not `list[dict[str, float | int]]` — a `_BetsSummary` TypedDict value doesn't structurally satisfy a `dict[str, float | int]` value type under `--strict`.

- [ ] **Step 6: Commit**

```bash
git add src/degenebet/modeling/backtest.py tests/modeling/test_backtest.py
git commit -m "feat: add Backtest.run_folds"
```

---

### Task 4: `BankrollTrajectory`, `compute_bankroll_trajectory`

**Files:**
- Modify: `src/degenebet/modeling/backtest.py`
- Modify: `tests/modeling/test_backtest.py`

**Interfaces:**
- Consumes: `BacktestResult.by_fold` (a `pl.DataFrame` with a `fold`/`units_won` column, produced by Task 3's `run_folds`).
- Produces: `BankrollTrajectory` (dataclass: `by_fold: pl.DataFrame`, `starting_bankroll: float`, `ending_bankroll: float`, `total_pnl: float`). `compute_bankroll_trajectory(by_fold: pl.DataFrame, starting_bankroll: float) -> BankrollTrajectory`.

- [ ] **Step 1: Write the failing tests**

In `tests/modeling/test_backtest.py`, change the import line:

```python
from degenebet.modeling.backtest import (
    Backtest,
    FlatSizing,
    _win_multiplier,
    compute_bankroll_trajectory,
)
```

Then append, at the end of the file:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/modeling/test_backtest.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_bankroll_trajectory' from 'degenebet.modeling.backtest'`

- [ ] **Step 3: Write the implementation**

Append to `src/degenebet/modeling/backtest.py`:

```python
@dataclass(frozen=True)
class BankrollTrajectory:
    by_fold: pl.DataFrame
    starting_bankroll: float
    ending_bankroll: float
    total_pnl: float


def compute_bankroll_trajectory(
    by_fold: pl.DataFrame, starting_bankroll: float
) -> BankrollTrajectory:
    """Sequentially compounds BacktestResult.by_fold's per-fold units_won
    (a normalized fractional return under FlatSizing's evenly-divided
    pool) into a dollar-denominated trajectory:
    bankroll_after = bankroll_before * (1 + units_won). Requires by_fold
    to already be in time order -- WalkForwardSplit's `fold` index (0, 1,
    2, ... in gameweek order) guarantees this as long as by_fold isn't
    re-sorted before calling this. Only correct for a SizingStrategy
    whose allocation is linear in the pool size (FlatSizing qualifies;
    see the design spec's Non-goals for what doesn't)."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_backtest.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/degenebet/modeling/backtest.py tests/modeling/test_backtest.py && uv run ruff format src/degenebet/modeling/backtest.py tests/modeling/test_backtest.py && uv run mypy`
Expected: all clean

- [ ] **Step 6: Commit**

```bash
git add src/degenebet/modeling/backtest.py tests/modeling/test_backtest.py
git commit -m "feat: add compute_bankroll_trajectory"
```

---

### Task 5: Migrate `test_property_invariants.py` and `test_real_data_sanity.py`

**Files:**
- Modify: `tests/modeling/test_property_invariants.py`
- Modify: `tests/modeling/test_real_data_sanity.py`
- Modify: `tests/modeling/_fixtures/real_schedules_2022_2024.parquet` (regenerated, binary file — see Step 1)

**Interfaces:**
- Consumes: `Backtest`, `FlatSizing` (Task 2/3, `backtest.py`). `SingleSplit`, `iterate_folds` (existing, `splits.py`).

These two files currently fail (Task 2 deleted `run_backtest`, which both import) — this task fixes that.

- [ ] **Step 1: Regenerate the real-data fixture with odds columns**

`real_schedules_2022_2024.parquet` was generated before `Backtest` needed `home_spread_odds`/`away_spread_odds` — regenerate it with those columns included. Run this from the repo root:

```bash
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
```

Verify it worked: `uv run python -c "import polars as pl; s = pl.read_parquet('tests/modeling/_fixtures/real_schedules_2022_2024.parquet'); print(s.columns); print(s.select('home_spread_odds', 'away_spread_odds').null_count())"` — expect `home_spread_odds`/`away_spread_odds` in the column list and `0` nulls for both (verified against this exact 2022-2024 range during this plan's own verification pass).

- [ ] **Step 2: Update `test_real_data_sanity.py`'s regeneration docstring and test**

In `tests/modeling/test_real_data_sanity.py`, update the module docstring's `schedules_cols` list (in the "To regenerate" code block) to match Step 1 exactly:

```python
    schedules_cols = [
        "game_id", "season", "week", "home_team", "away_team",
        "result", "spread_line", "home_spread_odds", "away_spread_odds",
    ]
```

Replace the imports:

```python
from degenebet.data.access import _widen_with_team_data
from degenebet.modeling.backtest import Backtest, FlatSizing
from degenebet.modeling.features import compute_rolling_features
from degenebet.modeling.splits import SingleSplit, iterate_folds
from degenebet.modeling.spread_model import SpreadModel
```

Replace the test function's body from `result = run_backtest(...)` onward:

```python
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
    # _widen_with_team_data always left-joins (a game where one team lacks
    # enough rolling history gets null feature columns, not a dropped row)
    # -- drop those explicitly, same games build_model_table's old inner
    # join used to drop, but visibly here rather than inside a shared join.
    model_table = _widen_with_team_data(schedules, rolling_features).drop_nulls(
        subset=[
            "home_offense_epa",
            "home_defense_epa_allowed",
            "home_turnover_margin",
            "away_offense_epa",
            "away_defense_epa_allowed",
            "away_turnover_margin",
        ]
    )

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
```

- [ ] **Step 3: Run the real-data test to verify it passes**

Run: `uv run pytest tests/modeling/test_real_data_sanity.py -v`
Expected: PASS. Verified during this plan's own verification pass: `bets_placed=224`, `ats_win_rate≈0.518` — comfortably inside the `0.35`-`0.65` band.

- [ ] **Step 4: Update `test_property_invariants.py`'s reconciliation property test**

Replace the file's imports:

```python
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
```

Update the module docstring's reference from `` `run_backtest` `` to `` `Backtest.run` ``:

```python
"""Property-based tests for invariants that would be expensive to get
silently wrong: cover probability bounds, backtest P&L reconciliation
under re-aggregation, and Efficacy's directional_accuracy/r_squared bounds
(per AGENTS.md's Automation & Verification mandate).

All three tests below exercise the real production code paths
(`SpreadModel.cover_probability`, `Backtest.run`, and `Efficacy.evaluate`)
rather than re-testing the scipy/polars primitives they're built on.
"""
```

Replace `_game_row`'s definition and the test itself:

```python
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
```

Everything from `_prediction_pair = st.tuples(...)` onward (the `Efficacy` property test) stays exactly as it is — do not modify it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_property_invariants.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Run the full modeling suite to confirm nothing else is broken**

Run: `uv run pytest tests/modeling/ -v`
Expected: PASS (all tests — this is the first point since Task 2 where the entire suite is green again)

- [ ] **Step 7: Lint, format, type-check**

Run: `uv run ruff check tests/modeling/test_property_invariants.py tests/modeling/test_real_data_sanity.py && uv run ruff format tests/modeling/test_property_invariants.py tests/modeling/test_real_data_sanity.py && uv run mypy`
Expected: all clean

- [ ] **Step 8: Commit**

```bash
git add tests/modeling/test_property_invariants.py tests/modeling/test_real_data_sanity.py tests/modeling/_fixtures/real_schedules_2022_2024.parquet
git commit -m "test: migrate property invariants and real-data sanity check to Backtest"
```

---

### Task 6: `notebooks/03_backtest.py`

**Files:**
- Create: `notebooks/03_backtest.py`

**Interfaces:**
- Consumes: `DataAccess`, `NflverseSource`, `SharpApiSource`, `Gameweek` (existing, `src/degenebet/data/`), `compute_rolling_features` (existing, `src/degenebet/modeling/features.py`), `SpreadModel` (existing), `WalkForwardSplit`/`iterate_folds` (Task 1, `splits.py`), `Backtest`/`FlatSizing`/`compute_bankroll_trajectory` (Tasks 2-4, `backtest.py`).

This is a runnable demonstration of the walk-forward, real-odds, compounding-bankroll workflow this plan's spec exists to support. No new production code; this task only creates the notebook file.

- [ ] **Step 1: Write the notebook**

Create `notebooks/03_backtest.py`:

```python
import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import date

    import altair as alt
    import marimo as mo
    import polars as pl

    from degenebet.data.access import DataAccess, NflverseSource, SharpApiSource
    from degenebet.data.gameweek import Gameweek
    from degenebet.modeling.backtest import Backtest, FlatSizing, compute_bankroll_trajectory
    from degenebet.modeling.features import compute_rolling_features
    from degenebet.modeling.splits import WalkForwardSplit, iterate_folds
    from degenebet.modeling.spread_model import SpreadModel

    return (
        Backtest,
        DataAccess,
        FlatSizing,
        Gameweek,
        NflverseSource,
        SharpApiSource,
        SpreadModel,
        WalkForwardSplit,
        alt,
        compute_bankroll_trajectory,
        compute_rolling_features,
        date,
        iterate_folds,
        mo,
        pl,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Backtest: real odds, walk-forward, compounding bankroll

    Runs `Backtest` + `WalkForwardSplit` against a `SpreadModel` candidate:
    one fold per gameweek after a warmup period, each scored at that
    game's own real American odds (`home_spread_odds`/`away_spread_odds`
    -- not a hardcoded -110). `FlatSizing` divides a normalized pool of
    `1.0` evenly across each week's bets; the top-level result is a
    pooled, **equal-weighted average across folds** (each gameweek counts
    the same, whether it had 2 games or 14) -- a different, still useful,
    question from the compounding dollar trajectory below (`what's my
    actual bankroll if I reinvest it every week`).

    Reads from the local cache only, via `DataAccess`; run
    `degenebet fetch schedules`/`degenebet fetch team-stats` (or
    `degenebet sync`) first.
    """)
    return


@app.cell
def _():
    SEASONS = [2022, 2023, 2024]
    return (SEASONS,)


@app.cell
def _(
    DataAccess,
    Gameweek,
    NflverseSource,
    SEASONS,
    SharpApiSource,
    compute_rolling_features,
    date,
    pl,
):
    access = DataAccess(NflverseSource(), SharpApiSource())
    start_week = Gameweek(min(SEASONS), 1)
    end_week = Gameweek(max(SEASONS), 22)  # NFL seasons run through the Super Bowl
    as_of_date = date.today()

    team_data = access.get_team_data(start_week, end_week, as_of_date=as_of_date)
    rolling = compute_rolling_features(team_data)
    model_table = (
        access.get_game_data(start_week, end_week, as_of_date=as_of_date, team_data=rolling)
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
        # Explicit opt-in, per Backtest's null-handling contract: only
        # played, priced games reach Backtest -- it never filters these
        # itself.
        .filter(
            pl.col("result").is_not_null()
            & pl.col("spread_line").is_not_null()
            & pl.col("home_spread_odds").is_not_null()
            & pl.col("away_spread_odds").is_not_null()
        )
    )
    model_table.head()
    return (model_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Walk-forward backtest
    """)
    return


@app.cell
def _(Backtest, FlatSizing, SpreadModel, WalkForwardSplit, iterate_folds, model_table):
    split_strategy = WalkForwardSplit(warmup_seasons=1)
    folds = iterate_folds(model_table, split_strategy, SpreadModel)
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)
    result = backtest.run_folds(folds)

    print(
        f"bets_placed={result.bets_placed} "
        f"ats_win_rate={result.ats_win_rate:.3f} "
        f"units_won={result.units_won:.3f} "
        f"roi_pct={result.roi_pct:.1f}%"
    )
    result.by_fold
    return (result,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Bankroll trajectory

    Compounding: each fold's stake is sized off the *actual* bankroll
    remaining after every prior fold, not a fixed amount. This is
    full-reinvestment sizing -- the whole current bankroll is staked
    every week, none held in reserve -- so a single week where every bet
    loses (`units_won == -1.0` exactly) wipes the bankroll to `$0`, which
    then correctly stays `$0` regardless of how well later weeks would
    have gone. That's not a bug; it's the real risk of this sizing rule,
    and it does happen in this real 2022-2024 walk-forward run below.
    """)
    return


@app.cell
def _(compute_bankroll_trajectory, result):
    trajectory = compute_bankroll_trajectory(result.by_fold, starting_bankroll=1000.0)

    print(
        f"starting_bankroll={trajectory.starting_bankroll:.2f} "
        f"ending_bankroll={trajectory.ending_bankroll:.2f} "
        f"total_pnl={trajectory.total_pnl:.2f}"
    )
    trajectory.by_fold
    return (trajectory,)


@app.cell
def _(alt, mo, trajectory):
    _chart = (
        alt.Chart(trajectory.by_fold)
        .mark_line(point=True)
        .encode(
            x=alt.X("fold:Q", title="Fold (gameweek, in order)"),
            y=alt.Y("bankroll_after:Q", title="Bankroll ($)"),
            tooltip=["fold", "bankroll_before", "pnl", "bankroll_after"],
        )
        .properties(title="Compounding bankroll trajectory", width=500, height=300)
    )
    mo.ui.altair_chart(_chart)
    return


if __name__ == "__main__":
    app.run()
```

- [ ] **Step 2: Validate the notebook's structure**

Run: `uv run marimo check notebooks/03_backtest.py`
Expected: no output (clean)

- [ ] **Step 3: Execute the notebook headlessly against real local cache data**

Run: `uv run marimo export html notebooks/03_backtest.py -o /tmp/notebook_check.html`
Expected: no error output; the `print()` calls produce something like `bets_placed=438 ats_win_rate=0.481 units_won=-2.624 roi_pct=-6.1%` and `starting_bankroll=1000.00 ending_bankroll=0.00 total_pnl=-1000.00` (verified during this plan's own verification pass — `ending_bankroll=0.00` is expected, not a bug: a real fold late in the walk had every bet lose, wiping the bankroll under full-reinvestment sizing, exactly as the notebook's own markdown explains). Then run `grep -c "marimo-cell-output-error" /tmp/notebook_check.html` — expect `0`.

If this fails because the local cache is empty or predates the DataAccess redesign's `gameweek` column, run `uv run degenebet fetch schedules && uv run degenebet fetch team-stats` first (deleting `~/.degenebet/cache/merged/{schedules,team_stats}.parquet` first if `load_or_merge`'s schema guard raises — confirm with the user before deleting anything there).

- [ ] **Step 4: Lint and format**

Run: `uv run ruff check notebooks/03_backtest.py && uv run ruff format notebooks/03_backtest.py`
Expected: clean (re-run Step 3 after formatting to confirm the formatted file still executes cleanly, since `ruff format` can change line breaks)

- [ ] **Step 5: Commit**

```bash
git add notebooks/03_backtest.py
git commit -m "feat: add walk-forward backtest notebook with compounding bankroll trajectory"
```

---

## Final Verification

After all 6 tasks are complete:

- [ ] Run `just check` from the repo root — expect ruff check, ruff format --check, mypy, and pytest (all tests, including the ones added/updated in this plan) to all pass.
- [ ] Confirm `git log --oneline` shows 6 commits since branching from `master`, each with a conventional-commit message.
- [ ] Confirm `run_backtest` and the old `BacktestResult` shape are genuinely gone: `grep -rn "run_backtest" src/ tests/ notebooks/` returns nothing.
- [ ] Confirm `src/degenebet/modeling/features.py` was not touched by this plan (`git diff master --stat -- src/degenebet/modeling/features.py` should be empty) — nothing in this plan's scope needs it.
