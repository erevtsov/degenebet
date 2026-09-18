# Modeling Efficacy Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared modeling foundation (`SplitStrategy`/`SingleSplit`, `iterate_folds`, a `Model` protocol) plus `Efficacy` (a pure, market-unaware prediction-quality scorer), and a notebook demonstrating the resulting model-selection workflow against real data.

**Architecture:** Two new modules, `src/degenebet/modeling/splits.py` (the split/fold orchestration primitives) and `src/degenebet/modeling/efficacy.py` (the scorer that consumes them), plus one small public-property addition to the existing `SpreadModel`. `backtest.py` and `features.py` are untouched.

**Tech Stack:** Python 3.12, polars, scikit-learn (`sklearn.metrics.r2_score`, `sklearn.linear_model.Ridge` in the notebook only), scipy (`scipy.stats.pearsonr`), `hypothesis` for one property test, `pytest`, `mypy --strict`, `ruff`, marimo/altair for the notebook.

**Spec:** `docs/superpowers/specs/2026-09-18-modeling-efficacy-foundation-design.md`

## Global Constraints

- Use polars for all tabular data; never pandas.
- `from collections.abc import Callable, Iterable, Iterator` — not `typing` — ruff's UP035 rule rejects `typing.Callable`/`Iterable`/`Iterator` when `from __future__ import annotations` is present (verified against this repo's ruff config: `select = ["E", "F", "I", "UP"]`). `Protocol` and `NamedTuple` still come from `typing` (no `collections.abc` equivalent), matching `src/degenebet/data/access.py` and `gameweek.py`.
- `mypy --strict` only checks `src/degenebet` (`[tool.mypy] packages = ["degenebet"]`) — test files are not type-checked, but still must pass `ruff check`/`ruff format`.
- `Efficacy` never reads `spread_line` — `predicted_result` vs. `result` only. `spread_line` is exclusively a `Backtest` concern (out of scope for this plan).
- `SplitStrategy`, `SingleSplit`, `Model`, `Split`, `FoldPredictions`, `iterate_folds` all live in `src/degenebet/modeling/splits.py`. `EfficacyResult`, `Efficacy` live in `src/degenebet/modeling/efficacy.py`.
- `src/degenebet/modeling/backtest.py` and `src/degenebet/modeling/features.py` are **not modified** by this plan.
- No CLI wiring — this plan's deliverable is library code, tests, and a notebook.
- Test-first for every task: write the failing test, confirm it fails, then implement.
- Never mock what can be faked with real in-memory Polars DataFrames — every test in this plan uses small, hand-constructed real data, never a mock.
- `notebooks/02_model_efficacy.py` is not part of `just check` (notebooks aren't CI-gated), but must pass `uv run marimo check notebooks/02_model_efficacy.py` and a headless `uv run marimo export html notebooks/02_model_efficacy.py -o <tmp>.html` with no error output before committing, per `CLAUDE.md`'s Notebooks convention. It must also pass `uv run ruff check`/`ruff format` like any other tracked `.py` file.
- Reads for the notebook go through `DataAccess` only, against the local cache — never live `nflreadpy`/SharpAPI calls.

---

### Task 1: `Split`, `SplitStrategy`, `SingleSplit`

**Files:**
- Create: `src/degenebet/modeling/splits.py`
- Create: `tests/modeling/test_splits.py`

**Interfaces:**
- Produces: `Split(NamedTuple)` with fields `train: pl.DataFrame`, `test: pl.DataFrame`. `SplitStrategy(Protocol)` with `splits(self, data: pl.DataFrame) -> Iterator[Split]`. `SingleSplit` with `__init__(self, train_seasons: list[int], test_seasons: list[int]) -> None` and `splits(self, data: pl.DataFrame) -> Iterator[Split]`.

`SingleSplit.splits()` mirrors `run_backtest`'s existing train/test season-filtering and leakage guards (`src/degenebet/modeling/backtest.py:45-56`) — extracted here rather than reimplemented from scratch; `run_backtest` itself is not modified.

- [ ] **Step 1: Write the failing tests**

Create `tests/modeling/test_splits.py`:

```python
from __future__ import annotations

import polars as pl
import pytest

from degenebet.modeling.splits import SingleSplit


def test_singlesplit_partitions_by_season() -> None:
    data = pl.DataFrame({"season": [2020, 2020, 2021, 2022], "value": [1, 2, 3, 4]})
    strategy = SingleSplit(train_seasons=[2020], test_seasons=[2021])

    splits = list(strategy.splits(data))

    assert len(splits) == 1
    assert splits[0].train["value"].to_list() == [1, 2]
    assert splits[0].test["value"].to_list() == [3]


def test_singlesplit_raises_on_overlapping_seasons() -> None:
    data = pl.DataFrame({"season": [2020, 2021], "value": [1, 2]})
    strategy = SingleSplit(train_seasons=[2020, 2021], test_seasons=[2021])

    with pytest.raises(ValueError, match="overlap"):
        list(strategy.splits(data))


def test_singlesplit_raises_on_empty_train() -> None:
    data = pl.DataFrame({"season": [2020], "value": [1]})
    strategy = SingleSplit(train_seasons=[2099], test_seasons=[2020])

    with pytest.raises(ValueError, match="No rows"):
        list(strategy.splits(data))


def test_singlesplit_yields_zero_row_test_frame_when_test_seasons_have_no_data() -> None:
    data = pl.DataFrame({"season": [2020], "value": [1]})
    strategy = SingleSplit(train_seasons=[2020], test_seasons=[2099])

    splits = list(strategy.splits(data))

    assert len(splits) == 1
    assert splits[0].test.height == 0
```

`SingleSplit.splits()` is a generator (uses `yield`) — Python doesn't execute a generator's body until it's iterated, so every `pytest.raises` block above wraps `list(strategy.splits(data))`, not the bare `strategy.splits(data)` call, or the exception would never actually be raised inside the `with` block.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/modeling/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'degenebet.modeling.splits'`

- [ ] **Step 3: Write the implementation**

Create `src/degenebet/modeling/splits.py`:

```python
"""Split/fold orchestration: SplitStrategy implementations partition data
into (train, test) pairs; iterate_folds (Task 2) is the one shared
fit/predict loop Efficacy and, later, Backtest both consume. See
docs/superpowers/specs/2026-09-18-modeling-efficacy-foundation-design.md.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple, Protocol

import polars as pl


class Split(NamedTuple):
    """One (train, test) partition of a model table."""

    train: pl.DataFrame
    test: pl.DataFrame


class SplitStrategy(Protocol):
    def splits(self, data: pl.DataFrame) -> Iterator[Split]: ...


class SingleSplit:
    """One (train, test) pair by season list. Mirrors run_backtest's
    existing train_seasons/test_seasons filtering and leakage guards,
    extracted here rather than reimplemented."""

    def __init__(self, train_seasons: list[int], test_seasons: list[int]) -> None:
        self.train_seasons = train_seasons
        self.test_seasons = test_seasons

    def splits(self, data: pl.DataFrame) -> Iterator[Split]:
        """Raises ValueError if train_seasons and test_seasons overlap
        (leakage), or if the train filter is empty (nothing to fit on).
        An empty test filter is not an error (e.g. an in-progress season
        with no played games yet) -- yields a zero-row test frame."""
        overlap = set(self.train_seasons) & set(self.test_seasons)
        if overlap:
            raise ValueError(
                f"train_seasons and test_seasons overlap: {sorted(overlap)} — "
                "this would leak test-season data into training and inflate results."
            )

        train = data.filter(pl.col("season").is_in(self.train_seasons))
        test = data.filter(pl.col("season").is_in(self.test_seasons))

        if train.height == 0:
            raise ValueError(f"No rows in data for train_seasons={self.train_seasons}")

        yield Split(train=train, test=test)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_splits.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/degenebet/modeling/splits.py tests/modeling/test_splits.py && uv run ruff format src/degenebet/modeling/splits.py tests/modeling/test_splits.py && uv run mypy`
Expected: all clean

- [ ] **Step 6: Commit**

```bash
git add src/degenebet/modeling/splits.py tests/modeling/test_splits.py
git commit -m "feat: add Split/SplitStrategy/SingleSplit"
```

---

### Task 2: `Model` protocol, `FoldPredictions`, `iterate_folds`

**Files:**
- Modify: `src/degenebet/modeling/splits.py`
- Modify: `tests/modeling/test_splits.py`

**Interfaces:**
- Consumes: `Split`, `SplitStrategy` from Task 1 (same file).
- Produces: `Model(Protocol)` with `fit(self, train_data: pl.DataFrame) -> None` and `predict(self, data: pl.DataFrame) -> pl.DataFrame`. `FoldPredictions(NamedTuple)` with fields `model: Model`, `in_sample: pl.DataFrame`, `out_of_sample: pl.DataFrame`. `iterate_folds(data: pl.DataFrame, split_strategy: SplitStrategy, model_factory: Callable[[], Model]) -> Iterator[FoldPredictions]`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/modeling/test_splits.py`'s top-of-file imports with:

```python
from __future__ import annotations

from collections.abc import Iterator

import polars as pl
import pytest

from degenebet.modeling.splits import FoldPredictions, SingleSplit, Split, iterate_folds
```

Then append, after the existing Task 1 tests:

```python
class _TwoFoldSplitStrategy:
    """Test double yielding two fixed (train, test) splits from a `fold`
    column -- standing in for a real multi-fold strategy like the future
    WalkForwardSplit, which doesn't exist yet. Exists only to prove
    iterate_folds creates a fresh model per fold rather than reusing one
    across folds; SingleSplit itself always yields exactly one fold, so it
    can't exercise that behavior."""

    def splits(self, data: pl.DataFrame) -> Iterator[Split]:
        yield Split(train=data.filter(pl.col("fold") == 0), test=data.filter(pl.col("fold") == 1))
        yield Split(train=data.filter(pl.col("fold") == 1), test=data.filter(pl.col("fold") == 0))


class _RecordingModel:
    """Fake Model recording fit() calls and tagging predict() output with
    this instance's identity, so a test can assert iterate_folds passes
    the right frames and creates a fresh instance per fold."""

    def __init__(self) -> None:
        self.fit_calls: list[pl.DataFrame] = []

    def fit(self, train_data: pl.DataFrame) -> None:
        self.fit_calls.append(train_data)

    def predict(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns(pl.lit(id(self)).alias("model_id"))


def test_iterate_folds_uses_a_fresh_model_per_fold() -> None:
    data = pl.DataFrame({"fold": [0, 1], "x": [1, 2]})

    results = list(iterate_folds(data, _TwoFoldSplitStrategy(), _RecordingModel))

    assert len(results) == 2
    assert results[0].model is not results[1].model
    assert isinstance(results[0].model, _RecordingModel)
    assert results[0].model.fit_calls[0].equals(data.filter(pl.col("fold") == 0))
    assert results[1].model.fit_calls[0].equals(data.filter(pl.col("fold") == 1))
    assert results[0].in_sample["model_id"].to_list() == [id(results[0].model)]
    assert results[0].out_of_sample["model_id"].to_list() == [id(results[0].model)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/modeling/test_splits.py -v`
Expected: FAIL with `ImportError: cannot import name 'FoldPredictions' from 'degenebet.modeling.splits'`

- [ ] **Step 3: Write the implementation**

Append to `src/degenebet/modeling/splits.py` (also change `from collections.abc import Iterator` to `from collections.abc import Callable, Iterator` at the top):

```python
class Model(Protocol):
    def fit(self, train_data: pl.DataFrame) -> None: ...
    def predict(self, data: pl.DataFrame) -> pl.DataFrame: ...


class FoldPredictions(NamedTuple):
    """One fold's fit model plus its predictions on both sides of the
    split. `in_sample`/`out_of_sample` are the precomputed,
    single-source-of-truth prediction frames -- Efficacy (and, later,
    Backtest) score these directly rather than re-deriving predictions
    themselves. `model` is for whatever predictions alone can't answer
    (residuals, learned weights, any other diagnostic); it costs nothing
    extra to expose since it's already been created and fit either way."""

    model: Model
    in_sample: pl.DataFrame
    out_of_sample: pl.DataFrame


def iterate_folds(
    data: pl.DataFrame,
    split_strategy: SplitStrategy,
    model_factory: Callable[[], Model],
) -> Iterator[FoldPredictions]:
    """The one shared fit/predict orchestration -- neither Efficacy nor
    (later) Backtest owns fitting or fold-looping itself. model_factory
    (not a fixed Model instance) matters because a stateful Model like
    SpreadModel needs a fresh instance per fold so fitted state never
    leaks across folds."""
    for split in split_strategy.splits(data):
        model = model_factory()
        model.fit(split.train)
        yield FoldPredictions(
            model=model,
            in_sample=model.predict(split.train),
            out_of_sample=model.predict(split.test),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_splits.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/degenebet/modeling/splits.py tests/modeling/test_splits.py && uv run ruff format src/degenebet/modeling/splits.py tests/modeling/test_splits.py && uv run mypy`
Expected: all clean

- [ ] **Step 6: Commit**

```bash
git add src/degenebet/modeling/splits.py tests/modeling/test_splits.py
git commit -m "feat: add Model protocol, FoldPredictions, iterate_folds"
```

---

### Task 3: `SpreadModel.residual_std` public property

**Files:**
- Modify: `src/degenebet/modeling/spread_model.py:59-61` (the `__init__` block that sets `self._residual_std`)
- Modify: `tests/modeling/test_spread_model.py`

**Interfaces:**
- Produces: `SpreadModel.residual_std` (read-only `float | None` property) — `None` before `fit()`, the fitted residual standard deviation after. This exists so `FoldPredictions.model.residual_std` (Task 2) is a legitimate public access path rather than reaching into the private `_residual_std`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/modeling/test_spread_model.py` (it already defines `_noisy_table_with_spread()` — reuse it):

```python
def test_residual_std_is_none_before_fit() -> None:
    model = SpreadModel()

    assert model.residual_std is None


def test_residual_std_matches_the_value_fit_computed() -> None:
    model = SpreadModel()
    table = _noisy_table_with_spread()

    model.fit(table)

    assert model.residual_std is not None
    assert model.residual_std > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/modeling/test_spread_model.py -k residual_std -v`
Expected: FAIL with `AttributeError: 'SpreadModel' object has no attribute 'residual_std'`

- [ ] **Step 3: Write the implementation**

In `src/degenebet/modeling/spread_model.py`, add a property immediately after `__init__` (before `fit`):

```python
    def __init__(self, model: RegressorProtocol | None = None) -> None:
        self.model: RegressorProtocol = model if model is not None else LinearRegression()
        self._residual_std: float | None = None

    @property
    def residual_std(self) -> float | None:
        """The training residual spread computed by fit(), or None before
        fit() is called."""
        return self._residual_std

    def fit(self, model_table: pl.DataFrame) -> None:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_spread_model.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/degenebet/modeling/spread_model.py tests/modeling/test_spread_model.py && uv run ruff format src/degenebet/modeling/spread_model.py tests/modeling/test_spread_model.py && uv run mypy`
Expected: all clean

- [ ] **Step 6: Commit**

```bash
git add src/degenebet/modeling/spread_model.py tests/modeling/test_spread_model.py
git commit -m "feat: expose SpreadModel.residual_std as a public property"
```

---

### Task 4: `EfficacyResult`, `Efficacy.evaluate()`

**Files:**
- Create: `src/degenebet/modeling/efficacy.py`
- Create: `tests/modeling/test_efficacy.py`

**Interfaces:**
- Produces: `EfficacyResult(dataclass, frozen=True)` with fields `metrics: dict[str, float]`, `by_fold: pl.DataFrame | None`. `Efficacy.evaluate(self, predictions: pl.DataFrame) -> EfficacyResult`. A private `_compute_metrics(predictions: pl.DataFrame) -> dict[str, float]` helper — Task 5 reuses this for `evaluate_folds`.

`predictions` must have `predicted_result` and `result` columns (the shape `Model.predict()`, Task 2, produces). `metrics` keys: `rmse`, `r_squared`, `directional_accuracy`, `information_coefficient`.

- [ ] **Step 1: Write the failing tests**

Create `tests/modeling/test_efficacy.py`:

```python
from __future__ import annotations

import polars as pl
import pytest

from degenebet.modeling.efficacy import Efficacy


def test_evaluate_computes_rmse_r_squared_directional_accuracy_and_ic() -> None:
    predictions = pl.DataFrame(
        {"predicted_result": [3.0, -2.0, 5.0, 0.0], "result": [4.0, -1.0, 5.0, 1.0]}
    )

    result = Efficacy().evaluate(predictions)

    assert result.metrics["rmse"] == pytest.approx(0.8660254037844386)
    assert result.metrics["r_squared"] == pytest.approx(0.8681318681318682)
    assert result.metrics["directional_accuracy"] == pytest.approx(0.75)
    assert result.metrics["information_coefficient"] == pytest.approx(0.9927741970308563)
    assert result.by_fold is None


def test_evaluate_raises_on_null_result() -> None:
    predictions = pl.DataFrame(
        {"predicted_result": [1.0, 2.0], "result": [1.0, None]},
        schema={"predicted_result": pl.Float64, "result": pl.Float64},
    )

    with pytest.raises(ValueError, match="null values"):
        Efficacy().evaluate(predictions)


def test_evaluate_raises_on_empty_frame() -> None:
    predictions = pl.DataFrame(
        {"predicted_result": [], "result": []},
        schema={"predicted_result": pl.Float64, "result": pl.Float64},
    )

    with pytest.raises(ValueError, match="empty"):
        Efficacy().evaluate(predictions)
```

The expected values in the first test were computed directly with `numpy`/`scipy.stats.pearsonr`/`sklearn.metrics.r2_score` against the same `predicted_result`/`result` arrays — not hand-derived — so the implementation in Step 3 must use those exact functions (not a different formula that happens to be mathematically equivalent in the general case but diverges in floating-point rounding).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/modeling/test_efficacy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'degenebet.modeling.efficacy'`

- [ ] **Step 3: Write the implementation**

Create `src/degenebet/modeling/efficacy.py`:

```python
"""Efficacy: pure prediction-quality scoring -- predicted_result vs.
result only, never spread_line (that's exclusively a Backtest concern).
See docs/superpowers/specs/2026-09-18-modeling-efficacy-foundation-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


@dataclass(frozen=True)
class EfficacyResult:
    metrics: dict[str, float]
    by_fold: pl.DataFrame | None


def _compute_metrics(predictions: pl.DataFrame) -> dict[str, float]:
    """predicted_result vs. result only -- spread_line never enters this
    computation. Raises ValueError on an empty frame or a null result
    (loud, not silent NaN metrics), same philosophy as SpreadModel's
    null-feature guard."""
    if predictions.height == 0:
        raise ValueError("Cannot evaluate an empty predictions frame.")
    if predictions["result"].null_count() > 0:
        raise ValueError(
            "predictions has null values in 'result' -- can't score against missing "
            "ground truth. Filter to played games before calling evaluate()."
        )

    predicted = predictions["predicted_result"].to_numpy()
    actual = predictions["result"].to_numpy()

    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    r_squared = float(r2_score(actual, predicted))
    # sign(0) == 0, so a tied game (result == 0, rare but legal in the NFL
    # regular season) only counts as a correct call if predicted_result is
    # exactly 0.0 too -- not worth special-casing for how rare it is.
    directional_accuracy = float(np.mean(np.sign(predicted) == np.sign(actual)))
    information_coefficient = float(pearsonr(predicted, actual)[0])

    return {
        "rmse": rmse,
        "r_squared": r_squared,
        "directional_accuracy": directional_accuracy,
        "information_coefficient": information_coefficient,
    }


class Efficacy:
    """Pure scorer -- no fitting, no fold-looping, no multi-configuration
    search. predicted_result vs. result only; spread_line never enters
    this class (that's exclusively a Backtest concern)."""

    def evaluate(self, predictions: pl.DataFrame) -> EfficacyResult:
        """Scores one frame. by_fold is always None here -- a single frame
        has no fold concept, so there's nothing to build a breakdown
        from."""
        return EfficacyResult(metrics=_compute_metrics(predictions), by_fold=None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_efficacy.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/degenebet/modeling/efficacy.py tests/modeling/test_efficacy.py && uv run ruff format src/degenebet/modeling/efficacy.py tests/modeling/test_efficacy.py && uv run mypy`
Expected: all clean

- [ ] **Step 6: Commit**

```bash
git add src/degenebet/modeling/efficacy.py tests/modeling/test_efficacy.py
git commit -m "feat: add EfficacyResult and Efficacy.evaluate"
```

---

### Task 5: `Efficacy.evaluate_folds()`

**Files:**
- Modify: `src/degenebet/modeling/efficacy.py`
- Modify: `tests/modeling/test_efficacy.py`

**Interfaces:**
- Consumes: `FoldPredictions` from Task 2 (`src/degenebet/modeling/splits.py`), `_compute_metrics`/`EfficacyResult` from Task 4 (same file).
- Produces: `Efficacy.evaluate_folds(self, folds: Iterable[FoldPredictions]) -> EfficacyResult`.

**by_fold shape:** long format — one row per `(fold, sample)` pair, columns `fold: int`, `sample: str` (`"in_sample"` or `"out_of_sample"`), then `rmse`, `r_squared`, `directional_accuracy`, `information_coefficient`. **metrics:** computed by concatenating every fold's `out_of_sample` frame and calling `_compute_metrics` once on the pooled result — never by averaging each fold's already-computed metrics.

- [ ] **Step 1: Write the failing tests**

Append to `tests/modeling/test_efficacy.py`. Change the top imports to:

```python
from __future__ import annotations

import polars as pl
import pytest

from degenebet.modeling.efficacy import Efficacy
from degenebet.modeling.splits import FoldPredictions, SingleSplit, iterate_folds
from degenebet.modeling.spread_model import SpreadModel
```

Then append:

```python
def _fold(
    *,
    in_sample_predicted: list[float],
    in_sample_result: list[float],
    out_of_sample_predicted: list[float],
    out_of_sample_result: list[float],
) -> FoldPredictions:
    # SpreadModel() here is an unfitted placeholder -- evaluate_folds never
    # reads FoldPredictions.model, only .in_sample/.out_of_sample, so its
    # exact identity doesn't matter for these tests.
    return FoldPredictions(
        model=SpreadModel(),
        in_sample=pl.DataFrame(
            {"predicted_result": in_sample_predicted, "result": in_sample_result}
        ),
        out_of_sample=pl.DataFrame(
            {"predicted_result": out_of_sample_predicted, "result": out_of_sample_result}
        ),
    )


def _two_fixed_folds() -> list[FoldPredictions]:
    # Deliberately different-sized out_of_sample frames (2 rows, 4 rows) --
    # a fold that mis-weights by averaging per-fold metrics instead of
    # pooling would produce a different (and wrong) rmse/r_squared/ic than
    # concatenating both frames and scoring once.
    fold_1 = _fold(
        in_sample_predicted=[5.0, 6.0, 7.0],
        in_sample_result=[5.0, 6.5, 6.0],
        out_of_sample_predicted=[1.0, 2.0],
        out_of_sample_result=[1.0, 3.0],
    )
    fold_2 = _fold(
        in_sample_predicted=[50.0, 60.0],
        in_sample_result=[55.0, 58.0],
        out_of_sample_predicted=[10.0, 20.0, 30.0, 40.0],
        out_of_sample_result=[12.0, 18.0, 33.0, 41.0],
    )
    return [fold_1, fold_2]


def test_evaluate_folds_pools_out_of_sample_metrics_not_averages_per_fold() -> None:
    result = Efficacy().evaluate_folds(_two_fixed_folds())

    # Pooled (concatenate both out_of_sample frames, score once). NOT the
    # naive average of each fold's own out_of_sample rmse (0.7071067811865476
    # for fold 1, 2.1213203435596424 for fold 2 -> average 1.414213562373095),
    # which would mis-weight the 2-row fold equally against the 4-row fold.
    assert result.metrics["rmse"] == pytest.approx(1.7795130420052185)
    assert result.metrics["r_squared"] == pytest.approx(0.9854294478527608)
    assert result.metrics["directional_accuracy"] == pytest.approx(1.0)
    assert result.metrics["information_coefficient"] == pytest.approx(0.9945095645744224)


def test_evaluate_folds_builds_long_format_by_fold_table() -> None:
    result = Efficacy().evaluate_folds(_two_fixed_folds())

    assert result.by_fold is not None
    assert result.by_fold.height == 4  # 2 folds x (in_sample, out_of_sample)
    assert sorted(result.by_fold["fold"].to_list()) == [0, 0, 1, 1]
    assert sorted(result.by_fold["sample"].unique().to_list()) == [
        "in_sample",
        "out_of_sample",
    ]
    fold_0_out_of_sample = result.by_fold.filter(
        (pl.col("fold") == 0) & (pl.col("sample") == "out_of_sample")
    ).row(0, named=True)
    assert fold_0_out_of_sample["rmse"] == pytest.approx(0.7071067811865476)
    fold_1_in_sample = result.by_fold.filter(
        (pl.col("fold") == 1) & (pl.col("sample") == "in_sample")
    ).row(0, named=True)
    assert fold_1_in_sample["rmse"] == pytest.approx(3.8078865529319543)


def test_evaluate_folds_raises_on_empty_folds_stream() -> None:
    with pytest.raises(ValueError, match="empty"):
        Efficacy().evaluate_folds([])


def test_evaluate_folds_on_a_single_fold_matches_evaluate_on_its_out_of_sample_frame() -> None:
    model_table = pl.DataFrame(
        {
            "season": [2000, 2000, 2000, 2001, 2001],
            "week": [1, 2, 3, 1, 2],
            "home_offense_epa": [0.1, -0.2, 0.3, 0.2, -0.1],
            "home_defense_epa_allowed": [0.0, 0.1, -0.1, 0.0, 0.2],
            "home_turnover_margin": [1.0, -1.0, 0.0, 1.0, -1.0],
            "away_offense_epa": [-0.1, 0.2, -0.3, -0.2, 0.1],
            "away_defense_epa_allowed": [0.1, 0.0, 0.2, -0.1, 0.0],
            "away_turnover_margin": [-1.0, 1.0, 0.0, -1.0, 1.0],
            "result": [3.0, -5.0, 7.0, 2.0, -4.0],
        }
    )
    split_strategy = SingleSplit(train_seasons=[2000], test_seasons=[2001])

    folds = list(iterate_folds(model_table, split_strategy, SpreadModel))
    assert len(folds) == 1

    via_evaluate_folds = Efficacy().evaluate_folds(folds).metrics
    via_evaluate = Efficacy().evaluate(folds[0].out_of_sample).metrics

    assert via_evaluate_folds == via_evaluate
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/modeling/test_efficacy.py -v`
Expected: FAIL with `AttributeError: 'Efficacy' object has no attribute 'evaluate_folds'`

- [ ] **Step 3: Write the implementation**

Replace `src/degenebet/modeling/efficacy.py`'s top-of-file imports with:

```python
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from degenebet.modeling.splits import FoldPredictions
```

Then add this method to the `Efficacy` class, after `evaluate`:

```python
    def evaluate_folds(self, folds: Iterable[FoldPredictions]) -> EfficacyResult:
        """Consumes iterate_folds's FoldPredictions stream. by_fold is long
        format: one row per (fold, sample) pair. metrics is computed by
        concatenating every fold's out_of_sample frame and scoring once --
        never by averaging each fold's already-computed metrics, so an
        uneven fold size (e.g. from a future WalkForwardSplit) doesn't
        mis-weight the result."""
        materialized = list(folds)
        if not materialized:
            raise ValueError("Cannot evaluate an empty folds stream.")

        by_fold_rows: list[dict[str, float | int | str]] = []
        out_of_sample_frames: list[pl.DataFrame] = []
        for fold_index, fold in enumerate(materialized):
            by_fold_rows.append(
                {"fold": fold_index, "sample": "in_sample", **_compute_metrics(fold.in_sample)}
            )
            by_fold_rows.append(
                {
                    "fold": fold_index,
                    "sample": "out_of_sample",
                    **_compute_metrics(fold.out_of_sample),
                }
            )
            out_of_sample_frames.append(fold.out_of_sample)

        pooled = pl.concat(out_of_sample_frames, how="vertical")
        return EfficacyResult(metrics=_compute_metrics(pooled), by_fold=pl.DataFrame(by_fold_rows))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/modeling/test_efficacy.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check src/degenebet/modeling/efficacy.py tests/modeling/test_efficacy.py && uv run ruff format src/degenebet/modeling/efficacy.py tests/modeling/test_efficacy.py && uv run mypy`
Expected: all clean

- [ ] **Step 6: Commit**

```bash
git add src/degenebet/modeling/efficacy.py tests/modeling/test_efficacy.py
git commit -m "feat: add Efficacy.evaluate_folds"
```

---

### Task 6: Property test for `Efficacy`'s metric bounds

**Files:**
- Modify: `tests/modeling/test_property_invariants.py`

**Interfaces:**
- Consumes: `Efficacy` from Task 4/5 (`src/degenebet/modeling/efficacy.py`).

Per `CLAUDE.md`'s Automation & Verification rule, every analytic surface needs a `hypothesis` property test for invariants that would be expensive to get silently wrong. `Efficacy` is a new analytic surface: `directional_accuracy` must stay in `[0, 1]` and `r_squared` must never exceed `1.0` (mathematically: `R² = 1 - ss_res/ss_tot`, and `ss_res`/`ss_tot` are both non-negative, so the ratio can't be negative — `R²` can be arbitrarily negative for bad predictions but never above `1.0`).

- [ ] **Step 1: Write the failing test**

Replace `tests/modeling/test_property_invariants.py`'s top-of-file imports with:

```python
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, reject
from hypothesis import strategies as st

from degenebet.modeling.backtest import run_backtest
from degenebet.modeling.efficacy import Efficacy
from degenebet.modeling.spread_model import SpreadModel
```

Then append, after the existing tests:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/modeling/test_property_invariants.py -k efficacy -v`
Expected: FAIL with `ImportError: cannot import name 'Efficacy' from 'degenebet.modeling.efficacy'` if Task 4/5 weren't actually completed first — but since they were, this should instead run and PASS immediately (the property already holds; there's no new production code to write in this task). Run it anyway to confirm hypothesis actually exercises multiple examples and finds no counterexample (check the test summary reports the test ran, not skipped).

- [ ] **Step 3: No implementation step**

This task adds test coverage only — `Efficacy.evaluate()` (Task 4) already satisfies the property. If Step 2 fails for any reason other than the import error above, stop and investigate before proceeding; it means Task 4/5's implementation has a real bug this property caught.

- [ ] **Step 4: Run the full modeling test suite to confirm nothing regressed**

Run: `uv run pytest tests/modeling/ -v`
Expected: PASS (all tests, including the new property test)

- [ ] **Step 5: Lint, format, type-check**

Run: `uv run ruff check tests/modeling/test_property_invariants.py && uv run ruff format tests/modeling/test_property_invariants.py && uv run mypy`
Expected: all clean

- [ ] **Step 6: Commit**

```bash
git add tests/modeling/test_property_invariants.py
git commit -m "test: add property test for Efficacy's directional_accuracy/r_squared bounds"
```

---

### Task 7: `notebooks/02_model_efficacy.py`

**Files:**
- Create: `notebooks/02_model_efficacy.py`

**Interfaces:**
- Consumes: `DataAccess`, `NflverseSource`, `SharpApiSource`, `Gameweek` (existing, `src/degenebet/data/`), `compute_rolling_features` (existing, `src/degenebet/modeling/features.py`), `SpreadModel` (existing, with Task 3's `residual_std` — not used directly in this notebook, no action needed), `SingleSplit`/`iterate_folds` (Task 1/2), `Efficacy` (Task 4/5).

This is a runnable demonstration of `architecture-notes.md`'s "step 1: model/hyperparameter selection" workflow — loop over candidate `SpreadModel` configs, score each with `Efficacy`, compare by eye. No new production code; this task only creates the notebook file.

- [ ] **Step 1: Write the notebook**

Create `notebooks/02_model_efficacy.py`:

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
    from sklearn.linear_model import Ridge

    from degenebet.data.access import DataAccess, NflverseSource, SharpApiSource
    from degenebet.data.gameweek import Gameweek
    from degenebet.modeling.efficacy import Efficacy
    from degenebet.modeling.features import compute_rolling_features
    from degenebet.modeling.splits import SingleSplit, iterate_folds
    from degenebet.modeling.spread_model import SpreadModel

    return (
        DataAccess,
        Efficacy,
        Gameweek,
        NflverseSource,
        Ridge,
        SharpApiSource,
        SingleSplit,
        SpreadModel,
        alt,
        compute_rolling_features,
        date,
        iterate_folds,
        mo,
        pl,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Model selection: comparing candidates with Efficacy

    Runs a few candidate `SpreadModel` configs through `SingleSplit` +
    `iterate_folds` + `Efficacy.evaluate_folds`, comparing their
    out-of-sample prediction quality (`predicted_result` vs. `result`
    only -- `spread_line` never enters `Efficacy`; that's a `Backtest`
    concern). This is `architecture-notes.md`'s "step 1: model/
    hyperparameter selection" workflow: loop over candidates, score each,
    pick one by human judgment -- `Efficacy` never picks for you.

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
):
    access = DataAccess(NflverseSource(), SharpApiSource())
    start_week = Gameweek(min(SEASONS), 1)
    end_week = Gameweek(max(SEASONS), 22)  # NFL seasons run through the Super Bowl
    as_of_date = date.today()

    team_data = access.get_team_data(start_week, end_week, as_of_date=as_of_date)
    rolling = compute_rolling_features(team_data)
    model_table = access.get_game_data(
        start_week, end_week, as_of_date=as_of_date, team_data=rolling
    ).drop_nulls(
        subset=[
            "home_offense_epa",
            "home_defense_epa_allowed",
            "home_turnover_margin",
            "away_offense_epa",
            "away_defense_epa_allowed",
            "away_turnover_margin",
        ]
    )
    model_table.head()
    return (model_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Candidate models
    """)
    return


@app.cell
def _(Ridge, SpreadModel):
    candidates = {
        "linear": lambda: SpreadModel(),
        "ridge (alpha=1.0)": lambda: SpreadModel(model=Ridge(alpha=1.0)),
        "ridge (alpha=10.0)": lambda: SpreadModel(model=Ridge(alpha=10.0)),
    }
    return (candidates,)


@app.cell
def _(Efficacy, SingleSplit, candidates, iterate_folds, model_table, pl):
    split_strategy = SingleSplit(train_seasons=[2022, 2023], test_seasons=[2024])
    efficacy = Efficacy()

    rows = []
    for name, model_factory in candidates.items():
        folds = iterate_folds(model_table, split_strategy, model_factory)
        result = efficacy.evaluate_folds(folds)
        rows.append({"candidate": name, **result.metrics})

    comparison = pl.DataFrame(rows)
    comparison
    return (comparison,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Out-of-sample RMSE by candidate

    Lower is better -- RMSE is in points of margin.
    """)
    return


@app.cell
def _(alt, comparison, mo):
    _chart = (
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            x=alt.X("candidate:N", title="Candidate"),
            y=alt.Y("rmse:Q", title="Out-of-sample RMSE"),
            tooltip=["candidate", "rmse", "r_squared", "directional_accuracy"],
        )
        .properties(title="Candidate comparison: out-of-sample RMSE", width=400, height=300)
    )
    mo.ui.altair_chart(_chart)
    return


if __name__ == "__main__":
    app.run()
```

- [ ] **Step 2: Validate the notebook's structure**

Run: `uv run marimo check notebooks/02_model_efficacy.py`
Expected: no output (clean)

- [ ] **Step 3: Execute the notebook headlessly against real local cache data**

Run: `uv run marimo export html notebooks/02_model_efficacy.py -o /tmp/notebook_check.html`
Expected: no error output. Then run `grep -c "marimo-cell-output-error" /tmp/notebook_check.html` — expect `0`.

If this fails because the local cache is empty or predates the DataAccess redesign's `gameweek` column, run `uv run degenebet fetch schedules && uv run degenebet fetch team-stats` first (deleting `~/.degenebet/cache/merged/{schedules,team_stats}.parquet` first if `load_or_merge`'s schema guard raises — confirm with the user before deleting anything there, same as any other cache reset).

- [ ] **Step 4: Lint and format**

Run: `uv run ruff check notebooks/02_model_efficacy.py && uv run ruff format notebooks/02_model_efficacy.py`
Expected: clean (re-run Step 3 after formatting to confirm the formatted file still executes cleanly, since `ruff format` can change line breaks)

- [ ] **Step 5: Commit**

```bash
git add notebooks/02_model_efficacy.py
git commit -m "feat: add model-selection notebook exercising SingleSplit/iterate_folds/Efficacy"
```

---

## Final Verification

After all 7 tasks are complete:

- [ ] Run `just check` from the repo root — expect ruff check, ruff format --check, mypy, and pytest (all tests, including the ones added in this plan) to all pass.
- [ ] Confirm `git log --oneline` shows 6 commits (Tasks 1-6; Task 7's notebook commit makes 7) since branching from `master`, each with a conventional-commit message.
- [ ] Confirm no changes were made to `src/degenebet/modeling/backtest.py` or `src/degenebet/modeling/features.py` (`git diff master --stat -- src/degenebet/modeling/backtest.py src/degenebet/modeling/features.py` should be empty).
