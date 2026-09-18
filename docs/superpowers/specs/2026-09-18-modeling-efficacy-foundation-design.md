# Modeling: Efficacy Foundation — Design Spec

## Context

`docs/architecture-notes.md` is a scratchpad from an earlier brainstorming
session that sketched a full rework of the modeling layer: `SplitStrategy`,
`iterate_folds`, a `Model` protocol with `TrainingResult`, `Efficacy`
(prediction-quality scoring), and a `Backtest` rework (`SizingStrategy`,
edge-aware sizing). Its own "open questions" section named the data layer's
`NflverseSource`/`SharpApiSource` shape and `TrainingResult`'s exact fields
as unsettled.

The data layer is now settled: `DataAccess.get_game_data`/`get_team_data`
(the 2026-09-18 DataAccess redesign) replaced the date-range sketch, and
`build_model_table` — the hand-written join `architecture-notes.md` didn't
anticipate needing — was retired in favor of `DataAccess`'s own `team_data`
widening. That removes the last blocker to starting the modeling rework.

This spec covers the first slice: the **shared foundation** (`SplitStrategy`,
`iterate_folds`, a `Model` protocol) plus **`Efficacy`** — together, this
completes `architecture-notes.md`'s "step 1: model/hyperparameter selection"
workflow (loop over candidate model configs, score each with `Efficacy`,
pick one by human judgment). The `Backtest` rework (`SizingStrategy`,
edge-aware sizing, `run_folds` pooled stats) and `WalkForwardSplit` are
explicitly **out of scope** — a follow-on sub-project once this foundation
exists to build on.

## Goals

- A `SplitStrategy` abstraction with one implementation, `SingleSplit`
  (train/test by season list) — a mechanical extraction of validation logic
  `run_backtest` already has today.
- `iterate_folds`, the one shared fit/predict orchestration `Efficacy` (and,
  later, `Backtest`) both consume — neither owns fitting or fold-looping
  itself.
- `Efficacy`: a pure, market-unaware prediction-quality scorer
  (`predicted_result` vs. `result` only — `spread_line` never enters this
  class).
- A runnable example: a new notebook exercising `SingleSplit` +
  `iterate_folds` + `Efficacy` against real `DataAccess`-sourced data,
  comparing a couple of candidate model configs.

## Non-goals

- `WalkForwardSplit` — deferred to the `Backtest` rework follow-on, which is
  where per-fold walk-forward validation actually gets exercised in
  production.
- `EnsembleModel` — already flagged as future work in `architecture-notes.md`.
- `TrainingResult` (`predictions`/`residuals`/`weights`/`metadata`) — see
  "`Model` protocol" below for why this is dropped, not deferred.
- `Backtest` rework (`SizingStrategy`, edge-aware sizing, `run_folds` pooled
  stats) — `backtest.py`/`run_backtest` are untouched by this spec.
- CLI wiring — this round's deliverable is library code, tests, and a
  notebook; a `degenebet` CLI command for predictions/backtests is a later
  increment.

## Design

### File layout

- `src/degenebet/modeling/splits.py` — new. `Model` protocol, `Split`,
  `SplitStrategy` protocol, `SingleSplit`, `FoldPredictions`,
  `iterate_folds`.
- `src/degenebet/modeling/efficacy.py` — new. `EfficacyResult`, `Efficacy`.
- `src/degenebet/modeling/spread_model.py`, `backtest.py`, `features.py` —
  **unchanged**. `SpreadModel` already structurally satisfies the `Model`
  protocol below; nothing about it needs to change for this spec.
- `notebooks/02_model_efficacy.py` — new.

### `Model` protocol

`architecture-notes.md`'s original sketch had `Model.fit(train_data) ->
TrainingResult`, with `TrainingResult` carrying `predictions`/`residuals`/
`weights`/`metadata`. Its own "open questions" section flagged this shape as
unvalidated against a real second `Model` implementation. Reconciling it
against `SpreadModel`'s actual, already-working interface —
`fit(model_table) -> None` (mutates internal state), `predict(model_table)
-> pl.DataFrame` (returns the input plus a `predicted_result` column,
already sitting next to `result`) — there's no need for a separate
"actual" carrier or a `TrainingResult` wrapper at all:

```python
class Model(Protocol):
    def fit(self, train_data: pl.DataFrame) -> None: ...
    def predict(self, data: pl.DataFrame) -> pl.DataFrame: ...
```

`data`'s required columns are implementation-specific, not part of the
protocol (`SpreadModel.predict` needs its own 6 feature columns; a future
`Model` could need different ones) — the protocol only fixes the shape:
`predict` returns `data` unchanged plus a `predicted_result` column, never
dropping or requiring `result`. `result` matters anyway, though: when
`iterate_folds` (below) calls `model.predict(split.test)`, `split.test`
already carries `result` from `model_table`, and `predict` passes it
through untouched — that's how `Efficacy` gets `predicted_result` sitting
next to `result` in the same frame without either `Model` or `iterate_folds`
having to know `Efficacy` exists.

This is a deliberate simplification, not a placeholder: `TrainingResult`'s
`residuals`/`weights`/`metadata` fields have no consumer in this spec's
design (nothing in `iterate_folds` or `Efficacy` reads them), and inventing
them now would be speculative. If a future second `Model` implementation
genuinely needs richer fit output, `TrainingResult` gets designed against
that real need then — not resurrected from an unvalidated sketch.

Consequence: `SpreadModel` and `backtest.py` need **zero changes** for this
spec. This round is purely additive.

### `Split`, `SplitStrategy`, `SingleSplit`

Bare `tuple[pl.DataFrame, pl.DataFrame]` return types get replaced with
named tuples throughout this design — the same reasoning that produced
`Gameweek` in the DataAccess redesign: positional tuples make call sites
remember field order/meaning; named tuples make them self-documenting.

```python
class Split(NamedTuple):
    train: pl.DataFrame
    test: pl.DataFrame

class SplitStrategy(Protocol):
    def splits(self, data: pl.DataFrame) -> Iterator[Split]: ...

class SingleSplit:
    """One (train, test) pair by season list. Mirrors run_backtest's
    existing train_seasons/test_seasons filtering and leakage guards,
    extracted here rather than reimplemented."""

    def __init__(self, train_seasons: list[int], test_seasons: list[int]) -> None: ...

    def splits(self, data: pl.DataFrame) -> Iterator[Split]:
        """Raises ValueError if train_seasons and test_seasons overlap
        (leakage), or if the train filter is empty (nothing to fit on).
        An empty test filter is not an error (e.g. an in-progress season
        with no played games yet) -- yields a zero-row test frame."""
        ...
```

`SingleSplit.splits()` does **not** filter null `result`/`spread_line` rows
the way `run_backtest` currently does internally — that filtering is
backtest-specific (it needs `spread_line` to place a bet). `SingleSplit`
only partitions by season; handling incomplete rows is `Efficacy`'s
explicit job (below), continuing the "no silent dropping inside a shared
function" precedent from the `build_model_table` retirement.

### `FoldPredictions`, `iterate_folds`

```python
class FoldPredictions(NamedTuple):
    in_sample: pl.DataFrame
    out_of_sample: pl.DataFrame

def iterate_folds(
    data: pl.DataFrame,
    split_strategy: SplitStrategy,
    model_factory: Callable[[], Model],
) -> Iterator[FoldPredictions]:
    """The one shared fit/predict orchestration -- neither Efficacy nor
    (later) Backtest owns fitting or fold-looping itself. model_factory
    (not a fixed Model instance) matters because SpreadModel is stateful:
    each fold needs a fresh model so fitted state never leaks across
    folds."""
    for split in split_strategy.splits(data):
        model = model_factory()
        model.fit(split.train)
        yield FoldPredictions(
            in_sample=model.predict(split.train),
            out_of_sample=model.predict(split.test),
        )
```

### `Efficacy`

```python
@dataclass(frozen=True)
class EfficacyResult:
    metrics: dict[str, float]       # pooled, out-of-sample only
    by_fold: pl.DataFrame | None    # one row per fold: in_sample_*/out_of_sample_* metrics

class Efficacy:
    def evaluate(self, predictions: pl.DataFrame) -> EfficacyResult: ...
    def evaluate_folds(self, folds: Iterable[FoldPredictions]) -> EfficacyResult: ...
```

**`evaluate(predictions)`** scores one frame (must have `predicted_result`
and `result` columns) — `predicted_result` vs. `result` only; `spread_line`
never enters this class (that's exclusively a `Backtest` concern, per
`architecture-notes.md`'s explicit "don't conflate the two 'win rate's"
note). Computes, as `EfficacyResult.metrics`:

- `rmse` — `sqrt(mean((predicted_result - result) ** 2))`.
- `r_squared` — `sklearn.metrics.r2_score(result, predicted_result)`.
- `directional_accuracy` — fraction of rows where
  `sign(predicted_result) == sign(result)` (did we call the winner
  correctly — **not** ATS win rate, which is a `Backtest` concept). A tied
  game (`result == 0`, rare but legal in NFL regular season) has
  `sign(result) == 0` and is counted as incorrect unless `predicted_result`
  is exactly `0.0` too — not worth special-casing for how rare it is.
- `information_coefficient` — Pearson correlation between `predicted_result`
  and `result` (`scipy.stats.pearsonr`).

Raises `ValueError` if `result` has any nulls (can't score against missing
ground truth) or if the frame is empty (zero rows) — loud, not silent NaN
metrics, same philosophy as `SpreadModel`'s null-feature guard from the
`build_model_table` retirement.

**`evaluate_folds(folds)`** consumes `iterate_folds`'s `FoldPredictions`
stream directly. Per fold, calls `evaluate()` on `fold.in_sample` and
`fold.out_of_sample` separately, populating one `by_fold` row (columns:
`fold` index, `in_sample_rmse`, `out_of_sample_rmse`,
`in_sample_r_squared`, `out_of_sample_r_squared`, etc. for every metric).
The top-level `metrics` is computed by **concatenating every fold's
`out_of_sample` frame and scoring once** — never by averaging each fold's
already-computed metrics — the same "pooled, not averaged" rule
`Backtest.run_folds` already documents, so a future `WalkForwardSplit` with
uneven fold sizes doesn't mis-weight. In-sample figures live only in
`by_fold` (the walk-forward-efficiency comparison, per Pardo); the
headline `metrics` is out-of-sample only, since that's the honest
generalization number a human compares across candidate model configs.

## Data flow (usage example)

```python
access = DataAccess(NflverseSource(), SharpApiSource())
team_data = access.get_team_data(start_week, end_week, as_of_date=date.today())
rolling = compute_rolling_features(team_data)
model_table = access.get_game_data(
    start_week, end_week, as_of_date=date.today(), team_data=rolling
).drop_nulls(subset=[...])  # explicit, per the null-feature guard's contract

split_strategy = SingleSplit(train_seasons=[2022, 2023], test_seasons=[2024])
efficacy = Efficacy()

for candidate_name, model_factory in candidates.items():
    folds = iterate_folds(model_table, split_strategy, model_factory)
    result = efficacy.evaluate_folds(folds)
    print(candidate_name, result.metrics)
```

This is exactly `architecture-notes.md`'s "step 1" workflow: loop over
candidate `model_factory` configs, call `evaluate_folds` per candidate,
compare metrics by eye, pick one by judgment. `Efficacy` does not iterate
over candidates itself, and never should (auto-selecting the best-scoring
config is backtest-overfitting, automated instead of manual) — that loop
stays human-driven, above `Efficacy`, exactly as sketched.

## Testing

- `SingleSplit.splits()`: overlap raises, empty-train raises, empty-test
  yields a zero-row frame (not an error), correct season partition.
- `iterate_folds`: a trivial fake `Model` test double proving each fold
  gets a fresh model instance (via `model_factory`) and that `fit`/`predict`
  are called with the right frames.
- `Efficacy.evaluate()`: hand-computed RMSE/R²/directional-accuracy/IC on
  small fixtures with known expected values (real small Polars DataFrames,
  same convention as `test_spread_model.py`/`test_features.py` — never
  mocked).
- `Efficacy.evaluate()`: null-`result` and empty-frame raise cases.
- `Efficacy.evaluate_folds()`: a fake `SplitStrategy` test double yielding
  2 synthetic folds (not `WalkForwardSplit`, which doesn't exist yet) to
  prove the pooling math is correct — differently-sized folds, so an
  averaging bug would be caught.
- `Efficacy.evaluate_folds()`: reconciliation check — calling it on a
  single fold produces the same `metrics` as calling `evaluate()` directly
  on that fold's `out_of_sample` frame.
- One `hypothesis` property test: `directional_accuracy` bounded to
  `[0, 1]`, `r_squared <= 1.0` — per this project's "golden fixture +
  property test per analytic surface" rule (`CLAUDE.md`'s Automation &
  Verification section).
- `notebooks/02_model_efficacy.py`: not part of `just check` (notebooks
  aren't CI-gated), but validated with `marimo check` and a headless
  `marimo export html` before committing, per this project's notebook
  convention.

## Notebook: `notebooks/02_model_efficacy.py`

Builds `model_table` via `DataAccess` + `compute_rolling_features` (same
pattern as `01_data_loading.py`), then runs 2-3 candidate `SpreadModel`
configs (e.g. default `LinearRegression` vs. a regularized variant) through
`SingleSplit` + `iterate_folds` + `Efficacy.evaluate_folds`, displaying each
candidate's `metrics` side by side — a runnable demonstration of the
model-selection workflow this spec's foundation exists to support. Reads
from the local cache only, via `DataAccess`, same as `01_data_loading.py`
— never live `nflreadpy`/SharpAPI calls.
