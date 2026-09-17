# Architecture design notes — Data/Model/Efficacy/Backtest rework

Running scratchpad from an ongoing brainstorming session, derived from
`docs/plan.md`. Not a formal spec yet — captures decisions as they're made
so nothing gets lost mid-conversation. Will be superseded by a proper spec
(or specs, if this decomposes into sub-projects) once the design settles.

## Pseudocode sketch (current)

Reflects every decision below — not the original draft. `...` marks an
unimplemented body; this is a shape to argue about, not working code.

```python
# ============================================================
# DATA
# ============================================================

class DataSource(Protocol):
    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame: ...

class NflverseSource:   # wraps existing degenebet.data.nflverse
    def fetch(self, start_date, end_date) -> pl.DataFrame: ...

class SharpApiSource:   # wraps existing degenebet.data.providers.sharpapi
    def fetch(self, start_date, end_date) -> pl.DataFrame: ...

class DataAccess:
    def __init__(self, historical: DataSource, current: DataSource): ...

    def get_team_data(
        self, start_date: date, end_date: date, as_of_date: date
    ) -> pl.DataFrame:
        """Data as it would have been known as of as_of_date.
        - as_of_date before earliest cached history -> warn, clamp to earliest.
        - historical rows from `historical`, capped at as_of_date.
        - gap between historical's latest observation and as_of_date filled
          from `current`, deduplicated on game_id so the two never overlap.
        """
        ...
    # get_player_data(...) later, same shape — hence "get_team_data" not
    # a generic "get_data".


# ============================================================
# FEATURES — no FeaturePipeline. features.py stays free functions:
#   compute_rolling_features(team_stats, *, window, min_history) -> pl.DataFrame
#   build_model_table(schedules, rolling_features) -> pl.DataFrame
# A future column-transform need (scaling, encoding) goes through
# sklearn's own Pipeline/StandardScaler, injected as SpreadModel's
# `model`, no wrapper class of our own.
# ============================================================


# ============================================================
# MODEL
# ============================================================

@dataclass(frozen=True)
class TrainingResult:
    """plan.md: 'weights, residuals, etc' — everything needed to evaluate
    a fit."""
    predictions: pl.DataFrame
    residuals: pl.DataFrame | None
    weights: dict[str, float] | None
    metadata: dict[str, Any]          # model class, fit timestamp, hyperparams

class Model(Protocol):
    def fit(self, train_data: pl.DataFrame) -> TrainingResult: ...
    def predict(self, data: pl.DataFrame) -> pl.DataFrame: ...

class SpreadModel:
    """NOT LinearSpreadModel — already composable via RegressorProtocol
    injection; naming it by algorithm would conflate task with algorithm."""
    def __init__(self, model: RegressorProtocol | None = None): ...
    def fit(self, train_data: pl.DataFrame) -> TrainingResult: ...
    def predict(self, data: pl.DataFrame) -> pl.DataFrame: ...

class EnsembleModel:   # not built yet — plan.md's ensembling ask
    def __init__(
        self, models: list[Model], combine: Callable[[list[pl.DataFrame]], pl.DataFrame]
    ): ...
    def fit(self, train_data) -> TrainingResult: ...
    def predict(self, data) -> pl.DataFrame: ...


# ============================================================
# SPLIT STRATEGY
# ============================================================

class SplitStrategy(Protocol):
    def splits(self, data: pl.DataFrame) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]: ...

class SingleSplit(SplitStrategy):
    """One (train, test) pair by season list. Fast; used for
    architecture/feature/hyperparameter screening."""
    def __init__(self, train_seasons: list[int], test_seasons: list[int]): ...

class WalkForwardSplit(SplitStrategy):
    """One (train, test) pair per week after warmup_seasons, expanding
    window. Validation today; the same loop IS production later — the
    final fold, predicting a week with no known result yet, is exactly
    'generate this week's picks.'"""
    def __init__(self, warmup_seasons: int = 1): ...


# ============================================================
# SHARED FOLD ORCHESTRATION — the only place model_factory is used.
# Neither Efficacy nor Backtest owns fitting or fold-looping.
# ============================================================

def iterate_folds(
    data: pl.DataFrame,
    split_strategy: SplitStrategy,
    model_factory: Callable[[], Model],
) -> Iterator[tuple[TrainingResult, pl.DataFrame]]:
    for train, test in split_strategy.splits(data):
        model = model_factory()
        training_result = model.fit(train)
        yield training_result, model.predict(test)


# ============================================================
# EFFICACY — pure scorer. predicted_result vs. result only; spread_line
# never enters this layer.
# ============================================================

@dataclass(frozen=True)
class EfficacyResult:
    metrics: dict[str, float]      # e.g. rmse, r_squared, directional_accuracy, information_coefficient
    by_fold: pl.DataFrame | None   # walk-forward efficiency: in-sample vs out-of-sample per fold

class Efficacy:
    def evaluate(self, training_result: TrainingResult, actual: pl.DataFrame) -> EfficacyResult:
        """Pure — no fitting, no fold-looping. Scores what one fold
        already produced: training_result.predictions vs actual['result']."""
        ...

    def evaluate_folds(self, folds: Iterable[tuple[TrainingResult, pl.DataFrame]]) -> EfficacyResult:
        """Runs evaluate() per fold from iterate_folds(...), aggregates.
        Does NOT iterate over candidate model configs itself — that loop
        is human-driven, above this class (see Workflow below)."""
        ...


# ============================================================
# BACKTEST — pure scorer, market-aware (spread_line), sizing-aware.
# Same run()/run_folds() shape as Efficacy's evaluate()/evaluate_folds().
# ============================================================

class SizingStrategy(Protocol):
    def size(self, predicted: pl.DataFrame) -> pl.DataFrame:
        """Input includes `edge` (predicted_result - spread_line) and
        `side`, so confidence-based sizing (larger |edge| -> larger bet)
        is possible once implemented. Adds a `units` column."""
        ...

class FlatSizing(SizingStrategy):
    """Today's implicit 1-unit-per-bet, made explicit. Ignores edge."""
    def size(self, predicted: pl.DataFrame) -> pl.DataFrame: ...

@dataclass(frozen=True)
class BacktestResult:
    bets: pl.DataFrame
    bets_placed: int
    win_rate: float
    units_won: float
    roi_pct: float
    by_fold: pl.DataFrame | None

class Backtest:
    def __init__(self, sizing_strategy: SizingStrategy, edge_threshold: float = 1.0): ...

    def run(self, training_result: TrainingResult, test_data: pl.DataFrame) -> BacktestResult:
        """Pure, single-fold, deterministic — no fitting. Computes
        edge = predicted_result - spread_line, decides side, sizes via
        sizing_strategy (which sees edge), scores win/loss/push at -110."""
        ...

    def run_folds(self, folds: Iterable[tuple[TrainingResult, pl.DataFrame]]) -> BacktestResult:
        """Calls run() once per fold, concatenates each fold's `bets`,
        and RECOMPUTES pooled stats from the concatenated raw bets —
        never by averaging each fold's already-computed win_rate/roi_pct
        (that would mis-weight folds with different bet counts)."""
        ...
```

## Settled

**DataAccess** (renamed from an earlier `PointInTimeDataAccess` draft):
- `get_team_data(start_date, end_date, as_of_date)` — named `get_team_data`
  specifically (not generic `get_data`) to leave room for `get_player_data`
  later.
- Stitches `NflverseSource` (history) + `SharpApiSource` (current) into one
  point-in-time-correct view; historical rows capped at `as_of_date`, any
  gap filled from the current source, deduplicated so the two never
  double-count the same game.
- `as_of_date` before earliest cached history: warn, clamp to earliest.

**No `FeaturePipeline` abstraction.** Two things got bundled under this
idea and both dissolve on inspection:
- Column-level transforms with a learned train-only parameter (scaling,
  imputation, encoding) are exactly sklearn's `Pipeline`/`StandardScaler`/
  etc. — and `SpreadModel`'s existing `RegressorProtocol` composition
  already accepts an sklearn `Pipeline` as the injected `model` with zero
  new code, since `Pipeline` implements `.fit(X, y)`/`.predict(X)` just
  like a bare regressor does.
- The rolling/no-lookahead team-efficiency computation (`compute_rolling_
  features`) was never sklearn-shaped in the first place (real temporal/
  per-team-group semantics, not i.i.d. rows or simple column stats) — it
  stays custom Polars code in `features.py`, as free functions, no wrapper
  class needed.
- Conclusion: don't build this, not even later for the "column transform"
  case — sklearn already solves that one and we already have the
  injection point.

**`SpreadModel` stays `SpreadModel`, not `LinearSpreadModel`.** It's
already composable with any `RegressorProtocol` (that was the whole point
of the Task 2 composition redesign) — naming it "Linear" would conflate
*which task* (predicting spread margin) with *which algorithm* (linear vs.
tree-based), undoing that decoupling. A future non-linear regressor is
`SpreadModel(model=XGBRegressor())`, not a new class. A future *different
task* (totals, moneyline) gets its own `Model` implementation, keyed on
target, not on algorithm.

**`SplitStrategy`** — unchanged from first draft: `SingleSplit` (one
train/test pair by season list, for fast screening) and `WalkForwardSplit`
(one pair per week after `warmup_seasons`, expanding window, for
validation and eventually production).

**`iterate_folds(data, split_strategy, model_factory)`** — the one shared
orchestration generator. Neither `Efficacy` nor `Backtest` owns fold-
looping or model-fitting itself; both consume this same stream of
`(TrainingResult, test_data)` pairs. Avoids re-deriving the same
fit/predict orchestration twice (the exact trap `run_walk_forward_backtest`
calling `run_backtest` internally was designed to avoid in the first
draft).

**`Efficacy`** — pure scorer, no fitting, no fold-looping, no
multi-configuration search:
- `evaluate(training_result, actual) -> EfficacyResult`: single fold.
- `evaluate_folds(folds) -> EfficacyResult`: runs `evaluate()` per fold
  from `iterate_folds(...)`, aggregates (this is where walk-forward
  efficiency — in-sample vs. out-of-sample per fold, per Pardo — would be
  computed).
- `EfficacyResult.metrics: dict[str, float]` — open dict, not hardcoded
  fields, since different tasks want different metrics (point-estimate:
  RMSE, R², directional accuracy "win ratio", information coefficient;
  probability-output tasks later: log-loss, Brier, calibration).
- **"Win ratio" here means directional accuracy** (`sign(predicted) ==
  sign(actual)` — did we call the winner correctly), NOT ATS win rate.
  ATS win rate depends on the market spread line and a bet-placement rule
  — that's a betting concept and belongs to `Backtest`, not prediction
  quality. Don't conflate the two "win rate"s.
- **Efficacy does NOT iterate over candidate model configurations
  (hyperparameter search) internally, and never should.** That loop lives
  *above* Efficacy — a human-driven script/notebook calling
  `evaluate_folds(...)` once per candidate config and comparing results by
  eye. If Efficacy auto-selected "whichever config scored best," that's
  backtest-overfitting (de Prado's warning from earlier in this session),
  just automated instead of manual. Model/hyperparameter selection is a
  deliberate human-judgment step, not something any class does for you.

**`Backtest`** — pure scorer, same shape as Efficacy:
- `run(training_result, test_data) -> BacktestResult`: single fold,
  deterministic — decide bets from `training_result.predictions` vs.
  `test_data`'s spread_line, size via `SizingStrategy`, score win/loss/
  push. No fitting happens here — this is what you want when you already
  have weights (e.g. from a single-split Efficacy check) and just want to
  score them.
- `run_folds(folds) -> BacktestResult`: calls `run()` once per fold,
  concatenates each fold's `bets`, and **recomputes pooled stats from the
  concatenated raw bets — never by averaging each fold's already-computed
  win_rate/roi_pct.** Averaging per-fold percentages weights a 2-bet fold
  equally to a 20-bet fold, which is wrong; concatenating first gets the
  weighting right for free. `by_fold` (the per-season/per-fold breakdown)
  is legitimately computed per-fold — that's a different, valid question
  ("how'd it do *within* this fold") from the pooled number.
- "Re-fit every observation or not" is fully determined by which
  `SplitStrategy` is passed to `iterate_folds` — no separate flag needed
  on `Backtest`.

**`edge` (`predicted_result - spread_line`) is a sizing signal, not just a
threshold gate.** Its magnitude is a natural confidence measure — matches
`plan.md`'s own roadmap ("size based on confidence of prediction" as the
eventual third sizing tier, after flat and odds-based). Implication:
`Backtest.run()` must compute `edge` and carry it through to whatever it
hands `SizingStrategy.size(...)` — not just the post-filtered `side` —
so a future `EdgeProportionalSizing` can read it. `FlatSizing` ignores it.

**Efficacy metrics for the spread model = `predicted_result` vs. `result`
(actual game margin) only — `spread_line` never enters Efficacy at all.**
RMSE/R²/IC/directional-accuracy are all predicted-vs-actual-margin
comparisons. This is worth stating explicitly because "spread" is
overloaded in this domain — the model's predicted margin gets casually
called "predicted spread" too, easy to conflate with the market's
`spread_line`, which is exclusively a Backtest concern.

## Workflow this architecture is meant to support

1. **Model/hyperparameter selection (human-in-the-loop):** loop over
   candidate `RegressorProtocol` configs, call `Efficacy.evaluate_folds`
   per candidate, look at the metrics/plots, pick one by judgment. This is
   the *only* place model hyperparameters get chosen or changed.
2. **Backtest-parameter selection (human-in-the-loop, separate loop):**
   with the model frozen from step 1, loop over candidate `edge_threshold`/
   `SizingStrategy` configs, call `Backtest.run_folds(...)` per candidate
   (model_factory held fixed), compare PnL curves/ROI by eye, pick one by
   judgment. Structurally identical to step 1's loop, one layer downstream
   — Efficacy can't evaluate this step at all (it never sees `spread_line`
   or places bets), which is *why* it has to be a separate loop rather than
   folded into step 1.
3. **Backtest, mechanical (no human input):** with both the model
   hyperparameters (step 1) and edge_threshold/sizing (step 2) frozen, run
   `Backtest.run_folds(iterate_folds(data, WalkForwardSplit(...),
   frozen_model_factory))`. Weights change every fold (the model absorbing
   new data); nothing tuned in steps 1-2 does. Changing either means going
   back to the relevant loop.
4. **PnL stream** falls out of `BacktestResult.bets` for free (game_id/
   season/week + `units` per bet) — cumulative sum of `units` ordered by
   time is the equity curve. No new field needed; this is where sizing
   strategy's effect actually becomes visible once real sizing exists.

## Open questions / not yet settled

- `TrainingResult` exact shape (predictions/residuals/weights/metadata) —
  sketched, not stress-tested against a real second `Model` implementation
  yet.
- `SizingStrategy` interface details beyond `FlatSizing` — needs `edge`
  available (see above), shape of `EdgeProportionalSizing`/odds-based
  sizing not designed yet.
- Data layer: exact `NflverseSource`/`SharpApiSource` interfaces, how
  "current" data ages out into "historical" over time.
