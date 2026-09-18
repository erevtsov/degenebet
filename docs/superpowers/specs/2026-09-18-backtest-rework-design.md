# Backtest Rework — Design Spec

## Context

`docs/architecture-notes.md` sketched `Backtest` as "pure scorer, market-aware
(`spread_line`), sizing-aware," with the same `run()`/`run_folds()` shape as
`Efficacy`'s `evaluate()`/`evaluate_folds()`. It was deliberately scoped out
of the modeling-efficacy-foundation spec/plan (PR #20/#21/#22) as a follow-on
sub-project, once `Model`/`iterate_folds`/`Efficacy` existed to build on. They
now do.

This spec reconciles the original sketch against what actually shipped in
that foundation — `TrainingResult` was dropped in favor of `FoldPredictions`
carrying precomputed `in_sample`/`out_of_sample` prediction frames directly,
which changes `Backtest.run()`'s signature from the sketch's
`run(training_result, test_data)` to a single frame, matching `Efficacy`'s
own `evaluate(predictions)` — and goes further than the sketch in one
respect: the sketch's `_UNITS_RISKED_PER_BET = 1.1` constant (still present
in today's `run_backtest`) hardcodes -110 odds for every bet. Real per-game,
per-side American odds already exist in `model_table` via nflverse
(`home_spread_odds`/`away_spread_odds` — verified populated with zero nulls
2010–2024, genuinely asymmetric per side, e.g. home -102/away -118, and
spanning both signs, -140 to +129). Scoring every bet at a fixed -110 makes
`Backtest` barely distinguishable from `Efficacy`'s directional-accuracy
metric; using the real per-bet odds is what actually makes it
"market-aware" in the sense the sketch's own header comment claims. This
spec builds that in from the start rather than deferring it.

`run_backtest`/`BacktestResult` (today's single-split baseline) are used
only by `backtest.py` itself and three test files (`test_backtest.py`,
`test_property_invariants.py`, `test_real_data_sanity.py`) — no notebook, no
CLI. This is a full replace, not an addition alongside the old code,
matching this project's established pattern (the DataAccess redesign,
`build_model_table`'s retirement): never keep two parallel implementations
of the same concept in the codebase.

## Goals

- `SizingStrategy` (`FlatSizing` today) deciding bet size (`stake`),
  decoupled from the win/loss/push scoring that uses real per-bet odds.
- `Backtest.run()`/`run_folds()`, mirroring `Efficacy.evaluate()`/
  `evaluate_folds()`'s shape: one frame in for `run()`, `FoldPredictions`
  stream in for `run_folds()`, pooled-not-averaged stats.
- Real per-bet American odds (`home_spread_odds`/`away_spread_odds`)
  replacing the hardcoded -110 assumption, via a general American-odds
  payout conversion (`_win_multiplier`) that works for both signs.
- `WalkForwardSplit`: one fold per `gameweek` after a warmup period,
  expanding training window — the `SplitStrategy` implementation deferred
  from the last round, now what actually makes `run_folds`'s pooled stats
  and `by_fold` breakdown pay off (`SingleSplit` only ever produces one
  fold).
- `compute_bankroll_trajectory`: a real, dollar-denominated, compounding
  bankroll trajectory (week 2's stake sized off week 1's actual realized
  outcome), computed as a cheap sequential pass *over already-computed,
  independently-derived per-fold results* — not by threading a live
  bankroll through fold computation itself. See "Compounding bankroll."
- **Design property (not a hard implementation requirement this round):**
  `run_folds()`'s per-fold work (fit, predict, decide, size, score) has no
  dependency on any other fold's result — every fold could be computed in
  parallel. Nothing in this spec mandates wiring up an actual parallel
  executor now; the point is that the design doesn't preclude it, and
  a sequential loop remains a correct (if not maximally fast) reference
  implementation.
- A new notebook, `notebooks/03_backtest.py`, running `Backtest` +
  `WalkForwardSplit` against real data, including the bankroll trajectory.

## Non-goals

- `EdgeProportionalSizing` or Kelly-criterion sizing — not built now, but
  `win_multiplier` is deliberately made available to `SizingStrategy.size()`
  (see Design) so a future Kelly-style strategy doesn't need a redesign to
  get the odds it needs.
- A `SizingStrategy` whose stake allocation is *not* linear in the pool
  size it's given (e.g. a fixed-dollar-amount-regardless-of-bankroll rule,
  capped by what's left) — `compute_bankroll_trajectory` only composes
  correctly with a linear rule like `FlatSizing`'s; making it work for a
  non-linear rule is real future work, not attempted here.
- Actually parallelizing `run_folds()`'s fold loop (e.g. via
  `concurrent.futures`) — see the Goals note above; this round establishes
  the design property, not the executor.
- CLI wiring — still deferred, as in every prior round.
- A live "generate this week's picks" feature (predicting on unplayed
  games). `Backtest` only ever scores *played* games — see the null-handling
  contract in Design. If this is built later, it doesn't call
  `Backtest.run()` at all (there's nothing to score); it's a different,
  simpler operation (predict + decide side, no win/loss/push).
- SharpAPI's live current-odds pricing is not used anywhere in `Backtest` —
  only nflverse's historical `home_spread_odds`/`away_spread_odds` (the
  actual closing-line prices for played games). `DataAccess`'s
  current/historical stitching already handles `spread_line` reconciliation;
  this spec adds nothing new there.

## Design

### File layout

- `src/degenebet/modeling/splits.py` — add `WalkForwardSplit`, alongside
  the existing `Split`/`SplitStrategy`/`SingleSplit`/`Model`/
  `FoldPredictions`/`iterate_folds`.
- `src/degenebet/modeling/backtest.py` — full replace. `run_backtest` and
  the old `BacktestResult` are deleted; `SizingStrategy`, `FlatSizing`,
  `BacktestResult` (new shape), `Backtest`, four private helpers
  (`_win_multiplier`, `_decide_bets`, `_score_bets`, `_summarize_bets`),
  and `BankrollTrajectory`/`compute_bankroll_trajectory` take their place.
- `tests/modeling/test_backtest.py` — rewritten against the new API.
- `tests/modeling/test_splits.py` — gains `WalkForwardSplit` tests.
- `tests/modeling/test_property_invariants.py` — the existing "backtest
  P&L reconciles under re-aggregation" property test updated to exercise
  the new `Backtest`/`WalkForwardSplit` (the reconciliation property itself
  — pooled equals concatenated-then-summed per-fold `units` — is unchanged
  by this spec; only the API it's exercised through changes).
- `tests/modeling/test_real_data_sanity.py` — updated to build a
  `Backtest` and call `run()`/`run_folds()` instead of `run_backtest()`.
- `notebooks/03_backtest.py` — new.

### Terminology: `size`, `stake`, `win_multiplier`, `units`

Resolved through discussion, not the original sketch's wording (which used
`units` for both bet size and outcome, ambiguously):

- **`size` / `SizingStrategy.size()`** — the action of deciding bet size.
  Not a stored value itself.
- **`stake`** — the *output* of `size()`: the amount *risked* on one bet,
  as a fraction of whatever pool `size()` was given to divide (see
  `FlatSizing` below and "Compounding bankroll"). This is a deliberate
  departure from today's `run_backtest`, which sizes its implicit unit to
  a target *win* amount instead (risk `1.1` to win `1.0`, at -110) — under
  that older convention, a win nets `+1.0` and a loss nets `-1.1`; under
  this spec's convention, a `stake=s` win at -110 nets `+s * 0.909`
  (`s * win_multiplier`) and a loss nets `-s` (`-stake`, always — you lose
  what you risked). Neither convention is more "correct"; this spec picks
  amount-risked because that's what "bet size" ordinarily means
  (bankroll-percentage and Kelly-style sizing are always framed as risk
  amount, not target payout). Only `units`'s magnitude changes relative to
  today's `run_backtest` — no win/loss/push *decision* depends on which
  convention is used.
- **`win_multiplier`** — American odds converted to profit-per-unit-risked:
  `100/abs(odds)` for negative odds (e.g. -110 → 0.909), `odds/100` for
  positive odds (e.g. +120 → 1.2). Derived from `bet_odds` (below), not
  stored as raw odds — every consumer (`SizingStrategy`, the final scoring
  step) uses the already-converted multiplier, never has to reimplement
  the American-odds formula itself.
- **`bet_odds`** — the raw American odds for the side actually bet:
  `home_spread_odds` if `side == "home"`, `away_spread_odds` if
  `side == "away"`. Only resolved for placed bets — the side not bet
  doesn't matter.
- **`units`** — the *realized outcome* of one settled bet, only meaningful
  after the game's `result` is known: `+stake * win_multiplier` if won,
  `-stake` if lost, `0` if push. **`FlatSizing` dividing its pool evenly
  does not mean the backtest ignores odds** — every bet is still scored at
  its own game's real `win_multiplier` regardless of which `SizingStrategy`
  was used. "Flat" describes the stake-division rule only; the payout
  calculation is always market-real.
- **`units_won`** (`BacktestResult` field) — `sum(units)` across every
  placed bet.
- **`roi_pct`** — `units_won / sum(stake) * 100`. Simpler than today's
  `bets_placed * 1.1` denominator: since `stake` now directly means
  "amount risked," summing it *is* the total amount put at risk, with no
  conversion constant needed.

### `_win_multiplier`

```python
def _win_multiplier(odds: int) -> float:
    """American odds -> profit per unit risked. Negative odds (e.g. -110,
    risk $1.10 to win $1.00): 100/abs(odds). Positive odds (e.g. +120,
    risk $100 to win $120): odds/100."""
    return 100.0 / abs(odds) if odds < 0 else odds / 100.0
```

### `SizingStrategy`, `FlatSizing`

```python
class SizingStrategy(Protocol):
    def size(self, predictions: pl.DataFrame) -> pl.DataFrame: ...
    """predictions already has edge, side, and win_multiplier computed
    (win_multiplier so a future odds-aware strategy, e.g. Kelly, can size
    against the real payout, not just the edge). Adds a stake column."""

class FlatSizing:
    """Divides a normalized pool of 1.0 evenly across every bet in this
    call -- stake_i = 1.0 / N for N placed bets, regardless of edge or
    odds. This is "flat" in the sense of splitting a period's betting
    pool evenly across that period's bets, not a constant per-bet
    dollar amount (a prior draft of this spec used the latter; revised
    after discussion, see "Compounding bankroll" below for why the pool
    interpretation is what makes fold-independent computation possible).
    The backtest still scores each bet at its own real payout odds;
    FlatSizing only decides how the pool is split, never the payout."""
    def size(self, predictions: pl.DataFrame) -> pl.DataFrame:
        n = predictions.height
        return predictions.with_columns(pl.lit(1.0 / n if n > 0 else 0.0).alias("stake"))
```

### Null-handling: caller opts in explicitly, `Backtest` never filters

Resolved through discussion: `Backtest` never silently drops unplayed
games. This mirrors `SpreadModel`'s existing null-feature guard and
`Efficacy._compute_metrics`'s null-`result` guard exactly — the caller
filters explicitly before the data ever reaches `iterate_folds`/`Backtest`
(e.g. `model_table.filter(pl.col("result").is_not_null() &
pl.col("spread_line").is_not_null())`), which in practice already has to
happen before `SpreadModel.fit()` can train at all (it can't fit against a
null target either). This is a deliberate change from today's
`run_backtest`, which silently filters unplayed rows internally.

Two separate guards, at two separate pipeline points, because they depend
on different things being resolved first:

1. **`predicted_result`/`result`/`spread_line`** must be non-null for
   *every* row passed to `_decide_bets` — needed to compute `edge` and
   decide `side` at all. Raises `ValueError` naming the null columns if
   not.
2. **`bet_odds`** (the side-specific odds) must be non-null for every
   *placed* bet — but this can only be checked after `side` is decided
   (a null `away_spread_odds` doesn't matter for a game where `side ==
   "home"`). Raises `ValueError` if not, after filtering to placed bets.

One thing this spec does **not** change: `run_backtest` today treats a
genuinely **empty** predictions frame (zero rows — e.g. no games in a
future test season yet, or every game's `edge` fell below `edge_threshold`
that week) as legitimate, returning a zeroed `BacktestResult` rather than
raising. That distinction stays: nulls on rows that exist are a caller
error (raise); zero rows is a legitimate "nothing to bet on" outcome (zeroed
result, not an error) — same as `SingleSplit`'s own "empty test filter is
not an error" design.

### Pipeline (`_decide_bets`, `_score_bets`, `_summarize_bets`)

```python
def _decide_bets(predictions: pl.DataFrame, edge_threshold: float) -> pl.DataFrame:
    """Guard 1 (predicted_result/result/spread_line non-null), compute
    edge = predicted_result - spread_line, decide side (home if
    edge > edge_threshold, away if edge < -edge_threshold, else none),
    filter to placed bets, resolve bet_odds and win_multiplier. Guard 2
    (bet_odds non-null among placed bets)."""
    ...

def _score_bets(sized: pl.DataFrame) -> pl.DataFrame:
    """sized already has stake (from SizingStrategy.size()). Computes
    home_cover_margin = result - spread_line, push/won, and units =
    stake * win_multiplier if won, -stake if lost, 0 if push."""
    ...

def _summarize_bets(bets_df: pl.DataFrame) -> dict[str, float | int]:
    """bets_placed, ats_win_rate, units_won, roi_pct. Returns zeroed
    values if bets_df is empty (0 rows) -- not an error, see null-handling
    above. Shared by run() and both the per-fold and pooled calls inside
    run_folds() -- same DRY principle as Efficacy's _compute_metrics."""
    ...
```

`_decide_bets` never raises on an empty (0-row) `predictions` input —
only on nulls in rows that exist. `run()`/`run_folds()` need no special
"nothing to bet on" branch of their own: `_decide_bets` and
`SizingStrategy.size()` (including `FlatSizing`) preserve a 0-row frame's
schema without erroring, `_score_bets` likewise, and `_summarize_bets`'s
own 0-row branch is the single place that turns "no placed bets" into a
zeroed result. One code path handles both "the whole test season is empty"
and "every game's edge fell below `edge_threshold` this week."

### `BacktestResult`, `Backtest`

```python
@dataclass(frozen=True)
class BacktestResult:
    bets: pl.DataFrame
    bets_placed: int
    ats_win_rate: float       # NOT Efficacy's directional_accuracy -- see
                               # architecture-notes.md's explicit warning
                               # against conflating the two "win rate"s
    units_won: float
    roi_pct: float
    by_fold: pl.DataFrame | None    # one row per fold: fold, bets_placed,
                                     # ats_win_rate, units_won, roi_pct

class Backtest:
    def __init__(self, sizing_strategy: SizingStrategy, edge_threshold: float = 1.0) -> None: ...

    def run(self, predictions: pl.DataFrame) -> BacktestResult:
        decided = _decide_bets(predictions, self.edge_threshold)
        bets_df = _score_bets(self.sizing_strategy.size(decided))
        return BacktestResult(bets=bets_df, by_fold=None, **_summarize_bets(bets_df))

    def run_folds(self, folds: Iterable[FoldPredictions]) -> BacktestResult:
        """Scores fold.out_of_sample only -- never in_sample; you don't
        bet on training data. Raises ValueError on an empty folds stream
        (zero folds total -- distinct from a fold with zero placed bets,
        which is legitimate). Pooled stats computed by concatenating every
        fold's scored bets and summarizing once -- never by averaging
        each fold's already-computed stats, same principle
        Efficacy.evaluate_folds established."""
        ...
```

`by_fold` here is simpler than `Efficacy`'s (no in-sample/out-of-sample
sample tag) — `Backtest` never scores in-sample data at all, so there's
only one kind of row: one per fold.

### `WalkForwardSplit`

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

Every fold's `test` is non-empty by construction (`gameweeks` is derived
directly from `data`'s own rows), unlike `SingleSplit`'s caller-supplied
season list — no separate empty-test guard needed here.

### Compounding bankroll

Resolved through discussion. The request: size a `WalkForwardSplit` week's
bets off the *actual, running* bankroll — start with `$50`, lose `$30` in
week 1, week 2 only has `$20` to bet with. The tension: computing that
requires knowing every prior week's real dollar outcome first, which
looks like it forces `run_folds()`'s per-fold loop to run strictly in
time order — directly opposed to the other goal, computing folds
independently (each fold's fit/predict/decide/score has no dependency on
any other fold, a design property worth preserving even if this round
doesn't wire up an actual parallel executor for it).

These resolve because `FlatSizing`'s "divide a normalized pool of `1.0`
evenly across this fold's bets" rule is *linear* in the pool size: for a
fixed set of bets, doubling the pool exactly doubles every bet's stake
and therefore the fold's total P&L. That means a fold's *fractional*
return (P&L as a fraction of whatever pool it was given) is identical
whether the pool is `1.0` or `$50,000` — so `BacktestResult.by_fold`'s
existing `units_won` column, computed once per fold with the normalized
`1.0` pool (exactly what `run_folds()` already does, no new field needed),
*is* that fold's fractional return, and it's valid to compute
independently per fold, in any order, before any bankroll is known.

Turning that sequence of fractional returns into an actual dollar
trajectory is a second, separate, sequential pass — but a cheap one (pure
arithmetic over a small per-fold table, not model fitting), so there's no
real cost to keeping it sequential:

```python
@dataclass(frozen=True)
class BankrollTrajectory:
    by_fold: pl.DataFrame        # fold, bankroll_before, pnl, bankroll_after
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
    re-sorted before calling this. Verified against the motivating
    example: starting_bankroll=50, week 1's units_won=-0.6 (lost $30 of
    $50) -> bankroll_after = 50 * 0.4 = 20, matching "$20 left for week
    2" exactly."""
    ...
```

**This only works because `FlatSizing`'s rule is linear in the pool
size** — a documented assumption, not something the types enforce. A
future non-linear `SizingStrategy` (e.g. "bet a fixed $10 regardless of
bankroll, capped by what's left") would not compose with
`compute_bankroll_trajectory` the same way; that's explicitly out of
scope here (see Non-goals).

**Consequence for `BacktestResult`'s existing pooled stats:** under the
old "constant `stake=1.0` per bet" `FlatSizing`, pooling by concatenating
every fold's bets naturally weighted busier weeks (more qualifying games)
more heavily. Under the new "divide a `1.0` pool per fold" rule, every
fold contributes the same total stake (`1.0`) regardless of how many bets
it placed, so the pooled `roi_pct`/`units_won` in `run_folds()`'s
top-level `BacktestResult` now represent an **equal-weighted average
across folds** (each gameweek counts the same, whether it had 2 games or
14) — a real, deliberate consequence of the redefinition, not a bug. This
is a genuinely different (and still useful) question from
`compute_bankroll_trajectory`'s compounding view: "what's my average
weekly return, treating every week the same" vs. "what's my actual dollar
trajectory if I reinvest my whole bankroll every week."

## Data flow (usage example)

```python
model_table = access.get_game_data(
    start_week, end_week, as_of_date=date.today(), team_data=rolling
).drop_nulls(subset=[...])  # feature columns, per SpreadModel's contract
model_table = model_table.filter(
    pl.col("result").is_not_null() & pl.col("spread_line").is_not_null()
)  # explicit opt-in: only played games reach Backtest

split_strategy = WalkForwardSplit(warmup_seasons=2)
backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)

folds = iterate_folds(model_table, split_strategy, lambda: SpreadModel())
result = backtest.run_folds(folds)
print(result.ats_win_rate, result.units_won, result.roi_pct)  # equal-weighted-across-folds view

trajectory = compute_bankroll_trajectory(result.by_fold, starting_bankroll=1000.0)
print(trajectory.ending_bankroll, trajectory.total_pnl)  # compounding, dollar-denominated view
```

## Testing

- `_win_multiplier`: hand-computed cases for negative and positive
  American odds.
- `_decide_bets`: null-guard tests for both guard points (predicted_result/
  result/spread_line; bet_odds among placed bets only), edge/side decision
  correctness (home/away/none boundaries at `edge_threshold`), empty-input
  handling.
- `FlatSizing`: stake is `1.0 / N` for N placed bets in one call (e.g. 4
  bets each get `0.25`), stakes sum to `1.0` for the call; `0.0` stake
  (not a division error) on a zero-row input.
- `Backtest.run()`: end-to-end hand-computed test using real, asymmetric
  odds (not -110 on both sides) verifying `units = stake * win_multiplier`
  on a win and `units = -stake` on a loss;
  push handling; `edge_threshold` filtering; empty predictions (zero rows,
  not nulls) returns a zeroed `BacktestResult`, not an error.
- `Backtest.run_folds()`: pooled-not-averaged test with differently-sized
  folds (same pattern as `Efficacy.evaluate_folds`'s test); `by_fold` row
  count/shape; empty-folds-stream raises.
- `WalkForwardSplit`: fold count matches expected post-warmup gameweeks;
  `train` genuinely expands fold-to-fold and never includes the current or
  a future `gameweek`; `warmup_seasons` boundary (first test fold is the
  first gameweek of the correct season); empty-train guard.
- `test_property_invariants.py`'s existing P&L-reconciliation property
  test updated to build a `Backtest`/`WalkForwardSplit` (or `SingleSplit`)
  pair instead of calling `run_backtest` — the property itself (pooled
  `units_won` reconciles under re-aggregation) is unchanged.
- `test_real_data_sanity.py` updated to the new `Backtest` API, same
  plausible-`ats_win_rate`-band sanity check against the frozen real
  fixture.
- `compute_bankroll_trajectory`: the motivating worked example itself
  (`starting_bankroll=50`, one fold with `units_won=-0.6` →
  `bankroll_after=20`) as a hand-computed test; multi-fold compounding
  (verify `bankroll_after` of fold N equals `bankroll_before` of fold
  N+1, and `total_pnl == ending_bankroll - starting_bankroll`); a fold
  with `units_won=0` (no bets placed, or all pushes) leaves the bankroll
  unchanged.

## Notebook: `notebooks/03_backtest.py`

Builds `model_table` via `DataAccess` + `compute_rolling_features` (same
pattern as `01_data_loading.py`/`02_model_efficacy.py`), filters to played
games explicitly (per the null-handling contract), then runs `Backtest` +
`WalkForwardSplit` against a `SpreadModel` candidate, displaying the
pooled, equal-weighted `ats_win_rate`/`units_won`/`roi_pct`, and a real
dollar equity-curve chart from `compute_bankroll_trajectory(result.by_fold,
starting_bankroll=...)` — the actual compounding trajectory, not just the
per-fold fractional returns. Reads from the local cache only, via
`DataAccess` — never live `nflreadpy`/SharpAPI calls, same as the two prior
notebooks.
