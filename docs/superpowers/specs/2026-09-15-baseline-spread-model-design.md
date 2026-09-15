# Baseline Spread Model + Backtest — Design Spec

Date: 2026-09-15
Status: Approved for implementation planning

## Context

Second sub-project of `degenebet`, building on the completed data-foundation
sub-project (historical schedules/closing lines via `nflreadpy`, current
odds via SharpAPI). Per the user's explicit approach: start with the
simplest reasonable model (linear regression) for one market (point
spread/ATS), build a backtest harness that can evaluate it, and iterate
toward more complex models later. Per `AGENTS.md`'s "Automation &
Verification" section (added in the prior agentic-automation sub-project),
this sub-project must ship golden regression fixtures and `hypothesis`
property-based tests alongside the model, not after.

All data facts below were verified directly against the real, working data
pipeline (not assumed) as part of this brainstorming session — including
catching and fixing a real bug (`nflreadpy`'s cache config silently broke on
a `str` vs `Path` type mismatch) before trusting the data layer for design
work.

## Goals

- A linear regression model predicting a game's margin of victory
  (`result = home_score - away_score`) from a small set of team efficiency
  features.
- A backtest harness that evaluates the model against real historical
  closing lines: given a train/test season split, report ATS win rate and
  ROI at standard -110 pricing for a flat-unit staking rule.
- Golden regression fixtures (pinned model output against a fixed
  historical data slice) and property-based tests for the invariants that
  would be expensive to get silently wrong.

## Non-goals (deferred, not decided against)

- **Walk-forward / rolling retraining backtest.** V1 uses a single
  train/test season split. Retraining weekly as new data arrives is a more
  realistic simulation but adds real complexity (correctly refitting and
  re-deriving rolling features at each step) — worth doing once the single-
  split model is validated and the codebase has a working baseline to
  compare against.
- **Other markets** (totals, moneyline) and **other model classes** (trees,
  gradient boosting, etc.) — this spec is deliberately narrow to point
  spread + linear regression. Both are natural follow-on specs once this
  one's patterns (feature pipeline, golden fixtures, backtest harness
  shape) are established.
- **Bet sizing / bankroll management.** The backtest uses flat 1-unit bets.
  Kelly criterion or any bankroll-aware sizing is its own future
  sub-project per the original project decomposition; this backtest's
  `cover_probability` output is designed to be a clean input to that later
  work, but doesn't do sizing itself.
- **Player-level features.** Team-level efficiency only for v1.

## Data facts verified during design (not assumptions)

- `degenebet.data.load_schedules()` already provides `result` (a column
  equal to `home_score - away_score`, confirmed by direct inspection — not
  computed manually) and `spread_line` with **zero nulls** for 2022-2023.
- `spread_line` sign convention (verified against 5 real lopsided 2023
  games): **negative means the home team is favored** by that many points
  (e.g. `-3.5` = home favored by 3.5). Home team's cover margin is
  `result + spread_line`: positive → home covers, negative → away covers,
  zero → push. This formula was checked against all 5 sample games and
  matched in every case.
- `degenebet.data.load_team_stats()` provides one row per team per game
  (`game_id`, `team`, `opponent_team` columns present) with `passing_epa`,
  `rushing_epa`, `attempts`, `carries` — all confirmed populated (zero
  nulls in a 2023 sample) — but **no direct "defensive EPA allowed"
  column**. Team stats are offense-oriented; a team's defensive EPA
  allowed in a game must be derived as *the opponent's offensive EPA in
  that same `game_id`* (self-join on `game_id` where `team` = the other
  row's `opponent_team`).
- `receiving_epa` is a separate column from `passing_epa` but both derive
  from the same pass plays (QB-perspective vs. receiver-perspective stats
  on the same play) — using both together would double-count. Offensive
  EPA/play uses only `passing_epa + rushing_epa`.

## Architecture

New subpackage `src/degenebet/modeling/`:

```
src/degenebet/modeling/
  __init__.py
  features.py       # rolling team efficiency features, no-lookahead
  spread_model.py     # SpreadModel: fit/predict/cover_probability
  backtest.py           # single-split backtest harness + result reporting
tests/modeling/
  __init__.py
  test_features.py
  test_spread_model.py
  test_backtest.py
  _golden_spread_fixture.py
  test_spread_model_golden.py
  test_property_invariants.py
```

### `modeling/features.py`

Per-team-per-game derived metrics (before any rolling window):

```python
team_offense_epa_per_play = (passing_epa + rushing_epa) / (attempts + carries)
team_defense_epa_allowed_per_play = <opponent's team_offense_epa_per_play in the same game_id>
team_turnover_margin = (def_interceptions + fumble_recovery_opp) - (passing_interceptions + fumbles_lost_total)
```

Public function:

```python
def compute_rolling_features(
    team_stats: pl.DataFrame,
    *,
    window: int = 8,
    min_history: int = 4,
) -> pl.DataFrame:
    """One row per (team, game_id): trailing rolling averages of the three
    per-game metrics above, computed strictly from that team's games
    *before* this one (window may span a season boundary). Rows where a
    team has fewer than `min_history` prior games are dropped, not
    padded — there is no legitimate rolling average to report yet.
    """
```

Second public function joins this onto schedules to build the model's
training/prediction table:

```python
def build_model_table(
    schedules: pl.DataFrame,
    rolling_features: pl.DataFrame,
) -> pl.DataFrame:
    """One row per game with home_offense_epa, home_defense_epa_allowed,
    home_turnover_margin, away_offense_epa, away_defense_epa_allowed,
    away_turnover_margin, result, spread_line, season, week, game_id.
    Games where either team lacks rolling features (see min_history above)
    are dropped.
    """
```

### `modeling/spread_model.py`

```python
class SpreadModel:
    """Linear regression predicting `result` from 6 rolling team features."""

    def fit(self, model_table: pl.DataFrame) -> None:
        """Fits on the 6 feature columns against `result`. Stores the
        fitted coefficients and the training residual std (needed for
        cover_probability)."""

    def predict(self, model_table: pl.DataFrame) -> pl.DataFrame:
        """Returns model_table with a new `predicted_result` column."""

    def cover_probability(self, model_table: pl.DataFrame) -> pl.DataFrame:
        """Requires `spread_line` and `predicted_result` present. Adds a
        `home_cover_probability` column via
        normal_cdf((predicted_result + spread_line) / residual_std).
        Must call predict() first, or this calls it internally."""
```

Uses `scikit-learn`'s `LinearRegression` internally (new dependency — user
approved adding it during brainstorming, in preference to hand-rolling OLS,
since it's the standard tool and leaves room to swap in other sklearn model
classes later without changing this module's shape). Polars → numpy
conversion happens at the sklearn call boundary only (`.to_numpy()`); no
pandas anywhere, consistent with the project's existing conventions.

### `modeling/backtest.py`

```python
@dataclass(frozen=True)
class BacktestResult:
    bets: pl.DataFrame        # one row per bet placed: game_id, side, result, won, units
    bets_placed: int
    ats_win_rate: float        # excludes pushes
    units_won: float
    roi_pct: float

def run_backtest(
    model_table: pl.DataFrame,
    *,
    train_seasons: list[int],
    test_seasons: list[int],
    edge_threshold: float = 1.0,
) -> BacktestResult:
    """Fits a fresh SpreadModel on train_seasons only, predicts on
    test_seasons. For each test game, bets 1 unit on the side (home or
    away) where |predicted_result - (-spread_line)| exceeds edge_threshold;
    skips games with no sufficient edge. Scores each bet against the real
    `result` using the cover-margin formula (see Data facts above) at -110
    pricing (risk 1.1 units to win 1.0; push refunds the bet, excluded from
    win rate)."""
```

## Testing

Per `AGENTS.md`'s mandate, in addition to normal unit tests for
`features.py`/`spread_model.py`/`backtest.py`:

- **Golden fixture** (`tests/modeling/_golden_spread_fixture.py` +
  `test_spread_model_golden.py`): a fixed, real historical slice (e.g. 2021
  season `model_table` output, committed as a small parquet or generated
  deterministically from a frozen input) with pinned expected fitted
  coefficients and predictions for a handful of games, tight numeric
  tolerance (not exact equality — floating point). Updating the pinned
  values is a deliberate, reviewed act, never a reflexive fix for a failing
  test — this is the mechanism that catches "the math silently changed."
- **Property tests** (`hypothesis`, `test_property_invariants.py`):
  - `home_cover_probability` is always in `[0, 1]` for arbitrary
    `predicted_result`/`spread_line`/`residual_std` inputs.
  - Backtest P&L reconciles under re-aggregation: summing `units_won`
    across per-week slices of a season's bets equals `run_backtest`'s
    reported season-level `units_won` for the same games.
- No real network calls — tests build small in-memory `pl.DataFrame`
  fixtures for `team_stats`/`schedules` shapes rather than fetching real
  data, consistent with existing testing conventions.

## Dependencies

- `scikit-learn` — new core dependency (user-approved).
- `hypothesis` — new `test`-extra dependency (required by `AGENTS.md`'s
  automation mandate).

## Global constraints carried over

- Polars for all tabular data; pandas never used. numpy only at the
  sklearn call boundary.
- ruff (`select = ["E", "F", "I", "UP"]`, `line-length = 100`), mypy
  strict, `just check` as the gate — unchanged from existing config.
- Conventional commits; PR + green CI required to merge to `master`
  (branch protection now active).
- No comments unless explaining genuine non-obvious "why" (e.g. the
  EPA-double-counting note above is exactly the kind of thing worth one
  comment in the code, not a restated docstring).
