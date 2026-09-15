# Data Foundation — Design Spec

Date: 2026-09-14
Status: Approved for implementation planning

## Context

`degenebet` is a quantitative, data-driven NFL betting system. The full
system will eventually include: data ingestion, predictive modeling
(spread/totals/moneyline), backtesting, bankroll/staking sizing, and bet
tracking. This spec covers only the **first sub-project: the data
foundation** — pulling and caching historical NFL stats/schedules/closing
lines plus current sportsbook odds, exposed behind a clean loading API and
a CLI. Modeling, backtesting, staking, and bet tracking are separate,
future specs.

## Goals

- Free, no-cost-required historical data: team/player stats, schedules,
  and historical closing lines (spread, total, moneyline) for all
  available NFL seasons.
- Free-tier current odds (spreads, totals, moneylines) from SharpAPI for
  today's lines, usable to compare against model output once modeling
  exists.
- On-demand CLI — no scheduler, no background jobs. The user runs commands
  when they want fresh data.
- Local parquet storage — no database server.

## Non-goals

- Modeling, backtesting, staking, bet tracking, or bet execution — future
  sub-projects.
- Player props — SharpAPI supports them but this spec's SharpAPI client
  covers only game-level markets (moneyline, spread, total); props can be
  added to `providers/sharpapi.py` later without restructuring.
- Automated scheduling — deferred until an operating-mode change is
  explicitly requested.

## Data sources

- **`nflreadpy`** — actively maintained Python port of nflverse's
  `nflreadr` (the predecessor `nfl_data_py` is archived/deprecated).
  Returns Polars DataFrames. Has its own built-in filesystem cache
  (configurable TTL), so this project does not need to build caching for
  it. Provides `load_schedules()` (games plus historical closing lines —
  `spread_line`, `total_line`, `away_moneyline`/`home_moneyline`,
  `away_spread_odds`/`home_spread_odds`, `over_odds`/`under_odds` — back to
  1999), `load_player_stats()`, `load_team_stats()`, `load_rosters()`. No
  API key required.
- **SharpAPI** — current NFL odds across 30+ sportsbooks. Free tier: 12
  req/min, odds from 2 sportsbooks (DraftKings, FanDuel), no credit card.
  Auth via `X-API-Key` header. The user already holds a free-tier key.
  Endpoint: `GET /api/v1/odds?league=nfl&market=moneyline,spread,total`.
  Response is `{"data": [...], "pagination": {...}, "updated_at": ...}`;
  each `data` row is one (sportsbook, event, market, selection) with
  `event_id`, `home_team`, `away_team`, `market_type`, `selection`,
  `selection_type` (`home`/`away`/`over`/`under`), `odds_american`,
  `odds_decimal`, `odds_probability`, `line` (null for moneyline, the
  spread/total number otherwise), `event_start_time`, `timestamp`.
  Paginated via `pagination.has_more` / `pagination.next_cursor` — one
  week of NFL moneyline+spread+total rows can exceed the 200-row page
  max, so the provider must follow pagination to completion.

## Architecture

Borrows the `data/` module pattern from the sibling `waypoint` project
(`/Users/erevtsov/dev/waypoint/src/waypoint/data/`): a vendor-agnostic
public API backed by a `Provider` protocol per vendor and a local parquet
cache. Adapted for two vendors with different caching needs:

```
src/degenebet/
  __init__.py
  config.py                  # env-based settings
  cli.py                     # typer app
  data/
    __init__.py               # public loading API
    cache.py                   # parquet snapshot cache (used by sharpapi provider only)
    nflverse.py                 # thin wrappers over nflreadpy.load_*
    providers/
      __init__.py
      base.py                    # Provider protocol (fetch_raw)
      sharpapi.py                  # SharpAPIProvider
tests/
  data/
    test_cache.py
    test_nflverse.py
    test_providers_sharpapi.py
  test_cli.py
```

### `data/nflverse.py`

Thin wrapper functions, one per table, delegating to `nflreadpy`:

```python
def load_schedules(seasons: list[int] | None = None) -> pl.DataFrame: ...
def load_player_stats(seasons: list[int] | None = None) -> pl.DataFrame: ...
def load_team_stats(seasons: list[int] | None = None) -> pl.DataFrame: ...
def load_rosters(seasons: list[int] | None = None) -> pl.DataFrame: ...
```

`seasons=None` means "all available seasons" (1999–present for schedules,
player stats, and rosters). Note this is `degenebet`'s own wrapper
convention, not `nflreadpy`'s raw signature — `nflreadpy`'s actual
sentinel for "all seasons" is `seasons=True` (its default), so each
wrapper translates `None` → `True` before calling the underlying
`load_*` function. No caching logic here; `nflreadpy`'s built-in
filesystem cache (configured once in `config.py`, see below) handles
re-fetch avoidance. These wrappers exist to give `degenebet` a stable,
typed interface independent of the underlying package's API, and to be
the single place that would change if the source library changes again.

### `data/providers/sharpapi.py`

```python
class SharpAPIProvider:
    def fetch_raw(self, league: str = "nfl") -> pl.DataFrame: ...
```

Calls `GET /api/v1/odds?league=nfl&market=moneyline,spread,total` with the
`X-API-Key` header set from `config.sharpapi_key`, via `httpx`. Follows
`pagination.next_cursor` until `pagination.has_more` is false, concatenating
all pages. Flattens the `data` rows into a Polars DataFrame with one row per
(event, sportsbook, market, selection) — `event_id`, `home_team`,
`away_team`, `market_type`, `selection`, `selection_type`, `odds_american`,
`odds_decimal`, `odds_probability`, `line`, `event_start_time`,
`sportsbook`, plus a `pulled_at` timestamp added at parse time (distinct
from the API's own `timestamp` field, which reflects when SharpAPI last
updated that price). Raises a clear error on HTTP 401 (bad/missing key) or
429 (rate limited) rather than swallowing failures.

### `data/cache.py`

Snapshot cache, not date-range gap-fill (unlike waypoint's price cache —
current odds aren't a historical series to fill gaps in, they're a
point-in-time pull). Cache key: `{cache_root}/sharpapi/odds_{pulled_at
iso timestamp}.parquet`. Public function:

```python
def load_or_fetch(
    provider: Provider,
    *,
    max_age: timedelta = timedelta(minutes=15),
    force_refresh: bool = False,
) -> pl.DataFrame: ...
```

Behavior: if the most recent cached snapshot is younger than `max_age`
and `force_refresh` is False, return it without a network call; otherwise
call `provider.fetch_raw()`, write a new timestamped snapshot, and return
it. `max_age` defaults to 15 minutes — short enough that re-running the
CLI within the same working session doesn't burn the 12 req/min budget on
accidental repeats, long enough to be clearly "current" for game-day use.

### `data/providers/base.py`

```python
class Provider(Protocol):
    def fetch_raw(self) -> pl.DataFrame: ...
```

### `config.py`

- `SHARPAPI_KEY` — required env var (via `.env`, loaded with
  `python-dotenv`, `override=False` so real shell env wins).
- `DEGENEBET_CACHE_DIR` — optional env var; defaults to
  `~/.degenebet/cache`. Used both to configure `nflreadpy`'s cache
  directory and as the root for `data/cache.py`'s SharpAPI snapshots.

### `cli.py`

`typer` app with one command group, `fetch`:

- `degenebet fetch schedules [--seasons 2020,2021,...]`
- `degenebet fetch player-stats [--seasons ...]`
- `degenebet fetch team-stats [--seasons ...]`
- `degenebet fetch rosters [--seasons ...]`
- `degenebet fetch odds [--force-refresh]`

Each command calls the corresponding `data/` function and prints a short
summary (row count, season/date range covered, or "N sportsbooks, pulled
at TIMESTAMP" for odds) to stdout. No output files beyond the cache —
this is a way to warm the cache and sanity-check what's available, not a
report generator.

## Error handling

- Network/HTTP failures from SharpAPI (auth, rate limit, timeout)
  surface as clear exceptions with the HTTP status and a short message —
  no silent fallback to stale data unless the cache is valid per
  `max_age`.
- `nflreadpy` failures (e.g. a requested season with no data yet)
  propagate as-is; no special handling needed since `nflreadpy` already
  raises informative errors.
- Missing `SHARPAPI_KEY` raises at CLI-command time (only when a command
  that needs it runs), not at import time.

## Testing

- `nflreadpy` calls are mocked in tests (`monkeypatch` or a small fake
  returning a fixed Polars DataFrame) — no real network calls in the
  default test run.
- `httpx` calls to SharpAPI are mocked with `respx`.
- `data/cache.py`'s snapshot logic (fresh-enough vs. stale vs.
  force-refresh) is tested against real small Polars DataFrames and a
  temp cache directory (`tmp_path` fixture) — no mocking of the parquet
  read/write itself, per the "never mock what you can fake with real
  in-memory data" rule.
- CLI commands tested via `typer.testing.CliRunner`, with the underlying
  `data/` functions mocked (CLI tests check argument wiring and output
  formatting, not data-fetching correctness — that's covered by the
  `data/` module's own tests).

## Tooling & conventions (borrowed from waypoint)

- `uv` + `src/` layout, hatchling build backend, Python 3.12+.
- Dependencies: `nflreadpy`, `httpx`, `polars`, `python-dotenv`, `typer`,
  `pyarrow`. Optional-dependency groups: `test` (pytest, pytest-cov),
  `lint` (ruff), `typecheck` (mypy), `dev` (umbrella of the above).
- ruff: `line-length = 100`, `target-version = "py312"`,
  `select = ["E", "F", "I", "UP"]`, `known-first-party = ["degenebet"]`.
- mypy: `strict = true`, `warn_return_any = true`, `mypy_path = "src"`.
- pytest: `testpaths = ["tests"]`, `--cov=degenebet --cov-report=term-missing -v`.
- Coverage: `branch = true`.
- **Difference from waypoint**: `degenebet` is an application (has a
  CLI, not a library for import), so `uv.lock` **is** committed (waypoint
  explicitly excludes it as a library).
- Use polars for all tabular data; never pandas.
- Explicit over implicit; no magic numbers; no commented-out code; no
  bare `except`; decompose functions over ~30 lines.
- Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`,
  `test:`); lint + typecheck must pass before any task is called done.
- Test-first; never mock what can be faked with real in-memory data.
- Agentic behavior: confirm before deleting/overwriting files; present a
  plan and wait before changes spanning more than 3 files; ask before
  adding new dependencies; present tradeoffs instead of silently picking
  when genuinely uncertain between approaches.

## Open items for future specs

- Modeling sub-project will define what "seasons=None" should trim to
  for training (spec explicitly defers this: "load all possible data, we
  can trim it at modeling stage").
- Player props and alternate lines from SharpAPI are available but unused
  until a props model exists.
- Scheduling/automation is out of scope until the operating mode changes
  from on-demand.
