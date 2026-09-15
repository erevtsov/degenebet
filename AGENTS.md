# degenebet

## Project Overview
- Quantitative, data-driven NFL betting system built on free data sources
- Application with a CLI (`degenebet fetch ...`) — not a library
- Data foundation only so far: historical stats/schedules/closing lines
  (`nflreadpy`) and current odds (SharpAPI); modeling, backtesting,
  staking, and bet tracking are future sub-projects

## Commands
- `uv sync --extra dev` — install all dev dependencies
- `just check` — the authoritative verification gate: ruff lint, ruff format check, mypy, pytest (identical to what CI runs on every PR)
- `uv run pytest` — run tests with coverage
- `uv run ruff check src/ tests/` — lint
- `uv run mypy` — type-check
- `uv run degenebet fetch <table>` — warm the local cache

## Automation & Verification
- `just check` is the single CI-authoritative gate — `.github/workflows/ci.yml` runs this exact command, nothing more, nothing less. Locally and in CI it always means: `ruff check src/ tests/`, `ruff format --check src/ tests/`, `mypy`, `pytest`, in that order, fail-fast.
- `master` is branch-protected: the `check` CI status must be green and every change must go through a pull request, with no exception for the repo owner — there is no direct-push path around the gate.
- No scheduled maintenance automation yet — see `docs/superpowers/specs/2026-09-15-agentic-automation-design.md` for the deferred design and why.
- No golden regression fixtures or property-based tests yet, and this is not optional to skip: **the modeling sub-project must add them before or alongside its first predictive model.** Pin numeric output for each model/analytic surface against a deterministic fixture (one fixture family per surface, tight numeric tolerance not exact equality, updates are a deliberate reviewed act never a reflexive fix for a failing test), and add `hypothesis` property tests for invariants that would be expensive to get silently wrong (e.g. predicted probabilities lie in `[0, 1]` and sum to 1 across a market's outcomes; backtest P&L reconciles under re-aggregation over a date range).

## Architecture
- `src/degenebet/config.py` — env-based settings: `cache_dir()`, `sharpapi_key()`
- `src/degenebet/cli.py` — typer app, `fetch` command group
- `src/degenebet/data/nflverse.py` — thin wrappers over `nflreadpy.load_*`;
  relies on `nflreadpy`'s own filesystem cache, not `data/cache.py`
- `src/degenebet/data/cache.py` — snapshot parquet cache, used only by the
  SharpAPI provider (current odds are a point-in-time pull, not a
  historical series to gap-fill)
- `src/degenebet/data/providers/base.py` — `Provider` protocol (`fetch_raw`)
- `src/degenebet/data/providers/sharpapi.py` — `SharpAPIProvider`
- `src/degenebet/data/__init__.py` — public loading API
- `tests/` mirrors `src/degenebet/` structure

## Conventions
- Use polars for all tabular data; never pandas
- `uv.lock` **is** committed (application, not library)
- Explicit over implicit; no magic numbers; no commented-out code; no bare
  `except`; decompose functions over ~30 lines
- Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`)
- Use `importlib.metadata.version("degenebet")` for `__version__` — never
  hardcode it

## Testing Rules
- Golden regression fixtures and `hypothesis` property tests are required for the modeling sub-project — see "Automation & Verification" below for what that means concretely
- Test-first for anything beyond a trivial change
- Never mock what can be faked with real in-memory data — network calls
  are mocked (`nflreadpy`, `httpx`), but parsing/transform logic and the
  parquet cache are tested against real small Polars DataFrames + `tmp_path`
- No real network calls in the default test run
- Run lint and typecheck before declaring any task done

## Agentic Behavior Rules
- Before deleting or overwriting any file, confirm with the user
- Before making changes across more than 3 files, present a plan and wait
- When uncertain between two approaches, present both with tradeoffs
- Never install new dependencies without asking first
- Never modify this file during a task unless explicitly asked

## Never Do
- Never use pandas — use polars instead
- Never commit secrets (`.env`) — only `.env.example`
