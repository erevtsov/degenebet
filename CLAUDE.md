# degenebet

## Project Overview
- Quantitative, data-driven NFL betting system built on free data sources
- Application with a CLI (`degenebet fetch ...`) — not a library
- Data foundation only so far: historical stats/schedules/closing lines
  (`nflreadpy`) and current odds (SharpAPI); modeling, backtesting,
  staking, and bet tracking are future sub-projects

## Commands
- `uv sync --extra dev` — install all dev dependencies
- `uv run pytest` — run tests with coverage
- `uv run ruff check src/ tests/` — lint
- `uv run mypy` — type-check
- `uv run degenebet fetch <table>` — warm the local cache

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
