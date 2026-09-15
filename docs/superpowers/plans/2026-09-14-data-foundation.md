# Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `degenebet` package's data foundation — free historical NFL stats/schedules/closing lines via `nflreadpy`, current odds via SharpAPI, and a CLI to warm the cache on demand.

**Architecture:** A `data/` sub-package with thin `nflreadpy` wrappers (which rely on nflreadpy's own filesystem cache) plus a small `Provider`-protocol + snapshot-parquet-cache pair used only by the SharpAPI client (current odds are a point-in-time pull, not a historical series). A `typer` CLI exposes one `fetch` command per table/source.

**Tech Stack:** Python 3.12+, `uv`, `nflreadpy`, `httpx`, `polars`, `typer`, `python-dotenv`, `pytest`/`respx` for tests, `ruff`, `mypy --strict`.

**Spec:** `docs/superpowers/specs/2026-09-14-data-foundation-design.md`

## Global Constraints

- Python `>=3.12`, `uv` + `src/` layout, hatchling build backend.
- Use polars for all tabular data; never pandas.
- ruff: `line-length = 100`, `target-version = "py312"`, `select = ["E", "F", "I", "UP"]`, `known-first-party = ["degenebet"]`.
- mypy: `strict = true`, `warn_return_any = true`, `mypy_path = "src"`.
- pytest: `testpaths = ["tests"]`, `addopts = ["--cov=degenebet", "--cov-report=term-missing", "-v"]`; coverage `branch = true`.
- `uv.lock` **is** committed (degenebet is an application, not a library — opposite of the sibling `waypoint` project's rule).
- Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`).
- Explicit over implicit; no magic numbers; no commented-out code; no bare `except`; decompose functions over ~30 lines.
- Test-first; never mock what can be faked with real in-memory data — network calls (`nflreadpy`, `httpx`) are mocked, but parsing/transform logic and the parquet cache itself are tested against real small Polars DataFrames and `tmp_path`.
- No real network calls in the default test run.
- SharpAPI auth: `X-API-Key` header. Endpoint: `GET https://api.sharpapi.io/api/v1/odds?league=nfl&market=moneyline,spread,total`, paginated via `pagination.has_more` / `pagination.next_cursor`.

---

## File Structure

```
degenebet/
  pyproject.toml
  uv.lock                          # committed (application)
  README.md
  .python-version
  .gitignore
  .env.example
  CLAUDE.md
  src/degenebet/
    __init__.py                     # __version__ via importlib.metadata
    config.py                        # cache_dir(), sharpapi_key()
    cli.py                            # typer app: `degenebet fetch ...`
    data/
      __init__.py                     # public API: load_schedules/load_player_stats/load_team_stats/load_rosters/load_odds
      cache.py                         # snapshot parquet cache (load_or_fetch)
      nflverse.py                       # thin wrappers over nflreadpy.load_*
      providers/
        __init__.py
        base.py                          # Provider protocol
        sharpapi.py                       # SharpAPIProvider
  tests/
    test_smoke.py
    test_config.py
    test_cli.py
    data/
      __init__.py
      test_cache.py
      test_nflverse.py
      test_public_api.py
      test_providers_sharpapi.py
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.python-version`
- Create: `.env.example`
- Create: `README.md`
- Create: `CLAUDE.md`
- Create: `src/degenebet/__init__.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: `degenebet.__version__: str` (package importable as `degenebet`).

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "degenebet"
version = "0.1.0"
description = "A quantitative, data-driven NFL betting system"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "nflreadpy>=0.1",
    "polars>=1.0",
    "pyarrow>=14.0",
    "python-dotenv>=1.0",
    "typer>=0.12",
]

[project.optional-dependencies]
test = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "respx>=0.21",
]
lint = [
    "ruff>=0.4",
]
typecheck = [
    "mypy>=1.10",
]
dev = [
    "degenebet[test,lint,typecheck]",
]

[tool.hatch.build.targets.wheel]
packages = ["src/degenebet"]

# ---------------------------------------------------------------------------
# Ruff
# ---------------------------------------------------------------------------
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.ruff.lint.isort]
known-first-party = ["degenebet"]

# ---------------------------------------------------------------------------
# Mypy
# ---------------------------------------------------------------------------
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
mypy_path = "src"
packages = ["degenebet"]

[[tool.mypy.overrides]]
module = ["nflreadpy"]
ignore_missing_imports = true

# ---------------------------------------------------------------------------
# Pytest
# ---------------------------------------------------------------------------
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "--cov=degenebet",
    "--cov-report=term-missing",
    "-v",
]

[tool.coverage.run]
source = ["src/degenebet"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

- [ ] **Step 2: Create `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
build/
dist/
*.egg-info/
*.egg

# Virtual environments
.venv/
venv/
env/

# Test / coverage
htmlcov/
.coverage
.coverage.*
.cache
.pytest_cache/
coverage.xml

# Type checking
.mypy_cache/
.dmypy.json

# Ruff
.ruff_cache/

# IDEs
.idea/
.vscode/
*.swp
*~

# macOS
.DS_Store

# Environment variables — never commit secrets
.env
```

Note: `uv.lock` is deliberately **not** listed here — it must be committed (see Global Constraints).

- [ ] **Step 3: Create `.python-version`**

```
3.12
```

- [ ] **Step 4: Create `.env.example`**

```
# Copy to .env and fill in your key.
SHARPAPI_KEY=

# Optional: override the default cache directory (~/.degenebet/cache)
# DEGENEBET_CACHE_DIR=
```

- [ ] **Step 5: Create `README.md`**

```markdown
# degenebet

A quantitative, data-driven NFL betting system built on free data sources.

## Setup

```bash
uv sync --extra dev
cp .env.example .env   # fill in SHARPAPI_KEY
```

## Usage

```bash
uv run degenebet fetch schedules
uv run degenebet fetch player-stats
uv run degenebet fetch team-stats
uv run degenebet fetch rosters
uv run degenebet fetch odds
```

## Development

```bash
uv run pytest
uv run ruff check src/ tests/
uv run mypy
```
```

- [ ] **Step 6: Create `CLAUDE.md`**

```markdown
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
```

- [ ] **Step 7: Create `src/degenebet/__init__.py` (empty package marker first)**

```python
```

(empty file — content added in Step 9 after the test is written)

- [ ] **Step 8: Write the failing smoke test**

`tests/test_smoke.py`:

```python
from __future__ import annotations

import degenebet


def test_package_has_version() -> None:
    assert degenebet.__version__ == "0.1.0"
```

- [ ] **Step 9: Run `uv sync --extra dev`, then run the test to verify it fails**

Run: `uv sync --extra dev`
Expected: creates `.venv`, installs all deps, succeeds.

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL — `AttributeError: module 'degenebet' has no attribute '__version__'`

- [ ] **Step 10: Implement `__version__` in `src/degenebet/__init__.py`**

```python
from __future__ import annotations

from importlib.metadata import version

__version__ = version("degenebet")
```

- [ ] **Step 11: Run the test to verify it passes**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 12: Verify lint and type-check are clean**

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mypy`
Expected: `Success: no issues found`

- [ ] **Step 13: Commit**

```bash
git add pyproject.toml .gitignore .python-version .env.example README.md CLAUDE.md src/degenebet/__init__.py tests/test_smoke.py uv.lock
git commit -m "chore: project scaffolding"
```

---

### Task 2: Config module

**Files:**
- Create: `src/degenebet/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (reads `SHARPAPI_KEY`, `DEGENEBET_CACHE_DIR` from environment/`.env`).
- Produces: `cache_dir() -> Path`, `sharpapi_key() -> str` (raises `RuntimeError` if unset) — used by `data/cache.py` (Task 3) and `data/providers/sharpapi.py` (Task 4).

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from degenebet import config


def test_cache_dir_defaults_to_home_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEGENEBET_CACHE_DIR", raising=False)
    assert config.cache_dir() == Path.home() / ".degenebet" / "cache"


def test_cache_dir_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEGENEBET_CACHE_DIR", "/tmp/custom-cache")
    assert config.cache_dir() == Path("/tmp/custom-cache")


def test_sharpapi_key_returns_value_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARPAPI_KEY", "abc123")
    assert config.sharpapi_key() == "abc123"


def test_sharpapi_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHARPAPI_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SHARPAPI_KEY"):
        config.sharpapi_key()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degenebet.config'`

- [ ] **Step 3: Implement `src/degenebet/config.py`**

```python
"""Environment-based settings: cache directory, SharpAPI key."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)  # shell env vars take precedence over .env

_DEFAULT_CACHE_ROOT = Path.home() / ".degenebet" / "cache"


def cache_dir() -> Path:
    """Return the local cache root, overridable via ``DEGENEBET_CACHE_DIR``."""
    env = os.environ.get("DEGENEBET_CACHE_DIR")
    return Path(env) if env else _DEFAULT_CACHE_ROOT


def sharpapi_key() -> str:
    """Return the SharpAPI key from the environment.

    Raises
    ------
    RuntimeError
        If ``SHARPAPI_KEY`` is not set.
    """
    key = os.environ.get("SHARPAPI_KEY")
    if not key:
        raise RuntimeError(
            "SHARPAPI_KEY is not set. Add it to .env or export it in your shell."
        )
    return key
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/ tests/ && uv run mypy`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add src/degenebet/config.py tests/test_config.py
git commit -m "feat: add config module for cache dir and SharpAPI key"
```

---

### Task 3: Provider protocol and snapshot cache

**Files:**
- Create: `src/degenebet/data/__init__.py` (empty for now — expanded in Task 6)
- Create: `src/degenebet/data/providers/__init__.py` (empty)
- Create: `src/degenebet/data/providers/base.py`
- Create: `src/degenebet/data/cache.py`
- Test: `tests/data/__init__.py` (empty)
- Test: `tests/data/test_cache.py`

**Interfaces:**
- Consumes: `degenebet.config.cache_dir() -> Path` (Task 2).
- Produces: `Provider` protocol (`fetch_raw() -> pl.DataFrame`, output must include a `pulled_at` column of dtype `pl.Datetime`); `load_or_fetch(provider: Provider, source: str, *, max_age: timedelta = timedelta(minutes=15), force_refresh: bool = False) -> pl.DataFrame` — used by `data/providers/sharpapi.py` (Task 4) and `data/__init__.py` (Task 6).

- [ ] **Step 1: Create empty package markers**

`src/degenebet/data/__init__.py`: empty file.
`src/degenebet/data/providers/__init__.py`: empty file.
`tests/data/__init__.py`: empty file.

- [ ] **Step 2: Create `src/degenebet/data/providers/base.py`**

```python
"""Provider protocol — the interface data/cache.py's load_or_fetch expects."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class Provider(Protocol):
    """Fetches a fresh, full-replacement snapshot from a vendor.

    ``fetch_raw()`` takes no arguments beyond what the implementation was
    constructed with (e.g. league, API key). The returned DataFrame must
    include a ``pulled_at`` column (``pl.Datetime``, UTC) recording when
    the snapshot was taken — ``data/cache.py`` uses it to judge freshness.
    """

    def fetch_raw(self) -> pl.DataFrame:
        """Return a fresh DataFrame snapshot from the vendor."""
        ...
```

- [ ] **Step 3: Write the failing tests for `data/cache.py`**

`tests/data/test_cache.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from degenebet.data import cache


class FakeProvider:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls = 0

    def fetch_raw(self) -> pl.DataFrame:
        self.calls += 1
        return self.frame


def _frame(pulled_at: datetime) -> pl.DataFrame:
    return pl.DataFrame({"value": [1, 2], "pulled_at": [pulled_at, pulled_at]})


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEGENEBET_CACHE_DIR", str(tmp_path))


def test_load_or_fetch_calls_provider_when_no_cache() -> None:
    provider = FakeProvider(_frame(datetime.now(timezone.utc)))
    result = cache.load_or_fetch(provider, "testsource")
    assert provider.calls == 1
    assert result.equals(provider.frame)


def test_load_or_fetch_reuses_fresh_cache() -> None:
    provider = FakeProvider(_frame(datetime.now(timezone.utc)))
    cache.load_or_fetch(provider, "testsource")
    cache.load_or_fetch(provider, "testsource")
    assert provider.calls == 1


def test_load_or_fetch_refetches_when_stale() -> None:
    old = datetime.now(timezone.utc) - timedelta(hours=1)
    provider = FakeProvider(_frame(old))
    cache.load_or_fetch(provider, "testsource", max_age=timedelta(minutes=15))
    provider.frame = _frame(datetime.now(timezone.utc))
    cache.load_or_fetch(provider, "testsource", max_age=timedelta(minutes=15))
    assert provider.calls == 2


def test_load_or_fetch_force_refresh_bypasses_fresh_cache() -> None:
    provider = FakeProvider(_frame(datetime.now(timezone.utc)))
    cache.load_or_fetch(provider, "testsource")
    cache.load_or_fetch(provider, "testsource", force_refresh=True)
    assert provider.calls == 2


def test_load_or_fetch_writes_a_parquet_snapshot(tmp_path: object) -> None:
    provider = FakeProvider(_frame(datetime.now(timezone.utc)))
    cache.load_or_fetch(provider, "testsource")
    snapshots = list((Path(str(tmp_path)) / "testsource").glob("testsource_*.parquet"))  # type: ignore[name-defined]
    assert len(snapshots) == 1
```

Note: the last test needs `from pathlib import Path` added to the imports above — include it.

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degenebet.data.cache'`

- [ ] **Step 5: Implement `src/degenebet/data/cache.py`**

```python
"""Snapshot parquet cache for point-in-time data sources (e.g. current odds).

Unlike a historical time-series cache, this does not gap-fill a date range —
each write is a full-replacement snapshot, timestamped by when it was pulled.
Freshness is judged from the snapshot's own ``pulled_at`` column (not file
mtime), so it stays correct even if files are copied or touched externally.

Cache layout: ``{cache_dir()}/{source}/{source}_{pulled_at}.parquet``
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from degenebet.config import cache_dir
from degenebet.data.providers.base import Provider

_TS_FORMAT = "%Y%m%dT%H%M%SZ"


def _source_dir(source: str) -> Path:
    return cache_dir() / source


def _snapshot_path(source: str, pulled_at: datetime) -> Path:
    ts = pulled_at.astimezone(timezone.utc).strftime(_TS_FORMAT)
    return _source_dir(source) / f"{source}_{ts}.parquet"


def _latest_snapshot(source: str) -> Path | None:
    source_dir = _source_dir(source)
    if not source_dir.exists():
        return None
    snapshots = sorted(source_dir.glob(f"{source}_*.parquet"))
    return snapshots[-1] if snapshots else None


def load_or_fetch(
    provider: Provider,
    source: str,
    *,
    max_age: timedelta = timedelta(minutes=15),
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Return a cached snapshot for *source*, fetching a fresh one if needed.

    If the most recent cached snapshot's ``pulled_at`` is within *max_age*
    of now and *force_refresh* is False, returns it without calling
    ``provider.fetch_raw()``. Otherwise fetches, writes a new timestamped
    snapshot, and returns the fresh data.

    Parameters
    ----------
    provider:
        Object implementing ``Provider``.
    source:
        Cache namespace, e.g. ``"sharpapi"``.
    max_age:
        How old a cached snapshot may be before it's considered stale.
    force_refresh:
        If True, always fetch fresh data regardless of cache age.
    """
    latest_path = None if force_refresh else _latest_snapshot(source)

    if latest_path is not None:
        cached = pl.read_parquet(latest_path)
        pulled_at: datetime = cached["pulled_at"].max()  # type: ignore[assignment]
        if datetime.now(timezone.utc) - pulled_at <= max_age:
            return cached

    fresh = provider.fetch_raw()
    pulled_at = fresh["pulled_at"].max()  # type: ignore[assignment]
    path = _snapshot_path(source, pulled_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_parquet(path)
    return fresh
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_cache.py -v`
Expected: PASS (5 tests)

- [ ] **Step 7: Lint and type-check**

Run: `uv run ruff check src/ tests/ && uv run mypy`
Expected: both clean

- [ ] **Step 8: Commit**

```bash
git add src/degenebet/data/__init__.py src/degenebet/data/providers/__init__.py src/degenebet/data/providers/base.py src/degenebet/data/cache.py tests/data/__init__.py tests/data/test_cache.py
git commit -m "feat: add Provider protocol and snapshot parquet cache"
```

---

### Task 4: SharpAPI provider

**Files:**
- Create: `src/degenebet/data/providers/sharpapi.py`
- Test: `tests/data/test_providers_sharpapi.py`

**Interfaces:**
- Consumes: `degenebet.config.sharpapi_key() -> str` (Task 2), `Provider` protocol shape (Task 3).
- Produces: `SharpAPIProvider(league: str = "nfl")` with `.fetch_raw() -> pl.DataFrame` (columns: `event_id`, `home_team`, `away_team`, `market_type`, `selection`, `selection_type`, `odds_american`, `odds_decimal`, `odds_probability`, `line`, `event_start_time`, `sportsbook`, `pulled_at`) — used by `data/__init__.py` (Task 6).

- [ ] **Step 1: Write the failing tests**

`tests/data/test_providers_sharpapi.py`:

```python
from __future__ import annotations

import httpx
import pytest
import respx

from degenebet.data.providers.sharpapi import SharpAPIProvider

_URL = "https://api.sharpapi.io/api/v1/odds"


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "draftkings_1_moneyline_PHI",
        "sportsbook": "draftkings",
        "event_id": "1",
        "sport": "football",
        "league": "nfl",
        "home_team": "PHI Eagles",
        "away_team": "DAL Cowboys",
        "market_type": "moneyline",
        "selection": "PHI Eagles",
        "selection_type": "home",
        "odds_american": -150,
        "odds_decimal": 1.667,
        "odds_probability": 0.6,
        "line": None,
        "is_main_line": True,
        "is_alternate_line": False,
        "event_start_time": "2026-09-21T17:00:00Z",
        "timestamp": "2026-09-14T02:10:24.125Z",
        "is_live": False,
    }
    row.update(overrides)
    return row


def _payload(rows: list[dict[str, object]], *, has_more: bool = False, next_cursor: str | None = None) -> dict[str, object]:
    return {
        "data": rows,
        "pagination": {
            "limit": 50,
            "offset": 0,
            "count": len(rows),
            "has_more": has_more,
            "next_cursor": next_cursor,
        },
        "updated_at": "2026-09-14T02:10:37.846Z",
    }


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARPAPI_KEY", "test-key")


@respx.mock
def test_fetch_raw_parses_single_page() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(200, json=_payload([_row()])))

    frame = SharpAPIProvider().fetch_raw()

    assert frame.height == 1
    assert frame["home_team"][0] == "PHI Eagles"
    assert frame["odds_american"][0] == -150
    assert "pulled_at" in frame.columns


@respx.mock
def test_fetch_raw_follows_pagination() -> None:
    route = respx.get(_URL)
    route.side_effect = [
        httpx.Response(200, json=_payload([_row(event_id="1")], has_more=True, next_cursor="abc")),
        httpx.Response(200, json=_payload([_row(event_id="2")])),
    ]

    frame = SharpAPIProvider().fetch_raw()

    assert sorted(frame["event_id"].to_list()) == ["1", "2"]


@respx.mock
def test_fetch_raw_drops_non_main_lines() -> None:
    rows = [_row(), _row(id="alt", is_main_line=False, line=-3.5)]
    respx.get(_URL).mock(return_value=httpx.Response(200, json=_payload(rows)))

    frame = SharpAPIProvider().fetch_raw()

    assert frame.height == 1


@respx.mock
def test_fetch_raw_raises_on_unauthorized() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(401, json={"error": "invalid key"}))

    with pytest.raises(RuntimeError, match="401"):
        SharpAPIProvider().fetch_raw()


@respx.mock
def test_fetch_raw_raises_on_rate_limit() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(429, json={"error": "rate limited"}))

    with pytest.raises(RuntimeError, match="429"):
        SharpAPIProvider().fetch_raw()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_providers_sharpapi.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degenebet.data.providers.sharpapi'`

- [ ] **Step 3: Implement `src/degenebet/data/providers/sharpapi.py`**

```python
"""SharpAPI provider — current NFL odds (moneyline, spread, total)."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import polars as pl

from degenebet.config import sharpapi_key

_BASE_URL = "https://api.sharpapi.io/api/v1/odds"
_MARKETS = "moneyline,spread,total"
_COLUMNS = [
    "event_id",
    "home_team",
    "away_team",
    "market_type",
    "selection",
    "selection_type",
    "odds_american",
    "odds_decimal",
    "odds_probability",
    "line",
    "event_start_time",
    "sportsbook",
]


class SharpAPIProvider:
    """Fetches current odds from SharpAPI, following pagination to completion."""

    def __init__(self, league: str = "nfl") -> None:
        self.league = league

    def fetch_raw(self) -> pl.DataFrame:
        """Return every current main-line odds row for ``self.league``.

        Adds a ``pulled_at`` column (UTC, one instant for the whole call)
        recording when this fetch ran.
        """
        pulled_at = datetime.now(timezone.utc)
        rows: list[dict[str, object]] = []
        cursor: str | None = None

        with httpx.Client(headers={"X-API-Key": sharpapi_key()}, timeout=30.0) as client:
            while True:
                params: dict[str, str] = {"league": self.league, "market": _MARKETS}
                if cursor is not None:
                    params["cursor"] = cursor

                response = client.get(_BASE_URL, params=params)
                if response.status_code == 401:
                    raise RuntimeError(
                        "SharpAPI rejected the API key (401). Check SHARPAPI_KEY."
                    )
                if response.status_code == 429:
                    raise RuntimeError(
                        "SharpAPI rate limit exceeded (429). "
                        "Free tier allows 12 requests/minute."
                    )
                response.raise_for_status()

                payload = response.json()
                rows.extend(row for row in payload["data"] if row["is_main_line"])

                pagination = payload["pagination"]
                if not pagination["has_more"]:
                    break
                cursor = pagination["next_cursor"]

        if rows:
            frame = pl.DataFrame(rows).select(_COLUMNS)
        else:
            frame = pl.DataFrame(schema={c: pl.Utf8 for c in _COLUMNS})

        return frame.with_columns(pl.lit(pulled_at).alias("pulled_at"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_providers_sharpapi.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/ tests/ && uv run mypy`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add src/degenebet/data/providers/sharpapi.py tests/data/test_providers_sharpapi.py
git commit -m "feat: add SharpAPI provider for current NFL odds"
```

---

### Task 5: nflverse wrappers

**Files:**
- Create: `src/degenebet/data/nflverse.py`
- Test: `tests/data/test_nflverse.py`

**Interfaces:**
- Consumes: `nflreadpy.load_schedules/load_player_stats/load_team_stats/load_rosters`.
- Produces: `load_schedules(seasons: list[int] | None = None) -> pl.DataFrame`, `load_player_stats(...)`, `load_team_stats(...)`, `load_rosters(...)` — used by `data/__init__.py` (Task 6).

- [ ] **Step 1: Verify nflreadpy's actual function signatures**

Run:

```bash
uv run python -c "
import inspect
import nflreadpy
for name in ['load_schedules', 'load_player_stats', 'load_team_stats', 'load_rosters']:
    fn = getattr(nflreadpy, name)
    print(name, inspect.signature(fn))
"
```

Confirm each accepts a `seasons` parameter whose default/sentinel for "all seasons" is `True` (per `load_schedules(seasons: int | list[int] | bool | None = True)`). If any function's signature differs (different parameter name, no "all seasons" sentinel, etc.), stop and adjust Step 3 below to match reality before writing code against an assumption.

- [ ] **Step 2: Write the failing tests**

`tests/data/test_nflverse.py`:

```python
from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest

from degenebet.data import nflverse

_WRAPPERS: list[tuple[Callable[..., pl.DataFrame], str]] = [
    (nflverse.load_schedules, "load_schedules"),
    (nflverse.load_player_stats, "load_player_stats"),
    (nflverse.load_team_stats, "load_team_stats"),
    (nflverse.load_rosters, "load_rosters"),
]


def _fake_frame() -> pl.DataFrame:
    return pl.DataFrame({"season": [2024]})


@pytest.mark.parametrize(("wrapper", "nflreadpy_fn"), _WRAPPERS)
def test_wrapper_translates_none_to_true(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: Callable[..., pl.DataFrame],
    nflreadpy_fn: str,
) -> None:
    calls: list[object] = []

    def fake(seasons: object) -> pl.DataFrame:
        calls.append(seasons)
        return _fake_frame()

    monkeypatch.setattr(f"nflreadpy.{nflreadpy_fn}", fake)

    result = wrapper()

    assert calls == [True]
    assert result.equals(_fake_frame())


@pytest.mark.parametrize(("wrapper", "nflreadpy_fn"), _WRAPPERS)
def test_wrapper_passes_explicit_seasons(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: Callable[..., pl.DataFrame],
    nflreadpy_fn: str,
) -> None:
    calls: list[object] = []

    def fake(seasons: object) -> pl.DataFrame:
        calls.append(seasons)
        return _fake_frame()

    monkeypatch.setattr(f"nflreadpy.{nflreadpy_fn}", fake)

    wrapper([2022, 2023])

    assert calls == [[2022, 2023]]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_nflverse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degenebet.data.nflverse'`

- [ ] **Step 4: Implement `src/degenebet/data/nflverse.py`**

```python
"""Thin wrappers over nflreadpy's load_* functions.

Kept separate from direct nflreadpy calls elsewhere in degenebet so there is
one stable, typed interface that would only need to change here if the
underlying library's API changes. ``seasons=None`` means "all available
seasons" — nflreadpy's own sentinel for that is ``seasons=True``, so each
wrapper translates ``None`` to ``True``.
"""

from __future__ import annotations

import nflreadpy
import polars as pl


def _seasons_arg(seasons: list[int] | None) -> list[int] | bool:
    return True if seasons is None else seasons


def load_schedules(seasons: list[int] | None = None) -> pl.DataFrame:
    """Return NFL schedules/games, including historical closing lines."""
    return nflreadpy.load_schedules(seasons=_seasons_arg(seasons))


def load_player_stats(seasons: list[int] | None = None) -> pl.DataFrame:
    """Return weekly player stats."""
    return nflreadpy.load_player_stats(seasons=_seasons_arg(seasons))


def load_team_stats(seasons: list[int] | None = None) -> pl.DataFrame:
    """Return weekly team stats."""
    return nflreadpy.load_team_stats(seasons=_seasons_arg(seasons))


def load_rosters(seasons: list[int] | None = None) -> pl.DataFrame:
    """Return seasonal roster data."""
    return nflreadpy.load_rosters(seasons=_seasons_arg(seasons))
```

If Step 1 found a differently-shaped signature for any function, adjust that function's wrapper (and its `_seasons_arg` usage) to match before proceeding.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_nflverse.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check src/ tests/ && uv run mypy`
Expected: both clean

- [ ] **Step 7: Commit**

```bash
git add src/degenebet/data/nflverse.py tests/data/test_nflverse.py
git commit -m "feat: add nflreadpy wrapper functions"
```

---

### Task 6: Public data API

**Files:**
- Modify: `src/degenebet/data/__init__.py` (currently empty from Task 3)
- Test: `tests/data/test_public_api.py`

**Interfaces:**
- Consumes: `nflverse.load_schedules/load_player_stats/load_team_stats/load_rosters` (Task 5), `cache.load_or_fetch` (Task 3), `SharpAPIProvider` (Task 4).
- Produces: `degenebet.data.load_schedules`, `load_player_stats`, `load_team_stats`, `load_rosters` (re-exports), `load_odds(*, max_age: timedelta = timedelta(minutes=15), force_refresh: bool = False) -> pl.DataFrame` — used by `cli.py` (Task 7).

- [ ] **Step 1: Write the failing tests**

`tests/data/test_public_api.py`:

```python
from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

import degenebet.data as data


def test_load_schedules_is_nflverse_load_schedules() -> None:
    assert data.load_schedules is data.nflverse.load_schedules


def test_load_player_stats_is_nflverse_load_player_stats() -> None:
    assert data.load_player_stats is data.nflverse.load_player_stats


def test_load_team_stats_is_nflverse_load_team_stats() -> None:
    assert data.load_team_stats is data.nflverse.load_team_stats


def test_load_rosters_is_nflverse_load_rosters() -> None:
    assert data.load_rosters is data.nflverse.load_rosters


def test_load_odds_delegates_to_cache_with_sharpapi_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_load_or_fetch(
        provider: object,
        source: str,
        *,
        max_age: timedelta,
        force_refresh: bool,
    ) -> pl.DataFrame:
        captured["source"] = source
        captured["max_age"] = max_age
        captured["force_refresh"] = force_refresh
        return pl.DataFrame({"a": [1]})

    monkeypatch.setattr(data.cache, "load_or_fetch", fake_load_or_fetch)

    result = data.load_odds(force_refresh=True)

    assert captured["source"] == "sharpapi"
    assert captured["force_refresh"] is True
    assert result.height == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_public_api.py -v`
Expected: FAIL — `AttributeError: module 'degenebet.data' has no attribute 'load_schedules'`

- [ ] **Step 3: Implement `src/degenebet/data/__init__.py`**

```python
"""Public data-loading API for degenebet.

Historical NFL data (schedules, stats, rosters) is unaffected by the cache
in this package — ``nflreadpy`` handles its own filesystem cache. Only the
current-odds pull goes through ``data/cache.py``, since it's a point-in-time
snapshot rather than a stable historical series.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from degenebet.data import cache, nflverse
from degenebet.data.providers.sharpapi import SharpAPIProvider

load_schedules = nflverse.load_schedules
load_player_stats = nflverse.load_player_stats
load_team_stats = nflverse.load_team_stats
load_rosters = nflverse.load_rosters


def load_odds(
    *,
    max_age: timedelta = timedelta(minutes=15),
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Return current NFL odds (moneyline, spread, total), using the cache.

    See ``degenebet.data.cache.load_or_fetch`` for freshness semantics.
    """
    return cache.load_or_fetch(
        SharpAPIProvider(), "sharpapi", max_age=max_age, force_refresh=force_refresh
    )


__all__ = [
    "load_schedules",
    "load_player_stats",
    "load_team_stats",
    "load_rosters",
    "load_odds",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_public_api.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/ tests/ && uv run mypy`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add src/degenebet/data/__init__.py tests/data/test_public_api.py
git commit -m "feat: expose public data loading API"
```

---

### Task 7: CLI

**Files:**
- Create: `src/degenebet/cli.py`
- Modify: `pyproject.toml:14-23` (add `[project.scripts]` entry point)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `degenebet.data.load_schedules/load_player_stats/load_team_stats/load_rosters/load_odds` (Task 6).
- Produces: `degenebet.cli.app` (typer app), console script `degenebet`.

- [ ] **Step 1: Add the console script entry point**

In `pyproject.toml`, add after the `[project]` `dependencies` block (before `[project.optional-dependencies]`):

```toml
[project.scripts]
degenebet = "degenebet.cli:app"
```

- [ ] **Step 2: Write the failing tests**

`tests/test_cli.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from typer.testing import CliRunner

from degenebet.cli import app

runner = CliRunner()


def test_fetch_schedules_reports_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame({"season": [2023, 2024]})
    monkeypatch.setattr("degenebet.cli.data.load_schedules", lambda seasons: frame)

    result = runner.invoke(app, ["fetch", "schedules"])

    assert result.exit_code == 0
    assert "2 games loaded" in result.stdout


def test_fetch_schedules_parses_seasons_option(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake(seasons: list[int] | None) -> pl.DataFrame:
        captured["seasons"] = seasons
        return pl.DataFrame({"season": [2022]})

    monkeypatch.setattr("degenebet.cli.data.load_schedules", fake)

    runner.invoke(app, ["fetch", "schedules", "--seasons", "2021,2022"])

    assert captured["seasons"] == [2021, 2022]


def test_fetch_player_stats_reports_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame({"season": [2024]})
    monkeypatch.setattr("degenebet.cli.data.load_player_stats", lambda seasons: frame)

    result = runner.invoke(app, ["fetch", "player-stats"])

    assert result.exit_code == 0
    assert "1 rows loaded" in result.stdout


def test_fetch_team_stats_reports_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame({"season": [2024]})
    monkeypatch.setattr("degenebet.cli.data.load_team_stats", lambda seasons: frame)

    result = runner.invoke(app, ["fetch", "team-stats"])

    assert result.exit_code == 0
    assert "1 rows loaded" in result.stdout


def test_fetch_rosters_reports_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame({"season": [2024]})
    monkeypatch.setattr("degenebet.cli.data.load_rosters", lambda seasons: frame)

    result = runner.invoke(app, ["fetch", "rosters"])

    assert result.exit_code == 0
    assert "1 rows loaded" in result.stdout


def test_fetch_odds_reports_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame(
        {
            "sportsbook": ["draftkings", "fanduel"],
            "pulled_at": [datetime.now(timezone.utc), datetime.now(timezone.utc)],
        }
    )
    monkeypatch.setattr("degenebet.cli.data.load_odds", lambda force_refresh: frame)

    result = runner.invoke(app, ["fetch", "odds"])

    assert result.exit_code == 0
    assert "2 odds rows from 2 sportsbook" in result.stdout


def test_fetch_odds_force_refresh_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake(force_refresh: bool) -> pl.DataFrame:
        captured["force_refresh"] = force_refresh
        return pl.DataFrame({"sportsbook": [], "pulled_at": []})

    monkeypatch.setattr("degenebet.cli.data.load_odds", fake)

    runner.invoke(app, ["fetch", "odds", "--force-refresh"])

    assert captured["force_refresh"] is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'degenebet.cli'`

- [ ] **Step 4: Implement `src/degenebet/cli.py`**

```python
"""degenebet CLI — on-demand commands to warm the local data cache."""

from __future__ import annotations

import typer

from degenebet import data

app = typer.Typer(no_args_is_help=True)
fetch_app = typer.Typer(no_args_is_help=True)
app.add_typer(fetch_app, name="fetch")


def _parse_seasons(seasons: str | None) -> list[int] | None:
    if seasons is None:
        return None
    return [int(s) for s in seasons.split(",")]


_SEASONS_OPTION = typer.Option(None, help="Comma-separated seasons, e.g. 2022,2023")


@fetch_app.command("schedules")
def fetch_schedules(seasons: str | None = _SEASONS_OPTION) -> None:
    """Load NFL schedules/games (including historical closing lines)."""
    frame = data.load_schedules(_parse_seasons(seasons))
    typer.echo(f"{frame.height} games loaded")


@fetch_app.command("player-stats")
def fetch_player_stats(seasons: str | None = _SEASONS_OPTION) -> None:
    """Load weekly player stats."""
    frame = data.load_player_stats(_parse_seasons(seasons))
    typer.echo(f"{frame.height} rows loaded")


@fetch_app.command("team-stats")
def fetch_team_stats(seasons: str | None = _SEASONS_OPTION) -> None:
    """Load weekly team stats."""
    frame = data.load_team_stats(_parse_seasons(seasons))
    typer.echo(f"{frame.height} rows loaded")


@fetch_app.command("rosters")
def fetch_rosters(seasons: str | None = _SEASONS_OPTION) -> None:
    """Load seasonal roster data."""
    frame = data.load_rosters(_parse_seasons(seasons))
    typer.echo(f"{frame.height} rows loaded")


@fetch_app.command("odds")
def fetch_odds(
    force_refresh: bool = typer.Option(False, "--force-refresh"),
) -> None:
    """Load current NFL odds (moneyline, spread, total) from SharpAPI."""
    frame = data.load_odds(force_refresh=force_refresh)
    books = frame["sportsbook"].n_unique()
    pulled_at = frame["pulled_at"].max()
    typer.echo(f"{frame.height} odds rows from {books} sportsbook(s), pulled at {pulled_at}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Reinstall so the `degenebet` console script is registered, then smoke-test it**

Run: `uv sync --extra dev`
Run: `uv run degenebet --help`
Expected: shows the `fetch` command group.

- [ ] **Step 7: Run the full test suite, lint, and type-check**

Run: `uv run pytest`
Expected: all tests pass, coverage report printed.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mypy`
Expected: `Success: no issues found`

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/degenebet/cli.py tests/test_cli.py uv.lock
git commit -m "feat: add CLI for fetching schedules, stats, rosters, and odds"
```

---

## Self-Review Notes

- **Spec coverage:** project scaffolding/tooling (Task 1), config (Task 2), `Provider` protocol + snapshot cache (Task 3), SharpAPI client with pagination/auth/rate-limit handling (Task 4), nflverse wrappers with `None`→`True` season translation (Task 5), public `data/` API including `load_odds` (Task 6), CLI (Task 7) — every spec section maps to a task.
- **Placeholder scan:** no TBD/TODO; every step has runnable code or an exact command.
- **Type consistency:** `load_or_fetch(provider, source, *, max_age, force_refresh)` signature is identical across Task 3's implementation, Task 4/6's callers, and Task 6's test fake. `Provider.fetch_raw() -> pl.DataFrame` matches `SharpAPIProvider.fetch_raw()`. `_parse_seasons` return type (`list[int] | None`) matches `load_schedules` etc.'s parameter type across Tasks 5–7.
- **One open assumption, flagged in-task:** Task 5 Step 1 has the implementer verify nflreadpy's real function signatures before trusting this plan's assumption that all four `load_*` functions share `load_schedules`'s `seasons: ... | bool = True` convention — confirmed for `load_schedules` from nflreadpy's own docs, inferred (not confirmed) for the other three.
