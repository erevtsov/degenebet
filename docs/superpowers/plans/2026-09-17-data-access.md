# DataAccess & Scheduled Fetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `DataAccess.get_team_data(start_date, end_date, as_of_date)` — a point-in-time-correct, network-free read layer stitching `nflreadpy` (historical) and SharpAPI (current) schedule data — backed by a durable local cache that a GitHub Actions scheduled workflow keeps fresh regardless of laptop uptime.

**Architecture:** A merge/upsert cache (`nflverse_cache.py`) makes `nflreadpy` fetches monotonically non-shrinking and persists them to `cache_dir()/merged/`. A scheduled GitHub Actions workflow runs `degenebet fetch` and commits the resulting cache to a dedicated `data` branch; `degenebet sync` pulls that branch's contents into the local cache on demand. `DataAccess` reads only from local files — `NflverseSource` from the merged store, `SharpApiSource` from SharpAPI's existing snapshot cache, transformed (team-name crosswalk, sign conversion, per-game consensus) into a clean schedule-shaped frame — and stitches them with result-based dedup.

**Tech Stack:** Python 3.12, Polars, `nflreadpy`, `httpx`, Typer, `pytest` + `respx` for HTTP mocking, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-17-data-access-design.md` (and `docs/plan.md` / `docs/architecture-notes.md` for the broader system this sits inside).

## Global Constraints

- Polars only, never pandas.
- `DataAccess`, `NflverseSource`, and `SharpApiSource` never call the network or `nflreadpy`/SharpAPI directly — they only read local files that a separate fetch/sync step already wrote.
- No live network calls in the default test run; mock `httpx`/`nflreadpy`/`subprocess`, never real ones.
- `spread_line` sign convention: **positive means home favored.** SharpAPI's raw `line` is the opposite (favorite negative) — confirmed against real API data on 2026-09-17 (Bears home vs. Vikings: home selection line `-2.5`, away `+2.5`).
- Never commit secrets. `SHARPAPI_KEY` reaches the scheduled workflow only via a GitHub Actions repository secret — a manual step for a human with repo admin access, not something this plan's tasks can do.
- `just check` (ruff check, ruff format --check, mypy, pytest) must pass before any task is done.
- Test-first: write the failing test before the implementation for every step below.
- Conventional commits.

## Manual setup required (not part of any task below)

Before Task 5's workflow can succeed in CI, a human must add a `SHARPAPI_KEY` repository secret at `https://github.com/erevtsov/degenebet/settings/secrets/actions`. Flag this to the user when Task 5 is ready; do not attempt to automate it.

---

### Task 1: Rate-limit-aware pagination in `SharpAPIProvider`

Real API traffic on 2026-09-17 confirmed `fetch_raw()`'s pagination loop has no backoff and can exhaust the free tier's 12 req/min budget against itself mid-call (a single spread-market page already reported `has_more: true` past 50 rows). The API returns real `x-ratelimit-remaining`/`x-ratelimit-reset` headers (confirmed: `x-ratelimit-limit: 12`, `x-ratelimit-reset` as a Unix timestamp) — use them to pace requests instead of guessing a fixed delay.

**Files:**
- Modify: `src/degenebet/data/providers/sharpapi.py`
- Test: `tests/data/test_providers_sharpapi.py`

**Interfaces:**
- Produces: `SharpAPIProvider.fetch_raw()` behavior unchanged from the caller's perspective (same return shape); internal pacing only.

- [ ] **Step 1: Write the failing test for rate-limit pacing**

```python
# add to tests/data/test_providers_sharpapi.py

def _headers(remaining: int, reset_at: float) -> dict[str, str]:
    return {"x-ratelimit-limit": "12", "x-ratelimit-remaining": str(remaining),
            "x-ratelimit-reset": str(reset_at)}


@respx.mock
def test_fetch_raw_sleeps_when_rate_limit_nearly_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_at = 1_700_000_100.0
    monkeypatch.setattr("degenebet.data.providers.sharpapi.time.time", lambda: 1_700_000_095.0)
    sleeps: list[float] = []
    monkeypatch.setattr("degenebet.data.providers.sharpapi.time.sleep", sleeps.append)

    route = respx.get(_URL)
    route.side_effect = [
        httpx.Response(
            200,
            json=_payload([_row(event_id="1")], has_more=True, next_cursor="abc"),
            headers=_headers(remaining=1, reset_at=reset_at),
        ),
        httpx.Response(
            200,
            json=_payload([_row(event_id="2")]),
            headers=_headers(remaining=11, reset_at=reset_at),
        ),
    ]

    frame = SharpAPIProvider().fetch_raw()

    assert sorted(frame["event_id"].to_list()) == ["1", "2"]
    assert sleeps == [5.0]  # reset_at (…100) - fake now (…095)


@respx.mock
def test_fetch_raw_does_not_sleep_with_budget_remaining(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("degenebet.data.providers.sharpapi.time.sleep", sleeps.append)

    respx.get(_URL).mock(
        return_value=httpx.Response(
            200, json=_payload([_row()]), headers=_headers(remaining=11, reset_at=9_999_999_999.0)
        )
    )

    SharpAPIProvider().fetch_raw()

    assert sleeps == []


@respx.mock
def test_fetch_raw_tolerates_missing_rate_limit_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("degenebet.data.providers.sharpapi.time.sleep", sleeps.append)

    respx.get(_URL).mock(return_value=httpx.Response(200, json=_payload([_row()])))

    frame = SharpAPIProvider().fetch_raw()

    assert frame.height == 1
    assert sleeps == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_providers_sharpapi.py -k rate_limit -v`
Expected: FAIL (`time` not imported / `_respect_rate_limit` not defined).

- [ ] **Step 3: Implement rate-limit pacing**

In `src/degenebet/data/providers/sharpapi.py`, add `import time` to the imports, add this near the top-level constants:

```python
_RATE_LIMIT_BUFFER = 1  # sleep before the budget hits 0, not after
```

Add this method to `SharpAPIProvider` and call it right after `response.raise_for_status()` inside the `while True:` loop, before `payload = response.json()`:

```python
    @staticmethod
    def _respect_rate_limit(headers: httpx.Headers) -> None:
        """Sleep until the window resets if this response used up the
        rate-limit budget, so the next paginated request doesn't 429.
        A real fetch_raw() call can span more pages than the free tier's
        per-minute budget allows -- confirmed against the live API."""
        remaining = headers.get("x-ratelimit-remaining")
        reset_at = headers.get("x-ratelimit-reset")
        if remaining is None or reset_at is None:
            return
        if int(remaining) > _RATE_LIMIT_BUFFER:
            return
        sleep_for = float(reset_at) - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
```

Add `self._respect_rate_limit(response.headers)` immediately after the existing `response.raise_for_status()` line inside the loop.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_providers_sharpapi.py -v`
Expected: PASS, all tests including the three new ones and all pre-existing ones.

- [ ] **Step 5: Lint, type-check, commit**

Run: `just check`

```bash
git add src/degenebet/data/providers/sharpapi.py tests/data/test_providers_sharpapi.py
git commit -m "fix: pace SharpAPI pagination against real rate-limit headers"
```

---

### Task 2: Merge-cache primitives (`nflverse_cache.py`)

Pure merge/upsert logic plus a persisted-store wrapper with a row-count-drop guard, per the design spec's "Storage semantics" section. A key present in both old and new data takes the new value (so real `nflreadpy` corrections propagate); a key present only in the old cache survives even if a new fetch doesn't have it (so a shrinking upstream response can't silently erase history); the guard refuses to write (raises) if any group's row count would drop.

**Files:**
- Create: `src/degenebet/data/nflverse_cache.py`
- Test: `tests/data/test_nflverse_cache.py`

**Interfaces:**
- Produces: `merge_frames(existing: pl.DataFrame | None, new: pl.DataFrame, key_columns: list[str]) -> pl.DataFrame`, `read_merged(name: str) -> pl.DataFrame | None`, `load_or_merge(new: pl.DataFrame, *, name: str, key_columns: list[str], group_column: str) -> pl.DataFrame`.
- Consumes: `degenebet.config.cache_dir()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_nflverse_cache.py
from __future__ import annotations

import polars as pl
import pytest

from degenebet.data import nflverse_cache


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEGENEBET_CACHE_DIR", str(tmp_path))


def test_merge_frames_key_in_both_takes_new_value() -> None:
    existing = pl.DataFrame({"game_id": ["a"], "result": [10]})
    new = pl.DataFrame({"game_id": ["a"], "result": [14]})

    merged = nflverse_cache.merge_frames(existing, new, key_columns=["game_id"])

    assert merged["result"].to_list() == [14]


def test_merge_frames_key_only_in_existing_survives() -> None:
    existing = pl.DataFrame({"game_id": ["a", "b"], "result": [10, 20]})
    new = pl.DataFrame({"game_id": ["a"], "result": [14]})

    merged = nflverse_cache.merge_frames(existing, new, key_columns=["game_id"])

    assert sorted(merged["game_id"].to_list()) == ["a", "b"]
    assert merged.filter(pl.col("game_id") == "a")["result"][0] == 14
    assert merged.filter(pl.col("game_id") == "b")["result"][0] == 20


def test_merge_frames_key_only_in_new_is_added() -> None:
    existing = pl.DataFrame({"game_id": ["a"], "result": [10]})
    new = pl.DataFrame({"game_id": ["a", "c"], "result": [14, 30]})

    merged = nflverse_cache.merge_frames(existing, new, key_columns=["game_id"])

    assert sorted(merged["game_id"].to_list()) == ["a", "c"]


def test_merge_frames_existing_none_returns_new() -> None:
    new = pl.DataFrame({"game_id": ["a"], "result": [14]})

    merged = nflverse_cache.merge_frames(None, new, key_columns=["game_id"])

    assert merged.equals(new)


def test_read_merged_returns_none_when_absent() -> None:
    assert nflverse_cache.read_merged("schedules") is None


def test_load_or_merge_first_write_has_no_existing_file() -> None:
    new = pl.DataFrame({"game_id": ["a"], "season": [2024], "result": [10]})

    result = nflverse_cache.load_or_merge(
        new, name="schedules", key_columns=["game_id"], group_column="season"
    )

    assert result.equals(new)
    assert nflverse_cache.read_merged("schedules").equals(new)


def test_load_or_merge_persists_and_merges_on_second_call() -> None:
    first = pl.DataFrame({"game_id": ["a"], "season": [2024], "result": [5]})
    second = pl.DataFrame({"game_id": ["a", "b"], "season": [2024, 2024], "result": [10, 20]})

    nflverse_cache.load_or_merge(first, name="schedules", key_columns=["game_id"], group_column="season")
    result = nflverse_cache.load_or_merge(
        second, name="schedules", key_columns=["game_id"], group_column="season"
    )

    assert sorted(result["game_id"].to_list()) == ["a", "b"]


def test_load_or_merge_raises_on_shrinkage() -> None:
    first = pl.DataFrame({"game_id": ["a", "b"], "season": [2023, 2024], "result": [10, 20]})
    nflverse_cache.load_or_merge(
        first, name="schedules", key_columns=["game_id"], group_column="season"
    )

    # "a" is reclassified from season 2023 to 2024 in the new fetch. merge_
    # frames doesn't drop the row (key "a" still exists, just under a new
    # season value), but season 2023's row count would drop from 1 to 0 --
    # exactly the shrinkage the guard exists to catch.
    reclassified = pl.DataFrame({"game_id": ["a"], "season": [2024], "result": [10]})

    with pytest.raises(ValueError, match="shrink"):
        nflverse_cache.load_or_merge(
            reclassified, name="schedules", key_columns=["game_id"], group_column="season"
        )


def test_load_or_merge_allows_growth() -> None:
    first = pl.DataFrame({"game_id": ["a"], "season": [2024], "result": [10]})
    grown = pl.DataFrame({"game_id": ["a", "b"], "season": [2024, 2024], "result": [10, 20]})

    nflverse_cache.load_or_merge(first, name="schedules", key_columns=["game_id"], group_column="season")
    result = nflverse_cache.load_or_merge(
        grown, name="schedules", key_columns=["game_id"], group_column="season"
    )

    assert result.height == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_nflverse_cache.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'degenebet.data.nflverse_cache'`).

- [ ] **Step 3: Implement `nflverse_cache.py`**

```python
"""Merge/upsert cache for nflreadpy-derived tables (schedules, team_stats).

Unlike data/cache.py's append-only SharpAPI snapshots, this cache is a
single persisted, monotonically-growing store per table: each fetch is
merged into what's already there rather than trusting the raw fetch as a
full replacement. See docs/superpowers/specs/2026-09-17-data-access-design.md
("Storage semantics") for why -- a naive overwrite would silently commit
data loss if the upstream API ever starts limiting the historical range it
returns.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from degenebet.config import cache_dir

_MERGED_DIRNAME = "merged"


def merge_frames(
    existing: pl.DataFrame | None, new: pl.DataFrame, key_columns: list[str]
) -> pl.DataFrame:
    """Union `new` into `existing`, keyed by `key_columns`.

    A key present in both takes `new`'s row (so legitimate corrections
    propagate); a key present only in `existing` survives even if `new`
    doesn't have it (so a `new` fetch covering less history than before
    can't silently erase rows already cached).
    """
    if existing is None or existing.height == 0:
        return new
    carried_over = existing.join(new.select(key_columns), on=key_columns, how="anti")
    return pl.concat([new, carried_over.select(new.columns)], how="vertical")


def _merged_path(name: str) -> Path:
    return cache_dir() / _MERGED_DIRNAME / f"{name}.parquet"


def read_merged(name: str) -> pl.DataFrame | None:
    """Return the persisted merged store for `name`, or None if it's never
    been synced."""
    path = _merged_path(name)
    return pl.read_parquet(path) if path.exists() else None


def load_or_merge(
    new: pl.DataFrame, *, name: str, key_columns: list[str], group_column: str
) -> pl.DataFrame:
    """Merge `new` into the persisted store at cache_dir()/merged/{name}.parquet,
    write the result back, and return it.

    Raises ValueError if the merge would drop `group_column`'s row count for
    any group already present in the existing store -- defense in depth on
    top of merge_frames's own no-loss guarantee, to surface a shrinking raw
    fetch rather than let it pass unnoticed.
    """
    existing = read_merged(name)
    merged = merge_frames(existing, new, key_columns)

    if existing is not None and existing.height > 0:
        old_counts = existing.group_by(group_column).len()
        new_counts = merged.group_by(group_column).len()
        comparison = old_counts.join(new_counts, on=group_column, how="left", suffix="_new")
        shrunk = comparison.filter(
            pl.col("len_new").is_null() | (pl.col("len_new") < pl.col("len"))
        )
        if shrunk.height > 0:
            groups = shrunk[group_column].to_list()
            raise ValueError(
                f"Merging '{name}' would shrink {group_column}(s) {groups} -- "
                "refusing to write; investigate the upstream fetch before retrying."
            )

    path = _merged_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(path)
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_nflverse_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `just check`

```bash
git add src/degenebet/data/nflverse_cache.py tests/data/test_nflverse_cache.py
git commit -m "feat: add merge/upsert cache for nflreadpy-derived tables"
```

---

### Task 3: Sync entrypoints + CLI wiring

Wire `nflverse_cache.load_or_merge` into the actual fetch path for schedules and team_stats, and switch the CLI's `fetch schedules`/`fetch team-stats` commands to use it instead of the raw `nflreadpy` loaders, so running `degenebet fetch schedules` durably persists into `cache_dir()/merged/` rather than only reflecting whatever `nflreadpy`'s own opaque cache currently holds.

**Files:**
- Modify: `src/degenebet/data/nflverse_cache.py` (add `sync_schedules`/`sync_team_stats`)
- Modify: `src/degenebet/data/__init__.py`
- Modify: `src/degenebet/cli.py:27-31,41-45`
- Modify: `tests/test_cli.py:14-21,49-56`
- Test: `tests/data/test_nflverse_cache.py` (append)

**Interfaces:**
- Consumes: `nflverse.load_schedules`, `nflverse.load_team_stats` (existing, `src/degenebet/data/nflverse.py:28,38`); `load_or_merge` (Task 2).
- Produces: `nflverse_cache.sync_schedules(seasons: list[int] | None = None) -> pl.DataFrame`, `nflverse_cache.sync_team_stats(seasons: list[int] | None = None) -> pl.DataFrame`; re-exported as `data.sync_schedules`/`data.sync_team_stats`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/data/test_nflverse_cache.py

def test_sync_schedules_fetches_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched = pl.DataFrame({"game_id": ["a"], "season": [2024], "result": [10]})
    monkeypatch.setattr(
        "degenebet.data.nflverse_cache.nflverse.load_schedules", lambda seasons: fetched
    )

    result = nflverse_cache.sync_schedules(seasons=[2024])

    assert result.equals(fetched)
    assert nflverse_cache.read_merged("schedules").equals(fetched)


def test_sync_team_stats_fetches_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched = pl.DataFrame({"game_id": ["a"], "team": ["BUF"], "season": [2024]})
    monkeypatch.setattr(
        "degenebet.data.nflverse_cache.nflverse.load_team_stats", lambda seasons: fetched
    )

    result = nflverse_cache.sync_team_stats(seasons=[2024])

    assert result.equals(fetched)
    assert nflverse_cache.read_merged("team_stats").equals(fetched)
```

Also update the two existing CLI tests that will break once `cli.py` stops calling the raw loaders — replace in `tests/test_cli.py`:

```python
def test_fetch_schedules_reports_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame({"season": [2023, 2024]})
    monkeypatch.setattr("degenebet.cli.data.sync_schedules", lambda seasons: frame)

    result = runner.invoke(app, ["fetch", "schedules"])

    assert result.exit_code == 0
    assert "2 games loaded" in result.stdout


def test_fetch_schedules_parses_seasons_option(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake(seasons: list[int] | None) -> pl.DataFrame:
        captured["seasons"] = seasons
        return pl.DataFrame({"season": [2022]})

    monkeypatch.setattr("degenebet.cli.data.sync_schedules", fake)

    result = runner.invoke(app, ["fetch", "schedules", "--seasons", "2021,2022"])

    assert result.exit_code == 0
    assert captured["seasons"] == [2021, 2022]
```

and:

```python
def test_fetch_team_stats_reports_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame({"season": [2024]})
    monkeypatch.setattr("degenebet.cli.data.sync_team_stats", lambda seasons: frame)

    result = runner.invoke(app, ["fetch", "team-stats"])

    assert result.exit_code == 0
    assert "1 rows loaded" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_nflverse_cache.py tests/test_cli.py -v`
Expected: FAIL (`sync_schedules`/`sync_team_stats` don't exist yet; CLI tests fail because `cli.py` still calls `data.load_schedules`/`data.load_team_stats`).

- [ ] **Step 3: Implement**

Append to `src/degenebet/data/nflverse_cache.py` (add `from degenebet.data import nflverse` to the imports at the top):

```python
def sync_schedules(seasons: list[int] | None = None) -> pl.DataFrame:
    """Fetch schedules from nflreadpy and merge into the persisted store."""
    fresh = nflverse.load_schedules(seasons)
    return load_or_merge(fresh, name="schedules", key_columns=["game_id"], group_column="season")


def sync_team_stats(seasons: list[int] | None = None) -> pl.DataFrame:
    """Fetch team stats from nflreadpy and merge into the persisted store."""
    fresh = nflverse.load_team_stats(seasons)
    return load_or_merge(
        fresh, name="team_stats", key_columns=["game_id", "team"], group_column="season"
    )
```

In `src/degenebet/data/__init__.py`, change the import line and add two exports:

```python
from degenebet.data import cache, nflverse, nflverse_cache
from degenebet.data.providers.sharpapi import SharpAPIProvider

load_schedules = nflverse.load_schedules
load_player_stats = nflverse.load_player_stats
load_team_stats = nflverse.load_team_stats
load_rosters = nflverse.load_rosters
sync_schedules = nflverse_cache.sync_schedules
sync_team_stats = nflverse_cache.sync_team_stats
```

and add `"sync_schedules"`, `"sync_team_stats"` to `__all__`.

In `src/degenebet/cli.py`, change lines 27-31 and 41-45:

```python
@fetch_app.command("schedules")
def fetch_schedules(seasons: str | None = _SEASONS_OPTION) -> None:
    """Load NFL schedules/games (including historical closing lines)."""
    frame = data.sync_schedules(_parse_seasons(seasons))
    typer.echo(f"{frame.height} games loaded")
```

```python
@fetch_app.command("team-stats")
def fetch_team_stats(seasons: str | None = _SEASONS_OPTION) -> None:
    """Load weekly team stats."""
    frame = data.sync_team_stats(_parse_seasons(seasons))
    typer.echo(f"{frame.height} rows loaded")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: PASS (full suite — this step also confirms nothing else broke).

- [ ] **Step 5: Lint, type-check, commit**

Run: `just check`

```bash
git add src/degenebet/data/nflverse_cache.py src/degenebet/data/__init__.py src/degenebet/cli.py tests/data/test_nflverse_cache.py tests/test_cli.py
git commit -m "feat: route fetch schedules/team-stats through the merge cache"
```

---

### Task 4: Local `degenebet sync` command

Pulls the scheduled-fetch workflow's committed data from the `data` branch into the local `cache_dir()`, via `git archive` (no need for a full clone/checkout of that branch into the working tree). Manual/on-demand per the design spec — not itself scheduled.

**Files:**
- Create: `src/degenebet/data/git_sync.py`
- Modify: `src/degenebet/cli.py`
- Test: `tests/data/test_git_sync.py`
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `degenebet.config.cache_dir()`.
- Produces: `git_sync.sync_from_data_branch(*, remote: str = "origin", branch: str = "data") -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_git_sync.py
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from degenebet.data import git_sync


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEGENEBET_CACHE_DIR", str(tmp_path))


def test_sync_from_data_branch_runs_expected_git_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"")

    monkeypatch.setattr("degenebet.data.git_sync.subprocess.run", fake_run)

    git_sync.sync_from_data_branch(remote="origin", branch="data")

    assert calls[0] == ["git", "fetch", "origin", "data"]
    assert calls[1] == ["git", "archive", "--format=tar", "origin/data"]
    assert calls[2] == ["tar", "-x", "-C", str(Path(str(tmp_path)))]


def test_sync_from_data_branch_uses_custom_remote_and_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"")

    monkeypatch.setattr("degenebet.data.git_sync.subprocess.run", fake_run)

    git_sync.sync_from_data_branch(remote="upstream", branch="scheduled-data")

    assert calls[0] == ["git", "fetch", "upstream", "scheduled-data"]
    assert calls[1] == ["git", "archive", "--format=tar", "upstream/scheduled-data"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_git_sync.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `git_sync.py`**

```python
"""Materializes the scheduled-fetch workflow's `data` branch into the local
cache_dir(), via `git archive` -- no working-tree checkout of that branch
needed. Must be run from inside a clone of the degenebet repository."""

from __future__ import annotations

import subprocess

from degenebet.config import cache_dir


def sync_from_data_branch(*, remote: str = "origin", branch: str = "data") -> None:
    """Fetch `branch` from `remote` and extract its contents into
    cache_dir(), overwriting any local copies of the same paths."""
    subprocess.run(["git", "fetch", remote, branch], check=True, capture_output=True)
    dest = cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", f"{remote}/{branch}"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
```

Add the CLI command to `src/degenebet/cli.py` — add `from degenebet.data import git_sync` to the imports, and this command at the end of the file:

```python
@app.command("sync")
def sync_cache(
    remote: str = typer.Option("origin", help="Git remote to fetch from"),
    branch: str = typer.Option("data", help="Branch holding scheduled-fetch snapshots"),
) -> None:
    """Pull the latest scheduled-fetch snapshots from the data branch into the local cache."""
    git_sync.sync_from_data_branch(remote=remote, branch=branch)
    typer.echo(f"Synced local cache from {remote}/{branch}")
```

- [ ] **Step 4: Write and run the CLI test**

```python
# append to tests/test_cli.py

def test_sync_command_calls_git_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake(*, remote: str, branch: str) -> None:
        captured["remote"] = remote
        captured["branch"] = branch

    monkeypatch.setattr("degenebet.cli.git_sync.sync_from_data_branch", fake)

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert captured == {"remote": "origin", "branch": "data"}
    assert "Synced local cache from origin/data" in result.stdout
```

Run: `uv run pytest tests/data/test_git_sync.py tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `just check`

```bash
git add src/degenebet/data/git_sync.py src/degenebet/cli.py tests/data/test_git_sync.py tests/test_cli.py
git commit -m "feat: add degenebet sync command to pull the data branch locally"
```

---

### Task 5: GitHub Actions scheduled fetch workflow

Runs `degenebet fetch` on a daily cron, on GitHub's infra rather than a personal laptop, and commits the resulting cache to a dedicated `data` branch. Not pytest-testable — verified by a manual `workflow_dispatch` run once merged (needs the `SHARPAPI_KEY` secret in place first; see "Manual setup required" above).

**Files:**
- Create: `.github/workflows/scheduled-fetch.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: Scheduled data fetch

on:
  schedule:
    - cron: "0 10 * * *"
  workflow_dispatch: {}

jobs:
  fetch:
    runs-on: ubuntu-latest
    env:
      DEGENEBET_CACHE_DIR: ${{ github.workspace }}/cache
      SHARPAPI_KEY: ${{ secrets.SHARPAPI_KEY }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Checkout or create the data branch
        run: git checkout data 2>/dev/null || git checkout -b data

      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev

      - run: uv run degenebet fetch schedules
      - run: uv run degenebet fetch team-stats
      - run: uv run degenebet fetch odds

      - name: Commit and push fetched data
        run: |
          git config user.name "degenebet-bot"
          git config user.email "degenebet-bot@users.noreply.github.com"
          git add cache/
          git diff --cached --quiet || git commit -m "chore: scheduled data fetch $(date -u +%Y-%m-%dT%H:%M:%SZ)"
          git push origin data
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/scheduled-fetch.yml
git commit -m "feat: add scheduled data fetch workflow"
```

- [ ] **Step 3: Note verification steps for after merge**

Cannot be exercised in this plan's own test suite. Once merged and the `SHARPAPI_KEY` secret is set: trigger manually via `gh workflow run scheduled-fetch.yml` (or the Actions UI's "Run workflow" button), then confirm the `data` branch was created/updated with a `cache/` directory containing `merged/schedules.parquet`, `merged/team_stats.parquet`, and a new `sharpapi/sharpapi_*.parquet` snapshot.

---

### Task 6: Canonical team codes (`teams.py`)

Every vendor `DataAccess` stitches together will have its own column names and identifiers — we already hit this concretely with SharpAPI's full "City Mascot" team names vs. `nflreadpy`'s 2-3 letter codes. Rather than bury a one-off crosswalk inside `SharpApiSource` with no shared reference point, extract the canonical team-code set into its own module now, so it's a single source of truth any future vendor's crosswalk maps onto, and so a vendor crosswalk's completeness/correctness can be validated generically instead of relying on that vendor's own tests to happen to exercise every team.

**Files:**
- Create: `src/degenebet/data/teams.py`
- Test: `tests/data/test_teams.py`

**Interfaces:**
- Produces: `CANONICAL_TEAMS: frozenset[str]` (the 32 `nflreadpy` team abbreviations), `assert_maps_to_canonical_teams(crosswalk: dict[str, str]) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_teams.py
from __future__ import annotations

import pytest

from degenebet.data import teams


def test_canonical_teams_has_32_teams() -> None:
    assert len(teams.CANONICAL_TEAMS) == 32


def test_assert_maps_to_canonical_teams_passes_for_complete_crosswalk() -> None:
    crosswalk = {f"Vendor Name {i}": code for i, code in enumerate(teams.CANONICAL_TEAMS)}

    teams.assert_maps_to_canonical_teams(crosswalk)  # does not raise


def test_assert_maps_to_canonical_teams_raises_on_missing_team() -> None:
    incomplete = {code: code for code in list(teams.CANONICAL_TEAMS)[:-1]}

    with pytest.raises(ValueError, match="missing"):
        teams.assert_maps_to_canonical_teams(incomplete)


def test_assert_maps_to_canonical_teams_raises_on_unknown_code() -> None:
    crosswalk = {code: code for code in teams.CANONICAL_TEAMS}
    crosswalk["Extra Team"] = "XXX"

    with pytest.raises(ValueError, match="unknown"):
        teams.assert_maps_to_canonical_teams(crosswalk)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_teams.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `teams.py`**

```python
"""Canonical NFL team identifiers every vendor's data gets translated into.

nflreadpy's own abbreviations are the canonical set: nflverse is this
project's primary/authoritative source (schedules, team stats), so any
other vendor's team identifiers (e.g. SharpAPI's "Buffalo Bills" full
names) have to line up with these codes for DataAccess to join across
sources at all. See DataSource's docstring in access.py for the full
cross-vendor join contract this is one piece of.
"""

from __future__ import annotations

CANONICAL_TEAMS: frozenset[str] = frozenset(
    {
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
        "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
        "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB",
        "TEN", "WAS",
    }
)


def assert_maps_to_canonical_teams(crosswalk: dict[str, str]) -> None:
    """Raise ValueError if `crosswalk`'s values don't exactly cover
    CANONICAL_TEAMS -- catches a typo'd code, a missing team, or an
    unrecognized code in a vendor's crosswalk generically."""
    mapped = set(crosswalk.values())
    missing = CANONICAL_TEAMS - mapped
    unknown = mapped - CANONICAL_TEAMS
    if missing or unknown:
        raise ValueError(
            f"Crosswalk doesn't map exactly onto CANONICAL_TEAMS -- "
            f"missing: {sorted(missing)}, unknown: {sorted(unknown)}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_teams.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `just check`

```bash
git add src/degenebet/data/teams.py tests/data/test_teams.py
git commit -m "feat: add canonical team-code registry for cross-vendor joins"
```

---

### Task 7: `SharpApiSource` — team crosswalk, sign conversion, per-game consensus

Transforms SharpAPI's raw long-format odds rows (one row per sportsbook per selection per market) into a clean one-row-per-game frame with a `spread_line` in the same sign convention as `nflreadpy`'s (positive = home favored). Confirmed against real API data on 2026-09-17: team names are full "City Mascot" strings ("Buffalo Bills"), and the home selection's `line` is negative when home is favored (opposite of our convention) — `spread_line = -line`. Multiple sportsbooks report different lines for the same game; this takes the **median** main-line home spread across books as the per-game consensus. The team crosswalk targets `teams.CANONICAL_TEAMS` (Task 6) and is validated against it, rather than being a self-contained dict with no shared reference point.

**Files:**
- Create: `src/degenebet/data/access.py`
- Modify: `src/degenebet/data/cache.py` (add `load_all_snapshots`)
- Test: `tests/data/test_access.py`
- Test: `tests/data/test_cache.py` (append)

**Interfaces:**
- Consumes: `cache.load_all_snapshots(source: str) -> pl.DataFrame` (this task, added to `cache.py`); `teams.CANONICAL_TEAMS`, `teams.assert_maps_to_canonical_teams` (Task 6).
- Produces: `DataSource` protocol (`fetch(self, start_date: date, end_date: date) -> pl.DataFrame`), `SharpApiSource` implementing it, returning columns `home_team, away_team, gameday, spread_line, pulled_at` (nflverse team abbreviations, ISO date strings, our sign convention).

**Cross-vendor join contract:** every `DataSource` implementation (this one, `NflverseSource` in Task 8, and any future vendor) must return `home_team`/`away_team` as codes from `teams.CANONICAL_TEAMS`, `gameday` as an ISO 8601 date string (`YYYY-MM-DD`), and `spread_line` (where present) in the positive-means-home-favored convention. These three are the actual join/comparison surface `DataAccess` relies on to stitch sources together — the rest of each source's columns can differ (a historical source and an odds source are fundamentally different shapes, and forcing full column parity between them isn't useful). This goes in `access.py`'s module docstring so it's visible next to the `DataSource` protocol itself, not just in this plan.

- [ ] **Step 1: Write the failing test for `cache.load_all_snapshots`**

```python
# append to tests/data/test_cache.py

def test_load_all_snapshots_concatenates_every_retained_file(tmp_path: object) -> None:
    provider = FakeProvider(_frame(datetime(2024, 1, 1, tzinfo=UTC)))
    cache.load_or_fetch(provider, "testsource", force_refresh=True)
    provider.frame = _frame(datetime(2024, 1, 2, tzinfo=UTC))
    cache.load_or_fetch(provider, "testsource", force_refresh=True)

    result = cache.load_all_snapshots("testsource")

    assert result.height == 4  # two rows per snapshot, two snapshots
    assert result["pulled_at"].n_unique() == 2


def test_load_all_snapshots_returns_empty_frame_when_nothing_cached() -> None:
    result = cache.load_all_snapshots("neversynced")

    assert result.height == 0
```

- [ ] **Step 2: Implement `cache.load_all_snapshots`**

Append to `src/degenebet/data/cache.py`:

```python
def load_all_snapshots(source: str) -> pl.DataFrame:
    """Return every retained snapshot for *source*, concatenated (each row
    keeps its own ``pulled_at``). Empty (zero rows) if nothing's cached yet.
    """
    source_dir = _source_dir(source)
    if not source_dir.exists():
        return pl.DataFrame()
    snapshots = sorted(source_dir.glob(f"{source}_*.parquet"))
    if not snapshots:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(p) for p in snapshots], how="vertical")
```

Run: `uv run pytest tests/data/test_cache.py -v` — expect PASS.

- [ ] **Step 3: Write the failing tests for `SharpApiSource`**

```python
# tests/data/test_access.py
from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from degenebet.data.access import SharpApiSource


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEGENEBET_CACHE_DIR", str(tmp_path))


def _write_snapshot(tmp_path: object, rows: list[dict[str, object]]) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "sharpapi"
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path / "sharpapi_20260917T000000Z.parquet")


def _spread_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": "e1",
        "home_team": "Chicago Bears",
        "away_team": "Minnesota Vikings",
        "market_type": "spread",
        "selection_type": "home",
        "line": -2.5,
        "event_start_time": "2026-09-20T17:00:00Z",
        "pulled_at": datetime(2026, 9, 17, tzinfo=UTC),
    }
    row.update(overrides)
    return row


def test_fetch_converts_sign_home_favored(tmp_path: object) -> None:
    _write_snapshot(tmp_path, [_spread_row(line=-2.5)])

    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    row = result.filter((pl.col("home_team") == "CHI") & (pl.col("away_team") == "MIN"))
    assert row["spread_line"][0] == pytest.approx(2.5)


def test_fetch_converts_sign_away_favored(tmp_path: object) -> None:
    _write_snapshot(tmp_path, [_spread_row(line=3.0)])

    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    row = result.filter((pl.col("home_team") == "CHI") & (pl.col("away_team") == "MIN"))
    assert row["spread_line"][0] == pytest.approx(-3.0)


def test_fetch_takes_median_across_sportsbooks(tmp_path: object) -> None:
    _write_snapshot(
        tmp_path,
        [
            _spread_row(line=-2.5, event_id="e1"),
            _spread_row(line=-3.0, event_id="e1"),
            _spread_row(line=-1.0, event_id="e1"),
        ],
    )

    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    row = result.filter((pl.col("home_team") == "CHI") & (pl.col("away_team") == "MIN"))
    assert row["spread_line"][0] == pytest.approx(2.5)  # median of 2.5/3.0/1.0


def test_fetch_ignores_non_spread_and_away_selection_rows(tmp_path: object) -> None:
    _write_snapshot(
        tmp_path,
        [
            _spread_row(),
            _spread_row(market_type="moneyline", selection_type="home", line=None),
            _spread_row(selection_type="away", line=2.5),
        ],
    )

    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    assert result.height == 1


def test_fetch_filters_to_date_range(tmp_path: object) -> None:
    _write_snapshot(tmp_path, [_spread_row(event_start_time="2026-10-15T17:00:00Z")])

    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    assert result.height == 0


def test_fetch_raises_on_unknown_team_name(tmp_path: object) -> None:
    _write_snapshot(tmp_path, [_spread_row(home_team="Springfield Isotopes")])

    with pytest.raises(Exception):  # noqa: B017 -- polars' replace_strict error type
        SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))


def test_fetch_returns_empty_frame_when_nothing_cached() -> None:
    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    assert result.height == 0


def test_sharpapi_crosswalk_covers_every_canonical_team() -> None:
    from degenebet.data import access, teams

    teams.assert_maps_to_canonical_teams(access._SHARPAPI_TEAM_CROSSWALK)  # does not raise
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_access.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 5: Implement `access.py` (`DataSource` protocol + `SharpApiSource` only — `NflverseSource`/`DataAccess` are Task 8)**

```python
"""DataAccess: point-in-time stitching of historical (nflreadpy) and current
(SharpAPI) schedule data. See
docs/superpowers/specs/2026-09-17-data-access-design.md.

Cross-vendor join contract: every DataSource implementation must return
`home_team`/`away_team` as codes from `teams.CANONICAL_TEAMS`, `gameday` as
an ISO 8601 date string (YYYY-MM-DD), and `spread_line` (where present) with
positive meaning home favored. These three are the actual join/comparison
surface DataAccess relies on to stitch sources together -- the rest of each
source's columns can differ; a historical source and an odds source are
fundamentally different shapes, and full column parity between them isn't
useful or required.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

import polars as pl

from degenebet.data import cache, teams

# SharpAPI's real full-name format, confirmed against live data 2026-09-17
# (e.g. "Buffalo Bills", "Chicago Bears") -- not the abbreviated-city guess
# in the provider's own test fixtures, which was never verified live.
# Validated against teams.CANONICAL_TEAMS below, not just by this module's
# own tests happening to exercise every team.
_SHARPAPI_TEAM_CROSSWALK: dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

teams.assert_maps_to_canonical_teams(_SHARPAPI_TEAM_CROSSWALK)


class DataSource(Protocol):
    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame: ...


class SharpApiSource:
    """Current-odds source: one row per (home_team, away_team, gameday) with
    `spread_line` in nflreadpy's sign convention (positive = home favored),
    the median main-line home spread across sportsbooks and retained
    snapshots as of the latest snapshot in cache."""

    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        raw = cache.load_all_snapshots("sharpapi")
        if raw.height == 0:
            return pl.DataFrame(
                schema={
                    "home_team": pl.Utf8,
                    "away_team": pl.Utf8,
                    "gameday": pl.Utf8,
                    "spread_line": pl.Float64,
                    "pulled_at": pl.Datetime(time_zone="UTC"),
                }
            )

        home_spreads = raw.filter(
            (pl.col("market_type") == "spread") & (pl.col("selection_type") == "home")
        ).with_columns(
            pl.col("home_team").replace_strict(_SHARPAPI_TEAM_CROSSWALK),
            pl.col("away_team").replace_strict(_SHARPAPI_TEAM_CROSSWALK),
            (-pl.col("line")).alias("spread_line"),
            pl.col("event_start_time").str.slice(0, 10).alias("gameday"),
        )

        in_range = home_spreads.filter(
            (pl.col("gameday") >= start_date.isoformat())
            & (pl.col("gameday") <= end_date.isoformat())
        )

        return in_range.group_by(["home_team", "away_team", "gameday"]).agg(
            pl.col("spread_line").median(), pl.col("pulled_at").max()
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_access.py tests/data/test_cache.py -v`
Expected: PASS.

- [ ] **Step 7: Lint, type-check, commit**

Run: `just check`

```bash
git add src/degenebet/data/access.py src/degenebet/data/cache.py tests/data/test_access.py tests/data/test_cache.py
git commit -m "feat: add SharpApiSource with team crosswalk and sign conversion"
```

---

### Task 8: `NflverseSource` + `DataAccess.get_team_data`

Completes `access.py`: `NflverseSource` reads the merged historical store (Task 2/3), and `DataAccess.get_team_data` stitches it with `SharpApiSource` (Task 7) using result-based dedup — `historical` wins for any `game_id` with a settled `result`; `current` wins for an unplayed game whenever SharpAPI covers it (even if `historical` already has an early line for it); `historical`'s own line is the fallback when `current` doesn't cover that game. Includes the as-of clamp-and-warn behavior.

**Files:**
- Modify: `src/degenebet/data/access.py`
- Test: `tests/data/test_access.py` (append)

**Interfaces:**
- Consumes: `nflverse_cache.read_merged` (Task 2), `SharpApiSource`/`DataSource` (Task 7).
- Produces: `NflverseSource` (implements `DataSource`), `DataAccess.__init__(self, historical: DataSource, current: DataSource)`, `DataAccess.get_team_data(self, start_date: date, end_date: date, as_of_date: date) -> pl.DataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/data/test_access.py

import warnings

from degenebet.data import nflverse_cache
from degenebet.data.access import DataAccess, NflverseSource


def _schedule_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": "2026_02_MIN_CHI",
        "season": 2026,
        "week": 2,
        "gameday": "2026-09-20",
        "home_team": "CHI",
        "away_team": "MIN",
        "result": None,
        "spread_line": None,
    }
    row.update(overrides)
    return row


class _FakeSource:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame

    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        return self.frame


def test_nflverse_source_reads_merged_store_filtered_to_range() -> None:
    nflverse_cache.load_or_merge(
        pl.DataFrame(
            [_schedule_row(gameday="2026-09-06"), _schedule_row(gameday="2026-10-06")]
        ),
        name="schedules",
        key_columns=["game_id"],
        group_column="season",
    )

    result = NflverseSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    assert result.height == 1
    assert result["gameday"][0] == "2026-09-06"


def test_get_team_data_prefers_current_for_unplayed_game_even_with_stale_historical_line() -> None:
    historical = _FakeSource(
        pl.DataFrame([_schedule_row(result=None, spread_line=1.0)])
    )
    current = _FakeSource(
        pl.DataFrame(
            {
                "home_team": ["CHI"],
                "away_team": ["MIN"],
                "gameday": ["2026-09-20"],
                "spread_line": [2.5],
                "pulled_at": [datetime(2026, 9, 17, tzinfo=UTC)],
            }
        )
    )

    result = DataAccess(historical, current).get_team_data(
        date(2026, 9, 1), date(2026, 9, 30), as_of_date=date(2026, 9, 17)
    )

    assert result["spread_line"][0] == pytest.approx(2.5)


def test_get_team_data_historical_wins_for_settled_result() -> None:
    historical = _FakeSource(
        pl.DataFrame([_schedule_row(result=7, spread_line=1.0)])
    )
    current = _FakeSource(
        pl.DataFrame(
            {
                "home_team": ["CHI"],
                "away_team": ["MIN"],
                "gameday": ["2026-09-20"],
                "spread_line": [2.5],
                "pulled_at": [datetime(2026, 9, 17, tzinfo=UTC)],
            }
        )
    )

    result = DataAccess(historical, current).get_team_data(
        date(2026, 9, 1), date(2026, 9, 30), as_of_date=date(2026, 9, 17)
    )

    assert result["spread_line"][0] == pytest.approx(1.0)


def test_get_team_data_falls_back_to_historical_when_current_missing_game() -> None:
    historical = _FakeSource(
        pl.DataFrame([_schedule_row(result=None, spread_line=1.0)])
    )
    current = _FakeSource(
        pl.DataFrame(
            schema={
                "home_team": pl.Utf8,
                "away_team": pl.Utf8,
                "gameday": pl.Utf8,
                "spread_line": pl.Float64,
                "pulled_at": pl.Datetime(time_zone="UTC"),
            }
        )
    )

    result = DataAccess(historical, current).get_team_data(
        date(2026, 9, 1), date(2026, 9, 30), as_of_date=date(2026, 9, 17)
    )

    assert result["spread_line"][0] == pytest.approx(1.0)


def test_get_team_data_warns_and_clamps_as_of_before_earliest_history() -> None:
    historical = _FakeSource(pl.DataFrame([_schedule_row(gameday="2026-09-06")]))
    current = _FakeSource(
        pl.DataFrame(
            schema={
                "home_team": pl.Utf8,
                "away_team": pl.Utf8,
                "gameday": pl.Utf8,
                "spread_line": pl.Float64,
                "pulled_at": pl.Datetime(time_zone="UTC"),
            }
        )
    )

    with pytest.warns(UserWarning, match="predates earliest cached history"):
        DataAccess(historical, current).get_team_data(
            date(2026, 9, 1), date(2026, 9, 30), as_of_date=date(2020, 1, 1)
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_access.py -v`
Expected: FAIL (`ImportError: cannot import name 'DataAccess'`).

- [ ] **Step 3: Implement**

Add `import warnings` to `access.py`'s imports, and add `from degenebet.data import nflverse_cache` alongside the existing `from degenebet.data import cache`. Append to `src/degenebet/data/access.py`:

```python
class NflverseSource:
    """Historical schedule source: reads the persisted merged schedules
    store (nflverse_cache.py), never the network."""

    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        merged = nflverse_cache.read_merged("schedules")
        if merged is None:
            raise RuntimeError(
                "No schedules have ever been synced -- run `degenebet fetch schedules` "
                "or `degenebet sync` first."
            )
        return merged.filter(
            (pl.col("gameday") >= start_date.isoformat())
            & (pl.col("gameday") <= end_date.isoformat())
        )


class DataAccess:
    """Stitches historical and current schedule sources into one
    point-in-time-correct view. See
    docs/superpowers/specs/2026-09-17-data-access-design.md."""

    def __init__(self, historical: DataSource, current: DataSource) -> None:
        self.historical = historical
        self.current = current

    def get_team_data(self, start_date: date, end_date: date, as_of_date: date) -> pl.DataFrame:
        historical = self.historical.fetch(start_date, end_date)
        current = self.current.fetch(start_date, end_date)

        if historical.height > 0:
            earliest = historical["gameday"].min()
            if as_of_date.isoformat() < earliest:
                warnings.warn(
                    f"as_of_date {as_of_date} predates earliest cached history "
                    f"{earliest}; clamping to {earliest}.",
                    stacklevel=2,
                )
                as_of_date = date.fromisoformat(earliest)

        current_asof = (
            current.filter(pl.col("pulled_at").dt.date() <= as_of_date)
            if current.height > 0
            else current
        )

        unresolved = historical.filter(pl.col("result").is_null()).select(
            "game_id", "home_team", "away_team", "gameday"
        )
        current_for_unresolved = current_asof.join(
            unresolved, on=["home_team", "away_team", "gameday"], how="inner"
        ).select("game_id", pl.col("spread_line").alias("current_spread_line"))

        return (
            historical.join(current_for_unresolved, on="game_id", how="left")
            .with_columns(pl.coalesce(["current_spread_line", "spread_line"]).alias("spread_line"))
            .drop("current_spread_line")
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_access.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, run the full suite, commit**

Run: `just check`

```bash
git add src/degenebet/data/access.py tests/data/test_access.py
git commit -m "feat: add DataAccess.get_team_data with result-based dedup"
```

---

## After all tasks: broader verification

Once every task's own tests pass and `just check` is green branch-wide:
- Manually confirm the `SharpApiSource` sign conversion one more time by running `SharpApiSource().fetch(...)` against a freshly synced real SharpAPI snapshot (via `degenebet fetch odds` then a quick Python check), not just the unit tests' fixtures — the fixtures encode today's verified real-data findings, but a second look after implementation catches any transcription slip.
- After merging, add the `SHARPAPI_KEY` GitHub Actions secret and manually trigger `scheduled-fetch.yml` once via `workflow_dispatch` to confirm the `data` branch gets created and populated end-to-end.
