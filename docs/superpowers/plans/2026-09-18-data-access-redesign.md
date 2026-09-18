# DataAccess Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `DataAccess` into the single, coherent place to fetch NFL data at whatever granularity a caller needs — `get_game_data` (game-indexed) and `get_team_data` (team-indexed), both `Gameweek`-indexed instead of raw date ranges, with team-code normalization owned once at the merge-cache write boundary.

**Architecture:** A new `Gameweek(season, week)` type replaces `start_date`/`end_date` for "which games" everywhere except `SharpApiSource`, which stays date-based and fully decoupled from `historical`/`Gameweek` — `DataAccess` derives the date bounds it needs to hand `SharpApiSource` from `historical`'s own fetched rows, no separate calendar table. `nflverse_cache.sync_schedules`/`sync_team_stats` normalize team codes and add a `gameweek` column at write time, so every reader — `DataAccess` or a direct `read_merged()` call — sees consistent data. `DataAccess` splits into two public methods sharing one private stitching core, plus a generic `team_data` widening parameter on `get_game_data` that works for any team-indexed input.

**Tech Stack:** Python 3.12, Polars, `pytest` + real in-memory DataFrames.

**Spec:** `docs/superpowers/specs/2026-09-18-data-access-redesign.md` (and `docs/superpowers/specs/2026-09-17-data-access-design.md` for the original `DataAccess` this extends).

## Global Constraints

- Polars only, never pandas.
- `SharpApiSource` never changes in this plan — no `Gameweek`, no `historical` awareness, no signature change. If a task's approach requires touching `SharpApiSource`, that's a sign the design's being violated; stop and reconsider.
- `spread_line` sign convention unchanged: **positive means home favored**. `team_spread_line`/`team_margin` (new) follow the same convention from *that team's* perspective (positive = that team favored / that team won by that margin).
- The `gameweek` column's formula (`season * 100 + week`) and `Gameweek.as_int()`'s formula must be identical — both exist so a materialized column and a Python-side boundary value always compare correctly against each other. Any task touching either must keep them in lockstep.
- No live network calls in the default test run; no real network in any test.
- `just check` (ruff check, ruff format --check, mypy, pytest) must pass before any task is done.
- Test-first: write the failing test before the implementation for every step below.
- Conventional commits.

## Manual setup required (not part of any task below)

**After this plan's PR merges, before relying on it for real data:** the already-persisted `team_stats` store (local dev caches and the shared `data` git branch) has uppercase team codes keyed on `(game_id, team)`. Since `team` is part of that table's merge key, a fresh sync under the new lowercase-normalizing code would *duplicate* every row rather than replace it (old `"BUF"` and new `"buf"` are different keys). `schedules` self-heals cleanly (its merge key is `game_id` alone), but for both tables: delete the local `cache_dir()/merged/schedules.parquet` and `cache_dir()/merged/team_stats.parquet`, then re-run `degenebet fetch schedules` and `degenebet fetch team-stats` (or trigger `scheduled-fetch.yml` once via `workflow_dispatch`, which will regenerate and publish a clean `data` branch). Flag this to the user when the plan's tasks are done; do not delete the shared `data` branch's committed files without confirming first — it's a destructive action outside this worktree.

---

### Task 1: `Gameweek` type

**Files:**
- Create: `src/degenebet/data/gameweek.py`
- Test: `tests/data/test_gameweek.py`

**Interfaces:**
- Produces: `Gameweek(NamedTuple)` with fields `season: int`, `week: int`, and method `as_int() -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_gameweek.py
from __future__ import annotations

from degenebet.data.gameweek import Gameweek


def test_as_int_combines_season_and_week() -> None:
    assert Gameweek(2024, 3).as_int() == 202403


def test_as_int_zero_pads_single_digit_weeks() -> None:
    assert Gameweek(2024, 1).as_int() == 202401


def test_gameweek_sorts_by_season_then_week() -> None:
    assert Gameweek(2023, 22) < Gameweek(2024, 1)
    assert Gameweek(2024, 1) < Gameweek(2024, 2)
    assert Gameweek(2024, 1) == Gameweek(2024, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_gameweek.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `gameweek.py`**

```python
"""Gameweek: the (season, week) granularity DataAccess indexes "which
games" queries by, instead of a raw calendar-date range. See
docs/superpowers/specs/2026-09-18-data-access-redesign.md, Decision 4 --
NFL's Thursday/Sunday/Monday week structure makes date-range filtering a
footgun (a naively-computed range can silently drop the Thursday or Monday
game of a week), and nflreadpy's own season/week columns already group
every game correctly regardless of which day it's played.
"""

from __future__ import annotations

from typing import NamedTuple


class Gameweek(NamedTuple):
    """A single NFL week within a season. Sorts correctly as a plain tuple
    (season compares first, then week) -- as_int() exists only for
    polars' convenience as a single sortable column. Its formula
    (season * 100 + week) must stay identical to the `gameweek` column
    nflverse_cache.py's sync_schedules/sync_team_stats compute -- see this
    plan's Global Constraints.
    """

    season: int
    week: int

    def as_int(self) -> int:
        return self.season * 100 + self.week
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_gameweek.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, commit**

Run: `just check`

```bash
git add src/degenebet/data/gameweek.py tests/data/test_gameweek.py
git commit -m "feat: add Gameweek type for season/week-indexed queries"
```

---

### Task 2: Write-boundary team-code normalization and `gameweek` column

**Files:**
- Modify: `src/degenebet/data/nflverse_cache.py`
- Modify: `tests/data/test_nflverse_cache.py`

**Interfaces:**
- Consumes: nothing new (uses `pl.col(...).str.to_lowercase()` and arithmetic already available).
- Produces: `sync_schedules`/`sync_team_stats` now write lowercase team codes and a `gameweek` int column into the persisted store, in addition to their existing behavior.

- [ ] **Step 1: Write the failing tests**

Replace the two existing sync tests in `tests/data/test_nflverse_cache.py` (`test_sync_schedules_fetches_and_merges` and `test_sync_team_stats_fetches_and_merges`) with:

```python
def test_sync_schedules_normalizes_team_codes_and_adds_gameweek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched = pl.DataFrame(
        {
            "game_id": ["a"],
            "season": [2024],
            "week": [3],
            "home_team": ["BUF"],
            "away_team": ["KC"],
            "result": [10],
        }
    )
    monkeypatch.setattr(
        "degenebet.data.nflverse_cache.nflverse.load_schedules", lambda seasons: fetched
    )

    result = nflverse_cache.sync_schedules(seasons=[2024])

    assert result["home_team"].to_list() == ["buf"]
    assert result["away_team"].to_list() == ["kc"]
    assert result["gameweek"].to_list() == [202403]
    stored = nflverse_cache.read_merged("schedules")
    assert stored is not None
    assert stored["home_team"].to_list() == ["buf"]
    assert stored["gameweek"].to_list() == [202403]


def test_sync_team_stats_normalizes_team_codes_and_adds_gameweek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched = pl.DataFrame(
        {
            "game_id": ["a"],
            "team": ["BUF"],
            "opponent_team": ["KC"],
            "season": [2024],
            "week": [3],
        }
    )
    monkeypatch.setattr(
        "degenebet.data.nflverse_cache.nflverse.load_team_stats", lambda seasons: fetched
    )

    result = nflverse_cache.sync_team_stats(seasons=[2024])

    assert result["team"].to_list() == ["buf"]
    assert result["opponent_team"].to_list() == ["kc"]
    assert result["gameweek"].to_list() == [202403]
    stored = nflverse_cache.read_merged("team_stats")
    assert stored is not None
    assert stored["team"].to_list() == ["buf"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_nflverse_cache.py -v`
Expected: FAIL — old assertions (`result.equals(fetched)`) no longer hold once normalization is expected; new assertions fail because the code doesn't normalize yet.

- [ ] **Step 3: Implement**

Replace `sync_schedules`/`sync_team_stats` in `src/degenebet/data/nflverse_cache.py`:

```python
def sync_schedules(seasons: list[int] | None = None) -> pl.DataFrame:
    """Fetch schedules from nflreadpy, normalize team codes to lowercase,
    add the `gameweek` sort/filter column, and merge into the persisted
    store."""
    fresh = nflverse.load_schedules(seasons).with_columns(
        pl.col("home_team").str.to_lowercase(),
        pl.col("away_team").str.to_lowercase(),
        (pl.col("season") * 100 + pl.col("week")).alias("gameweek"),
    )
    return load_or_merge(fresh, name="schedules", key_columns=["game_id"], group_column="season")


def sync_team_stats(seasons: list[int] | None = None) -> pl.DataFrame:
    """Fetch team stats from nflreadpy, normalize team codes to lowercase,
    add the `gameweek` sort/filter column, and merge into the persisted
    store."""
    fresh = nflverse.load_team_stats(seasons).with_columns(
        pl.col("team").str.to_lowercase(),
        pl.col("opponent_team").str.to_lowercase(),
        (pl.col("season") * 100 + pl.col("week")).alias("gameweek"),
    )
    return load_or_merge(
        fresh, name="team_stats", key_columns=["game_id", "team"], group_column="season"
    )
```

Also update this module's docstring to note the store now carries a `gameweek` column too (extend the existing "persisted store is always in canonical form" paragraph to mention it), for a future reader who doesn't have this plan in front of them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_nflverse_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, run the full suite, commit**

Run: `just check` — note `tests/data/test_access.py` will likely show failures at this point (it depends on `NflverseSource`, which Task 3 hasn't updated yet, and currently asserts old-shape behavior). That's expected and not this task's concern; confirm the failures are confined to `test_access.py` and not `test_nflverse_cache.py` or elsewhere before committing.

```bash
git add src/degenebet/data/nflverse_cache.py tests/data/test_nflverse_cache.py
git commit -m "feat: normalize team codes and add gameweek column at the merge-cache write boundary"
```

---

### Task 3: `NflverseSource` becomes `Gameweek`-native; split the `DataSource` protocol

**Files:**
- Modify: `src/degenebet/data/access.py`
- Modify: `tests/data/test_access.py`

**Interfaces:**
- Consumes: `Gameweek` (Task 1); the `gameweek` column on the merged schedules store (Task 2).
- Produces: `HistoricalDataSource` protocol (`fetch(self, start_week: Gameweek, end_week: Gameweek) -> pl.DataFrame`), `CurrentDataSource` protocol (`fetch(self, start_date: date, end_date: date) -> pl.DataFrame`, identical to today's `DataSource` but renamed), `NflverseSource.fetch(start_week: Gameweek, end_week: Gameweek) -> pl.DataFrame`.

- [ ] **Step 1: Write the failing tests**

Remove `test_nflverse_source_lowercases_team_codes` from `tests/data/test_access.py` entirely — normalization no longer happens in `NflverseSource` (Task 2 moved it to the write boundary), so this test's premise no longer applies.

Remove the `_raw_nflverse_schedule_row` helper — the merged store is now always pre-normalized (lowercase, `gameweek` column present), so there's no more "raw uppercase store content" scenario worth a dedicated fixture. Replace `test_nflverse_source_reads_merged_store_filtered_to_range` with:

```python
def test_nflverse_source_filters_by_gameweek_range() -> None:
    nflverse_cache.load_or_merge(
        pl.DataFrame(
            [
                _schedule_row(game_id="g_w2", season=2026, week=2, gameweek=202602),
                _schedule_row(game_id="g_w5", season=2026, week=5, gameweek=202605),
            ]
        ),
        name="schedules",
        key_columns=["game_id"],
        group_column="season",
    )

    result = NflverseSource().fetch(Gameweek(2026, 1), Gameweek(2026, 3))

    assert result["game_id"].to_list() == ["g_w2"]
```

Update the `_schedule_row` helper (used throughout this file) to include `season`, `week`, and `gameweek` by default, matching what the real normalized store now always carries:

```python
def _schedule_row(**overrides: object) -> dict[str, object]:
    """A row as the merged schedules store holds it post-normalization
    (Task 2) -- lowercase team codes, gameweek present. Used both to seed
    the store directly and to build _FakeSource fixtures standing in for
    a DataSource's output."""
    row: dict[str, object] = {
        "game_id": "2026_02_MIN_CHI",
        "season": 2026,
        "week": 2,
        "gameweek": 202602,
        "gameday": "2026-09-20",
        "home_team": "chi",
        "away_team": "min",
        "result": None,
        "spread_line": None,
    }
    row.update(overrides)
    return row
```

Add `from degenebet.data.gameweek import Gameweek` to this test file's imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_access.py -v`
Expected: FAIL — `NflverseSource.fetch` doesn't accept `Gameweek` yet, `_schedule_row` calls with `gameweek=` fail since the current signature doesn't expect it (it will, it's just `**overrides`, so this part won't fail, but callers passing positional `date(...)` args to `.fetch()` will).

- [ ] **Step 3: Implement**

In `src/degenebet/data/access.py`, add `from degenebet.data.gameweek import Gameweek` to imports. Replace the `DataSource` protocol and `NflverseSource`:

```python
class HistoricalDataSource(Protocol):
    def fetch(self, start_week: Gameweek, end_week: Gameweek) -> pl.DataFrame: ...


class CurrentDataSource(Protocol):
    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame: ...
```

```python
class NflverseSource:
    """Historical schedule source: reads the persisted merged schedules
    store (nflverse_cache.py), never the network. Team codes and the
    `gameweek` column are already normalized at the merge-cache write
    boundary (nflverse_cache.sync_schedules) -- this is a pure read, no
    transformation of its own."""

    def fetch(self, start_week: Gameweek, end_week: Gameweek) -> pl.DataFrame:
        merged = nflverse_cache.read_merged("schedules")
        if merged is None:
            raise RuntimeError(
                "No schedules have ever been synced -- run `degenebet fetch schedules` "
                "or `degenebet sync` first."
            )
        return merged.filter(
            (pl.col("gameweek") >= start_week.as_int())
            & (pl.col("gameweek") <= end_week.as_int())
        )
```

Update `SharpApiSource`'s class docstring only (no behavior change) to state it implements `CurrentDataSource`, not the old `DataSource` name, for clarity to a future reader. Leave `SharpApiSource.fetch`'s body untouched.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_access.py -v`
Expected: `NflverseSource`-specific tests PASS. Tests exercising `DataAccess.get_team_data(...)` will still FAIL at this point — Task 4 hasn't rewritten `DataAccess` yet. Confirm the only failures remaining are `DataAccess`-level tests, not `NflverseSource`/`SharpApiSource`-level ones.

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run mypy` and `uv run ruff check src/ tests/` directly (not full `just check`, since `DataAccess`-level test failures are expected and addressed in Task 4) — confirm no lint/type errors.

```bash
git add src/degenebet/data/access.py tests/data/test_access.py
git commit -m "feat: make NflverseSource Gameweek-native, split DataSource protocol"
```

---

### Task 4: Rewrite `DataAccess`'s stitching core for `Gameweek` and gameweek-based reconciliation

**Files:**
- Modify: `src/degenebet/data/access.py`
- Modify: `tests/data/test_access.py`

**Interfaces:**
- Consumes: `HistoricalDataSource`, `CurrentDataSource`, `Gameweek` (Task 3, Task 1).
- Produces: `DataAccess._get_stitched_schedule(self, start_week: Gameweek, end_week: Gameweek, as_of_date: date) -> pl.DataFrame` (private, game-indexed — today's `get_team_data`'s exact output shape, just renamed and reached via the new params). `DataAccess.__init__(self, historical: HistoricalDataSource, current: CurrentDataSource)`.

This task does **not** yet expose `get_game_data`/`get_team_data` publicly — Task 5 does. This task only gets the core stitching logic correct and privately callable, so it can be tested in isolation from the public-API/widening work.

- [ ] **Step 1: Write the failing tests**

Replace every `DataAccess.get_team_data(...)` call in the existing tests with `DataAccess(...)._get_stitched_schedule(...)`, and replace `date(...)` range arguments with `Gameweek(...)`. **Rename each test too** — these six currently say `test_get_team_data_*`, but Task 5 adds a real *public* `get_team_data` with a completely different (team-indexed) shape; leaving these names as-is would leave two differently-shaped things both called "get_team_data" in the same test file. Rename the `test_get_team_data_` prefix to `test_stitched_schedule_` throughout. Concretely, in `tests/data/test_access.py`:

- `_FakeSource.fetch` keeps its existing signature shape but is now called with whatever args the real protocol expects per source (historical fakes receive `Gameweek` args in tests that call `_get_stitched_schedule` directly with them; the fake itself just ignores args and returns its fixed frame, as today).
- Update and rename all six existing `DataAccess`-level tests:
  - `test_get_team_data_prefers_current_for_unplayed_game_even_with_stale_historical_line` → `test_stitched_schedule_prefers_current_for_unplayed_game_even_with_stale_historical_line`
  - `test_get_team_data_historical_wins_for_settled_result` → `test_stitched_schedule_historical_wins_for_settled_result`
  - `test_get_team_data_falls_back_to_historical_when_current_missing_game` → `test_stitched_schedule_falls_back_to_historical_when_current_missing_game`
  - `test_get_team_data_warns_when_as_of_before_earliest_history` → `test_stitched_schedule_warns_when_as_of_before_earliest_history`
  - `test_get_team_data_selects_snapshot_closest_to_but_not_after_as_of_date` → `test_stitched_schedule_selects_snapshot_closest_to_but_not_after_as_of_date`
  - `test_get_team_data_end_to_end_with_real_sources` → `test_stitched_schedule_end_to_end_with_real_sources`

  Each calls `._get_stitched_schedule(Gameweek(2026, 1), Gameweek(2026, 3), as_of_date=...)` instead of `.get_team_data(date(2026, 9, 1), date(2026, 9, 30), as_of_date=...)`. The fixtures' `_schedule_row(...)` already carry `season`/`week`/`gameweek` (Task 3), so no fixture data changes are needed beyond the method-call rewrite and rename.

Add one new test for the reconciliation join's new robustness property — the exact scenario an exact-`gameday` join would have missed:

```python
def test_stitched_schedule_reconciles_on_gameweek_not_exact_gameday() -> None:
    # historical's gameday (2026-09-20, a Sunday) and current's derived
    # gameday (2026-09-19, one day off -- simulating any date-precision
    # discrepancy) disagree, but both belong to the same (home_team,
    # away_team, season, week). The old exact-gameday join would silently
    # miss this; gameweek-based reconciliation must not.
    historical = _FakeSource(
        pl.DataFrame([_schedule_row(gameday="2026-09-20", result=None, spread_line=1.0)])
    )
    current = _FakeSource(
        pl.DataFrame(
            {
                "home_team": ["chi"],
                "away_team": ["min"],
                "gameday": ["2026-09-19"],
                "spread_line": [2.5],
                "pulled_at": [datetime(2026, 9, 17, tzinfo=UTC)],
            }
        )
    )

    result = DataAccess(historical, current)._get_stitched_schedule(
        Gameweek(2026, 1), Gameweek(2026, 3), as_of_date=date(2026, 9, 17)
    )

    assert result["spread_line"][0] == pytest.approx(2.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_access.py -v`
Expected: FAIL — `_get_stitched_schedule` doesn't exist yet; `DataAccess.__init__` still type-hints the old protocol names (harmless at runtime, but the method calls fail).

- [ ] **Step 3: Implement**

Replace `DataAccess` in `src/degenebet/data/access.py`:

```python
class DataAccess:
    """Fetches NFL data at whatever granularity a caller needs -- see
    docs/superpowers/specs/2026-09-18-data-access-redesign.md."""

    def __init__(self, historical: HistoricalDataSource, current: CurrentDataSource) -> None:
        self.historical = historical
        self.current = current

    def _get_stitched_schedule(
        self, start_week: Gameweek, end_week: Gameweek, as_of_date: date
    ) -> pl.DataFrame:
        """Game-indexed, point-in-time-correct: historical's own spread_line
        for a settled game; current's (SharpAPI's) for an unplayed one,
        whenever current covers it, even if historical already has an early
        line for that same game; historical's line as the fallback when
        current doesn't cover it. Shared by get_game_data and
        get_team_data."""
        historical = self.historical.fetch(start_week, end_week)
        if historical.height == 0:
            return historical

        earliest = str(historical["gameday"].min())
        if as_of_date.isoformat() < earliest:
            # Informational only -- do NOT clamp as_of_date forward.
            # Clamping would admit current snapshots pulled after the true
            # requested as_of_date (a look-ahead leak); the honest behavior
            # for "as_of_date predates any cached history" is to proceed
            # with the original as_of_date.
            warnings.warn(
                f"as_of_date {as_of_date} predates earliest cached history {earliest}.",
                stacklevel=2,
            )

        min_date = date.fromisoformat(earliest)
        max_date = date.fromisoformat(str(historical["gameday"].max()))
        current = self.current.fetch(min_date, max_date)

        current_asof = (
            current.filter(pl.col("pulled_at").dt.date() <= as_of_date)
            if current.height > 0
            else current
        )

        if current_asof.height > 0:
            # current has no season/week of its own -- resolve each row's
            # (season, week) by matching (home_team, away_team) against
            # historical and taking the closest gameday, disambiguating the
            # rare case where a Gameweek range spans multiple seasons and
            # the same team pairing appears more than once.
            historical_games = historical.select(
                "home_team",
                "away_team",
                pl.col("gameday").alias("historical_gameday"),
                "season",
                "week",
            )
            current_asof = (
                current_asof.join(historical_games, on=["home_team", "away_team"], how="inner")
                .with_columns(
                    (
                        pl.col("gameday").str.to_date()
                        - pl.col("historical_gameday").str.to_date()
                    )
                    .dt.total_days()
                    .abs()
                    .alias("_date_distance")
                )
                .sort("_date_distance")
                .group_by(["home_team", "away_team", "gameday", "pulled_at"], maintain_order=True)
                .first()
                .drop("historical_gameday", "_date_distance")
            )
            # Multiple snapshots may survive the as-of cutoff for the same
            # game; keep only the one closest to (but not after) as_of_date.
            current_asof = (
                current_asof.sort("pulled_at")
                .group_by(["season", "week", "home_team", "away_team"], maintain_order=True)
                .last()
            )

        unresolved = historical.filter(pl.col("result").is_null()).select(
            "game_id", "home_team", "away_team", "season", "week"
        )
        if current_asof.height > 0:
            current_for_unresolved = current_asof.join(
                unresolved, on=["home_team", "away_team", "season", "week"], how="inner"
            ).select("game_id", pl.col("spread_line").alias("current_spread_line"))
        else:
            current_for_unresolved = pl.DataFrame(
                schema={"game_id": pl.Utf8, "current_spread_line": pl.Float64}
            )

        return (
            historical.join(current_for_unresolved, on="game_id", how="left")
            .with_columns(
                pl.coalesce(["current_spread_line", "spread_line"]).alias("spread_line")
            )
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
git commit -m "feat: rewrite DataAccess stitching for Gameweek and gameweek-based reconciliation"
```

---

### Task 5: `get_game_data`, `get_team_data`, and `team_data` widening

**Files:**
- Modify: `src/degenebet/data/access.py`
- Modify: `tests/data/test_access.py`

**Interfaces:**
- Consumes: `DataAccess._get_stitched_schedule` (Task 4); `nflverse_cache.read_merged("team_stats")`.
- Produces: `DataAccess.get_game_data(start_week, end_week, as_of_date, *, team_data=None) -> pl.DataFrame`, `DataAccess.get_team_data(start_week, end_week, as_of_date) -> pl.DataFrame`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_access.py`:

```python
def test_to_team_indexed_produces_two_rows_per_game_with_signed_perspective() -> None:
    game_table = pl.DataFrame(
        [_schedule_row(result=7, spread_line=-3.0)]
    )

    long_table = DataAccess(
        _FakeSource(pl.DataFrame()), _FakeSource(pl.DataFrame())
    )._to_team_indexed(game_table)

    assert long_table.height == 2
    home_row = long_table.filter(pl.col("is_home"))
    away_row = long_table.filter(~pl.col("is_home"))
    assert home_row["team"][0] == "chi"
    assert home_row["opponent"][0] == "min"
    assert home_row["team_spread_line"][0] == pytest.approx(-3.0)
    assert home_row["team_margin"][0] == pytest.approx(7)
    assert away_row["team"][0] == "min"
    assert away_row["opponent"][0] == "chi"
    assert away_row["team_spread_line"][0] == pytest.approx(3.0)
    assert away_row["team_margin"][0] == pytest.approx(-7)


def test_get_team_data_left_joins_team_stats_null_for_unplayed_game() -> None:
    historical = _FakeSource(pl.DataFrame([_schedule_row(result=None, spread_line=1.0)]))
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
    nflverse_cache.load_or_merge(
        pl.DataFrame(
            {
                "game_id": ["2026_02_MIN_CHI"],
                "team": ["chi"],
                "season": [2026],
                "week": [2],
                "passing_epa": [12.5],
            }
        ),
        name="team_stats",
        key_columns=["game_id", "team"],
        group_column="season",
    )

    result = DataAccess(historical, current).get_team_data(
        Gameweek(2026, 1), Gameweek(2026, 3), as_of_date=date(2026, 9, 17)
    )

    chi_row = result.filter(pl.col("team") == "chi")
    min_row = result.filter(pl.col("team") == "min")
    assert chi_row["passing_epa"][0] == pytest.approx(12.5)
    assert min_row["passing_epa"][0] is None


def test_get_game_data_widens_team_data_with_home_away_prefixes() -> None:
    historical = _FakeSource(pl.DataFrame([_schedule_row(result=7, spread_line=-3.0)]))
    current = _FakeSource(pl.DataFrame())
    team_data = pl.DataFrame(
        {
            "game_id": ["2026_02_MIN_CHI", "2026_02_MIN_CHI"],
            "team": ["chi", "min"],
            "season": [2026, 2026],
            "week": [2, 2],
            "gameweek": [202602, 202602],
            "gameday": ["2026-09-20", "2026-09-20"],
            "opponent": ["min", "chi"],
            "is_home": [True, False],
            "team_spread_line": [-3.0, 3.0],
            "team_margin": [7.0, -7.0],
            "rolling_offense_epa_per_play": [1.2, 0.8],
        }
    )

    result = DataAccess(historical, current).get_game_data(
        Gameweek(2026, 1), Gameweek(2026, 3), as_of_date=date(2026, 9, 17), team_data=team_data
    )

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["home_rolling_offense_epa_per_play"] == pytest.approx(1.2)
    assert row["away_rolling_offense_epa_per_play"] == pytest.approx(0.8)
    # The duplicate-column trap this design exists to avoid:
    assert "home_team_spread_line" not in result.columns
    assert "home_opponent" not in result.columns
    assert "home_is_home" not in result.columns
    assert "home_season" not in result.columns


def test_get_game_data_without_team_data_returns_bare_schedule() -> None:
    historical = _FakeSource(pl.DataFrame([_schedule_row(result=7, spread_line=-3.0)]))
    current = _FakeSource(pl.DataFrame())

    result = DataAccess(historical, current).get_game_data(
        Gameweek(2026, 1), Gameweek(2026, 3), as_of_date=date(2026, 9, 17)
    )

    assert result.height == 1
    assert "home_team" in result.columns
    assert "rolling_offense_epa_per_play" not in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_access.py -v`
Expected: FAIL (`AttributeError`/`ModuleNotFoundError`-shaped failures — `_to_team_indexed`, `get_team_data`, `get_game_data` don't exist in the new shape yet).

- [ ] **Step 3: Implement**

Add to `src/degenebet/data/access.py`, near `_SHARPAPI_TEAM_CROSSWALK`:

```python
# get_team_data's own columns that are just a team-indexed restatement of
# information get_game_data's base table already has (season/week/gameweek/
# gameday, the spread/result pair re-signed per team, home/away, opponent).
# _widen_with_team_data excludes these so widening never produces columns
# like home_team_spread_line duplicating spread_line.
_TEAM_DATA_CONTEXT_COLUMNS = frozenset(
    {
        "game_id",
        "team",
        "season",
        "week",
        "gameweek",
        "gameday",
        "opponent",
        "is_home",
        "team_spread_line",
        "team_margin",
    }
)
```

Add to `DataAccess`:

```python
    def _to_team_indexed(self, game_table: pl.DataFrame) -> pl.DataFrame:
        """One game row -> two team rows (home's perspective, away's),
        team_spread_line/team_margin re-signed per team."""
        if game_table.height == 0:
            return game_table
        home_side = game_table.select(
            "game_id",
            "season",
            "week",
            "gameweek",
            "gameday",
            pl.col("home_team").alias("team"),
            pl.col("away_team").alias("opponent"),
            pl.lit(True).alias("is_home"),
            pl.col("spread_line").alias("team_spread_line"),
            pl.col("result").alias("team_margin"),
        )
        away_side = game_table.select(
            "game_id",
            "season",
            "week",
            "gameweek",
            "gameday",
            pl.col("away_team").alias("team"),
            pl.col("home_team").alias("opponent"),
            pl.lit(False).alias("is_home"),
            (-pl.col("spread_line")).alias("team_spread_line"),
            (-pl.col("result")).alias("team_margin"),
        )
        return pl.concat([home_side, away_side], how="vertical")

    def _widen_with_team_data(
        self, game_table: pl.DataFrame, team_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Join `team_data` (keyed on game_id, team) onto `game_table` twice
        -- home perspective and away perspective -- prefixing every
        non-context column home_/away_. Generic: works whether `team_data`
        is get_team_data()'s own output or compute_rolling_features()'s."""
        payload_columns = [c for c in team_data.columns if c not in _TEAM_DATA_CONTEXT_COLUMNS]
        home_payload = team_data.select(
            "game_id",
            "team",
            *[pl.col(c).alias(f"home_{c}") for c in payload_columns],
        )
        away_payload = team_data.select(
            "game_id",
            "team",
            *[pl.col(c).alias(f"away_{c}") for c in payload_columns],
        )
        return (
            game_table.join(
                home_payload, left_on=["game_id", "home_team"], right_on=["game_id", "team"],
                how="left",
            )
            .join(
                away_payload, left_on=["game_id", "away_team"], right_on=["game_id", "team"],
                how="left",
            )
        )

    def get_game_data(
        self,
        start_week: Gameweek,
        end_week: Gameweek,
        as_of_date: date,
        *,
        team_data: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """One row per game: game_id, season, week, gameweek, gameday,
        home_team, away_team, result, spread_line. If `team_data` is given
        (any team-indexed table keyed on game_id, team), its payload
        columns are joined in twice, prefixed home_/away_."""
        game_table = self._get_stitched_schedule(start_week, end_week, as_of_date)
        if team_data is None or game_table.height == 0:
            return game_table
        return self._widen_with_team_data(game_table, team_data)

    def get_team_data(
        self, start_week: Gameweek, end_week: Gameweek, as_of_date: date
    ) -> pl.DataFrame:
        """One row per (game_id, team): schedule/spread context from that
        team's own perspective, plus the full raw team_stats row for
        (game_id, team) left-joined -- null for a game with no stats yet
        (it hasn't been played)."""
        game_table = self._get_stitched_schedule(start_week, end_week, as_of_date)
        long_table = self._to_team_indexed(game_table)
        team_stats = nflverse_cache.read_merged("team_stats")
        if team_stats is None or long_table.height == 0:
            return long_table
        return long_table.join(team_stats, on=["game_id", "team"], how="left")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_access.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, type-check, run the full suite, commit**

Run: `just check`

```bash
git add src/degenebet/data/access.py tests/data/test_access.py
git commit -m "feat: add DataAccess.get_game_data/get_team_data with team_data widening"
```

---

## After all tasks: broader verification

- Confirm `docs/superpowers/specs/2026-09-18-data-access-redesign.md`'s Testing approach section is covered: write-time normalization ✅ (Task 2), context-vs-payload column shape ✅ (Task 5), null stats for unplayed game ✅ (Task 5), sign correctness for both home/away rows ✅ (Task 5), gameweek-based reconciliation resolving a non-exact-gameday match ✅ (Task 4), a `Gameweek` range spanning Thursday-through-Monday — add this as one more explicit regression test if no existing task's fixtures happen to cover a genuine 3-games-one-week scenario (`NflverseSource`'s Task 3 test uses two different weeks, not three games in one week — worth a final check here, not a new task if Task 3's filtering logic already makes it structurally impossible to drop a same-week game).
- Perform the "Manual setup required" step above (regenerate local + `data`-branch stores) and confirm `degenebet fetch schedules`/`degenebet fetch team-stats` succeed and `DataAccess.get_team_data(...)`/`get_game_data(...)` work against the regenerated real data, not just fixtures.
- `features.build_model_table`'s retirement and `backtest.py`/`spread_model.py`'s migration to the new `get_game_data(team_data=...)` shape is explicitly **out of scope** for this plan (see the spec's Non-goals) — a follow-up plan, not a loose end of this one.
