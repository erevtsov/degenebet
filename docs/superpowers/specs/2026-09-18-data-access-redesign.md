# DataAccess redesign — design

Date: 2026-09-18
Status: Approved for implementation planning

## Context

The first `DataAccess` sub-project (`docs/superpowers/specs/2026-09-17-data-access-design.md`)
shipped `get_team_data(start_date, end_date, as_of_date)` — despite its name,
a *game*-indexed table (one row per game, `home_team`/`away_team` side by
side). `team_stats` (EPA/turnover data) has no `DataAccess` method at all;
it was deliberately scoped out, since it has no "current" counterpart to
stitch against.

Building the first real consumer of this layer (a marimo notebook doing
simple analysis over `DataAccess`) exposed the consequence directly: a
caller who wants both schedule/spread context and team stats has to call
`DataAccess` for one and reach into `nflverse_cache.read_merged("team_stats")`
directly for the other, then manually reconcile that `DataAccess` normalizes
team codes to lowercase (`teams.CANONICAL_TEAMS`) while a raw
`nflverse_cache` read does not. That's exactly the kind of naming/convention
leak `DataAccess` exists to prevent.

This spec redesigns `DataAccess` into the single, coherent place to fetch
NFL data, at whatever granularity the caller needs, with no reconciliation
work left for the caller.

## Decisions

### 1. Team-code normalization moves to the write boundary, not the read boundary

Today, `NflverseSource.fetch()` lowercases `home_team`/`away_team` at *read*
time; the persisted merged store itself stays in `nflreadpy`'s native
uppercase. That's why a direct `nflverse_cache.read_merged(...)` call (the
only way to read `team_stats`, which has no `DataAccess` wrapper) sees raw
uppercase — there's no read-time boundary for it to normalize at, since it
was never wrapped in a `DataSource`.

Fix: `nflverse_cache.sync_schedules()`/`sync_team_stats()` normalize team
codes to lowercase immediately after fetching from `nflreadpy`, before
merging into the persisted store. The persisted store is now *always*
canonical, regardless of read path — `DataAccess`, a direct `read_merged()`
call, or any future consumer. `NflverseSource.fetch()` drops its own
lowercasing step entirely (redundant once the store is pre-normalized).

**Operational consequence, not a design question but worth stating
plainly:** the already-persisted stores (local dev caches and the shared
`data` git branch, both populated by real runs before this change) have
uppercase codes. For `schedules` (keyed on `game_id` alone), a fresh sync
after this ships self-heals cleanly — the merge key doesn't include team
casing, so the new lowercase row simply replaces the old uppercase one.
For `team_stats` (keyed on `game_id, team`), it does **not** self-heal —
old `"BUF"` and new `"buf"` are different keys under `merge_frames`, so a
fresh sync would *duplicate* every row rather than replace it. The
implementation plan must include a one-time regeneration of the persisted
`team_stats` store (and, for symmetry, `schedules`) after this lands.

### 2. `DataAccess` gains two granularities: `get_game_data` and `get_team_data`, on the same class

Per `docs/plan.md`'s original naming intent (`get_team_data`, not a generic
`get_data`, specifically to leave room for a future `get_player_data`) —
"team" was always about *granularity* (team vs. player), not "one row per
team." Today's method just happens to be game-indexed under that name. This
redesign corrects that:

- **`get_game_data(start_week, end_week, as_of_date, *, team_data=None)`**
  — one row per game. Columns: `game_id, season, week, gameweek, gameday,
  home_team, away_team, result, spread_line` (today's shape, unchanged in
  spirit — see Decision 4 for the parameter change from dates to
  `Gameweek`).
- **`get_team_data(start_week, end_week, as_of_date)`** — one row per
  `(game_id, team)`. Columns:
  - Context: `game_id, team, opponent, is_home, season, week, gameweek,
    gameday, team_spread_line, team_margin`. `team_spread_line` and
    `team_margin` are `spread_line`/`result` re-signed to this team's own
    perspective (positive `team_spread_line` = this team favored); both
    null until known, exactly as `spread_line`/`result` are today.
  - Payload: every column of the raw `team_stats` row for
    `(game_id, team)`, left-joined — null for a game with no stats yet
    (it hasn't been played). This is `nflreadpy`'s own per-game team
    stats, not the rolling/no-lookahead features `compute_rolling_features`
    computes — see Decision 3 for why that distinction matters.
  - `get_player_data(...)` is anticipated by this naming but not built now
    — see Non-goals.

Both methods share a private `_get_stitched_schedule(start_week, end_week,
as_of_date)` (today's historical/current dedup logic, reused rather than
duplicated — see Decision 4 for how the join itself changes) and a private
`_to_team_indexed(game_table)` unpivot helper (one game row → two team
rows: home's perspective and away's perspective).

### 3. `get_game_data(team_data=...)` is the one place team-indexed data gets widened onto game rows

Rather than a boolean `include_team_stats` flag, `get_game_data` accepts
`team_data: pl.DataFrame | None` — any team-indexed table keyed on
`(game_id, team)`. This is deliberately generic because there are two
legitimately different things a caller might want to widen:

- `get_team_data(...)`'s own output (raw per-game team stats) — useful for
  reporting/analysis, **not** safe to feed a model directly: using a
  team's stats *from the game being predicted* is leakage.
- `compute_rolling_features(get_team_data(...))`'s output (trailing,
  `.shift(1)`-ed rolling averages) — the actual leakage-safe model input.

Both are team-indexed tables keyed on `(game_id, team)` (well, `season,
week, team, game_id` for `compute_rolling_features`'s current output);
`get_game_data` doesn't need to know or care which kind it's given.

**The duplicate-column trap, and how the widening avoids it:** naively
joining `team_data`'s full column set (prefixed `home_`/`away_`) would
produce `home_team_spread_line` duplicating `spread_line`,
`home_team_margin` duplicating `result`, `home_opponent` duplicating
`away_team`, etc. — `get_team_data`'s *context* columns are just
team-indexed restatements of information `get_game_data`'s base table
already has. The widening excludes a fixed, documented context-column set
(module-level constant):

```
{"game_id", "team", "season", "week", "gameweek", "gameday",
 "opponent", "is_home", "team_spread_line", "team_margin"}
```

Only the remaining *payload* columns get joined (on `game_id` + a
home/away-specific `team` match) and prefixed. This same exclusion set
correctly handles both `get_team_data()`'s full output and
`compute_rolling_features()`'s narrower output, since the latter's only
non-payload columns (`season, week, team, game_id`) are already a subset
of it — no special-casing needed between the two cases.

**Consequence: `features.build_model_table` becomes redundant.** It exists
today to do exactly this home/away pivot-and-rename, by hand, for one
specific case (rolling features). Once `get_game_data(team_data=...)` does
this generically, `build_model_table` should be retired and its callers
(`backtest.py`, `spread_model.py`'s golden tests, any other consumer)
migrated to `compute_rolling_features(...)` piped into
`DataAccess.get_game_data(..., team_data=...)`. `compute_rolling_features`
itself is unaffected — its rolling-window math has nothing to do with this
redesign and stays exactly where it is, consistent with the original
`DataAccess` design's decision to keep no-lookahead feature computation out
of the access layer. Working through every affected call site is
implementation-plan scope, not this spec's — flagged here so it isn't
discovered mid-implementation.

### 4. `Gameweek` replaces date ranges for "which games"; `as_of_date` stays a real date for "as of when"

**The problem:** `start_date`/`end_date` filtering directly on `gameday` is
a footgun for NFL's Thursday/Sunday/Monday week structure — a caller who
picks a date range meant to capture "week 2" using that Sunday as both
bounds silently drops week 2's Thursday game (three days earlier) and
Monday game (a day later), both genuinely part of that week.  The same
imprecision affects the *internal* historical/current reconciliation join
today, which matches on exact `(home_team, away_team, gameday)` equality —
a SharpAPI event landing even one day off `nflreadpy`'s `gameday` (a
reschedule, a data-quality quirk, an edge case like the UTC/Eastern
timezone bug already found and fixed once) silently fails that join.

**The fix:** a new small type,

```python
class Gameweek(NamedTuple):
    season: int
    week: int

    def as_int(self) -> int:
        return self.season * 100 + self.week
```

`Gameweek` already sorts correctly as a plain Python tuple (season first,
then week) — `as_int()` exists purely for polars' convenience: a single
sortable integer column beats a two-column boundary check or `gameday`'s
string-comparison trick.

**Signatures change asymmetrically, and that's intentional:**
- `NflverseSource.fetch(start_week: Gameweek, end_week: Gameweek)` — it has
  real `season`/`week` columns, so this is a direct filter, no translation
  needed.
- `SharpApiSource.fetch(start_date: date, end_date: date)` — **unchanged**.
  SharpAPI has no concept of an NFL week and should not be made to know
  one, and it should not be made to know about `historical` either — it
  stays exactly the self-contained, decoupled source it is today.
- `DataAccess`'s two public methods take `(start_week: Gameweek, end_week:
  Gameweek, as_of_date: date)`. `as_of_date` stays a real `date` regardless
  — it answers "as of what real-world moment," a genuinely different axis
  from "which games," and folding it into `Gameweek` would conflate them.

**How `DataAccess` reconciles the two without a separate season-calendar
table:** `_get_stitched_schedule` calls `historical.fetch(start_week,
end_week)` first. The calendar-date bounds implied by that range fall out
for free — `historical["gameday"].min()`/`.max()` on the result — no
separate "what dates does week 2 of 2026 span" lookup needed. Those derived
bounds are what get passed to `current.fetch(...)`.

For the actual reconciliation, each `current` row's `(season, week)` is
resolved by matching `(home_team, away_team)` within the already-fetched
`historical` set and, where more than one historical row shares that pair
(a `Gameweek` range spanning multiple seasons), taking the one whose
`gameday` is closest to the current row's own `gameday`. The join then
proceeds on `(home_team, away_team, season, week)` instead of exact
`gameday` — coarser, and correspondingly more robust to the class of
date-precision bug already found once.

**Consequence: `DataSource` is no longer one uniform protocol.** Today both
sources implement identical `fetch(start_date, end_date)` signatures;
after this, `NflverseSource` takes `Gameweek` and `SharpApiSource` takes
`date`. This is a real, deliberate departure from the original design's
protocol uniformity — but `DataAccess` never actually treats the two
sources polymorphically in the code (only `current` gets `pulled_at`-based
as-of filtering; only `historical` gets the earliest-history check), so the
shared protocol was cosmetic, not load-bearing. Split into two honestly
different, correctly-typed interfaces rather than force artificial
symmetry — exact shape (two small `Protocol`s vs. typing each `DataAccess`
attribute directly) is implementation-plan detail.

`gameweek` (the materialized `season*100+week` int column) is added at the
same write boundary as Decision 1's team-code normalization
(`sync_schedules`/`sync_team_stats`) — `season`/`week` stay untouched
alongside it, consistent with "normalize/derive once at write time, every
reader benefits regardless of path."

## Non-goals (deferred, not decided against)

- `get_player_data(...)` — the naming leaves room for it (per `docs/plan.md`
  and the original `DataAccess` spec), but it isn't built now.
- Exact `DataSource` protocol formalization (two split `Protocol`s vs.
  per-attribute typing on `DataAccess.__init__`) — implementation-plan
  detail, not a design question with two materially different outcomes.
- Migrating already-persisted local and `data`-branch stores to the new
  normalized/`gameweek`-bearing schema — a real, required one-time
  operational step (see Decision 1), but not a design decision; tracked as
  an implementation-plan task.
- Working through every existing caller of `features.build_model_table`
  (Decision 3) in detail — real, required, but implementation-plan scope.

## Testing approach

Per `AGENTS.md`'s testing rules — real in-memory Polars DataFrames, no
network calls in the default test run:
- Write-time normalization: `sync_schedules`/`sync_team_stats` lowercase
  correctly; a fresh `team_stats` sync after an existing uppercase-keyed
  store would need the migration step from Decision 1 (not silently
  producing duplicates going forward, once regenerated).
- `get_team_data`: context vs. payload column shape; null stats for an
  unplayed game; `team_spread_line`/`team_margin` sign correctness for
  both home and away rows of the same game.
- `get_game_data(team_data=...)`: the exact duplicate-column trap from
  Decision 3 needs a direct regression test (widen `get_team_data()`'s own
  output and assert `home_team_spread_line` etc. do *not* appear); a
  second test widening `compute_rolling_features()`'s output confirms the
  same exclusion set handles both shapes.
- `Gameweek` reconciliation: a case where a `current` row's `gameday`
  doesn't exactly match `historical`'s `gameday` for the same game (the
  scenario the old exact-match join would have silently failed) resolves
  correctly under nearest-date `(season, week)` matching.
- A `Gameweek` range spanning a Thursday-through-Monday week confirms all
  three games are included — the regression test for the bug motivating
  Decision 4 in the first place.
