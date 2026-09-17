# DataAccess & scheduled fetch — design

Date: 2026-09-17
Status: Approved for implementation planning

## Context

`docs/plan.md`'s Data section calls for a `get_data(start_date, end_date,
as_of_date)` access layer that stitches historical (`nflreadpy`) and current
(SharpAPI) sources into one point-in-time-correct view, stored locally with a
timestamp, never re-fetched live per call. `docs/architecture-notes.md`
already settled the shape of this as `DataAccess.get_team_data` (named for
future `get_player_data` symmetry) plus `NflverseSource`/`SharpApiSource`
wrapping the existing loaders — but left two things open: the exact source
interfaces, and how the cache stays fresh without either (a) a live fetch on
every read, or (b) depending on a personal laptop being on 24/7.

This spec settles those two gaps so the sub-project can be planned and
built. It does not revisit anything already settled in
`docs/architecture-notes.md`.

## Decisions

### 1. Scope: `DataAccess` stitches schedules, not team stats

`NflverseSource`/`SharpApiSource` don't actually provide the same kind of
data. `nflreadpy` team_stats (EPA/play, turnover margin, etc.) only exist for
games that have already been played — there is no such thing as a future
game's EPA. SharpAPI, conversely, only has odds for upcoming games. So the
two sources never compete over team-stat rows; they compete over **schedule**
rows (game_id, season, week, home_team, away_team, spread_line, result) —
`nflreadpy` for completed games, SharpAPI for the ones it hasn't reached yet.

`DataAccess.get_team_data` therefore stitches at the schedule level:
- `historical` (`NflverseSource`) supplies every game up to `as_of_date`
  that `nflreadpy` already has, `result`/`spread_line` included where known.
- `current` (`SharpApiSource`) fills only the gap — upcoming games
  `nflreadpy` doesn't have yet, with `result` null and `spread_line` from the
  odds snapshot closest to (but not after) `as_of_date`.
- Dedup is on `game_id`: `historical` always wins if both sources have the
  same game, since a completed game's SharpAPI odds snapshot is stale
  relative to the actual settled result.

Team-stat rows (`compute_rolling_features`'s input) stay a pure `nflreadpy`
concern, unchanged from today's `features.py` — they're never stitched,
because there's nothing to stitch against.

This also answers architecture-notes.md's open "how does current age out
into historical" question: it isn't a separate mechanism. Once a scheduled
fetch pulls a completed game into the historical cache, `historical` simply
starts winning that `game_id` in the dedup — no explicit aging step needed.

### 2. Freshness: scheduled fetch on GitHub Actions, not the laptop

`DataAccess` must never fetch live in its read path (`get_team_data` reads
local files only, per plan.md's "stored with a timestamp"). But something
has to keep those local files fresh, and a personal laptop isn't reliably
on. Rather than stand up paid external storage (rejected — unnecessary cost
for NFL data, which totals a few MB even across seasons), reuse
infrastructure this repo already has:

- A new scheduled GitHub Actions workflow runs `degenebet fetch` on GitHub's
  infra (cron-triggered, so it runs regardless of the laptop's state) and
  commits the resulting cache files to a dedicated `data` branch — never
  `master`, so branch protection is untouched. Cadence: daily to start
  (games/odds don't move fast enough to need finer-grained automation yet;
  easy to tighten later).
- The local machine (or CI, or anywhere else) never fetches live either. A
  new `degenebet fetch --sync` command pulls the latest `data` branch content
  from `origin` and materializes it into the local `cache_dir()`. This is a
  manual/on-demand step, not itself scheduled — staleness is bounded by "how
  long since the last sync," which is an accepted tradeoff for now.
- `DataAccess` always reads from `cache_dir()`; it has no knowledge of git,
  branches, or the fetch workflow at all. This keeps the read path exactly
  as simple as architecture-notes.md already specified.

### 3. Storage semantics: append vs. overwrite differ per source

The two sources are committed to the `data` branch as-is, and each already
has (or gets) a different storage shape — this sub-project doesn't invent a
new scheme, it makes explicit what each already does or needs:

- **`nflreadpy` cache** — opaque, managed entirely by `nflreadpy`'s own
  filesystem cache (`src/degenebet/data/nflverse.py` just wraps
  `nflreadpy.load_*`). Whatever that library does internally (in practice,
  refreshing/overwriting its own per-season files as more games complete) is
  committed to the `data` branch unmodified — this project doesn't manage
  that format. "Stored with a timestamp" for this source is satisfied by git
  itself: every scheduled-fetch commit is a timestamped snapshot in the
  branch's history, even though the *working tree* shows only the latest
  state. `DataAccess` never needs an old commit's version of this data —
  completed-game stats don't get revised, so the latest fetch is always a
  superset of any earlier one.
- **SharpAPI cache (`src/degenebet/data/cache.py`)** — already append-only,
  confirmed by reading the existing implementation: `load_or_fetch` writes a
  new `{source}_{pulled_at}.parquet` file per fetch and never deletes older
  ones; `_latest_snapshot` just picks the newest by filename sort. This
  sub-project keeps that behavior as-is rather than collapsing it to a
  single overwritten file — multiple retained snapshots are exactly what
  lets `DataAccess` pick "the snapshot closest to, but not after,
  `as_of_date`" for a given point in time, not just "the latest."
- **Growth**: daily cadence × one small odds snapshot/day is on the order of
  a few KB/day, trivial even over years — no pruning/retention policy for
  now. Revisit only if the `data` branch's size actually becomes a problem.

### Non-goals (deferred, not decided against)

- Automating the local sync trigger (a launchd/cron job that runs
  `degenebet fetch --sync` periodically) — confirmed deferred by explicit
  user decision (2026-09-17); a manual sync is acceptable for now.
- External cloud object storage (R2/S3/B2) — the `data` git branch covers
  this project's data volume at zero additional cost or infrastructure.
- Player-level data access (`get_player_data`) — `get_team_data`'s naming
  leaves room for it, but it isn't built now.

## Testing approach

Per `AGENTS.md`'s testing rules — real in-memory Polars DataFrames, no
network calls in the default test run:
- As-of clamp-and-warn behavior (`as_of_date` before earliest cached
  history).
- Dedup-on-`game_id` behavior (historical wins when both sources have the
  same game).
- "Current fills only the gap" behavior (current-source rows for games
  `historical` already has are dropped, not merged).
- The scheduled-fetch workflow and `--sync` command are integration/CI
  concerns, not unit-testable against fixtures; they get verified by running
  them for real (workflow dispatch / manual `fetch --sync` run) rather than
  a pytest case.
