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

**Verified against real data (2026 season, checked 2026-09-17), not
assumed:** `nflreadpy.load_schedules()` already lists every future game
(`game_id`/`home_team`/`away_team`/`week`) well before kickoff — of 256
unplayed games, all had the schedule skeleton, but only the next week's 10
games had `spread_line` populated (32 of 256 total). So `nflreadpy` and
SharpAPI can both have a line for the *same* unplayed game near kickoff —
this isn't a clean "historical has it or it doesn't" split, which changes
the dedup rule below from what the first draft of this spec assumed.

`DataAccess.get_team_data` therefore stitches at the schedule level:
- `historical` (`NflverseSource`) supplies every game up to `as_of_date`
  that `nflreadpy` already has, `result`/`spread_line` included where known.
- `current` (`SharpApiSource`) supplies a `spread_line` for any game
  `historical` hasn't settled yet, from the odds snapshot closest to (but
  not after) `as_of_date`.
- **Dedup is on `result`, not on `game_id` presence:** if `historical` has a
  settled `result` for a `game_id`, its `spread_line` is authoritative (it's
  the real closing line). If `result` is null — the game hasn't been played,
  regardless of whether `historical` happens to already carry an early
  `spread_line` for it — `current`'s line wins when SharpAPI has that game,
  since SharpAPI exists specifically to be the fresher, more current number
  to actually bet against; `historical`'s early line is only a fallback for
  a game SharpAPI doesn't cover.

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

- **`nflreadpy` cache** — **not** a bare pass-through of `nflreadpy`'s own
  internal cache. Relying on "the latest fetch is always a superset of any
  earlier one" is an assumption, not a guarantee: if the upstream API ever
  starts limiting the historical range it returns, hits a rate limit
  mid-pull, or returns a transient partial response, a naive overwrite would
  silently commit that shrinkage to the `data` branch — recoverable only by
  manually digging through git history, which a scheduled unattended job
  won't do on its own.

  Instead, `degenebet fetch` **merges** each fetch into our own persisted
  parquet store for these tables (a new small module alongside
  `data/cache.py`, since `nflreadpy`'s own cache format is opaque and not
  something we control): union the new fetch with the existing cached file,
  keyed by natural identity (`game_id` for schedules; `(game_id, team)` for
  team_stats). A key present in both keeps the **new** value (so legitimate
  `nflverse` corrections still propagate); a key present in the old cache
  but **missing** from the new fetch keeps the **old** value rather than
  disappearing. This makes the cache monotonically non-shrinking regardless
  of what the upstream API decides to return on any given fetch.

  As defense in depth on top of the merge (which already prevents data
  loss), the fetch also compares the new pull's row count against the
  existing cache per season/table and **fails loudly — no commit — if it
  drops**. This doesn't protect against loss (the merge already does that);
  it protects against a shrinking upstream response going unnoticed, since a
  sustained range limit would otherwise just quietly stop contributing
  new-key overwrites without ever surfacing as something to investigate.

  "Stored with a timestamp" for this source is satisfied by git itself:
  every scheduled-fetch commit is a timestamped snapshot in the branch's
  history.
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

- **Handling legitimate upstream deletions** (e.g. `nflverse` retracting a
  duplicate or erroneous row entirely, not correcting its values) — the
  merge is keep-old-if-missing-from-new by design, so a genuine deletion
  upstream would not propagate. Not building reconciliation for this now;
  it's a rare case and the row-count-drop guard would at least surface it
  for manual review rather than silently diverging.

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
- Result-based dedup: `historical` wins for a `game_id` with a settled
  `result`; `current` wins for a null-`result` `game_id` when SharpAPI has
  that game — including the case where `historical` *also* has an early
  `spread_line` for that same unplayed game, to guard against regressing
  back to game_id-presence dedup.
- `historical`-fallback behavior: a null-`result` game `current` doesn't
  cover falls back to `historical`'s line rather than being dropped.
- Merge-cache upsert behavior: a key in both old and new takes the new
  value; a key only in the old cache survives; a key only in the new fetch
  is added. This is the core correctness property this whole design exists
  for — needs direct unit coverage, not just exercised incidentally through
  `get_team_data`.
- Row-count-drop guard: fetch raises/refuses to commit when a season's
  merged row count would be lower than the existing cache's for that
  season.
- The scheduled-fetch workflow and `--sync` command are integration/CI
  concerns, not unit-testable against fixtures; they get verified by running
  them for real (workflow dispatch / manual `fetch --sync` run) rather than
  a pytest case.
