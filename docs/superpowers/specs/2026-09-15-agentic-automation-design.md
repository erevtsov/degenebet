# Agentic automation for degenebet — design

Date: 2026-09-15
Status: Approved for implementation planning

## Context

The sibling project `waypoint` (`/Users/erevtsov/dev/waypoint`) already has this
worked out and running: a `justfile` + GitHub Actions verification loop, an
`AGENTS.md` convention, and branch protection on `main` (verified live via
`gh api repos/erevtsov/waypoint/branches/main/protection`). It also has a
weekly scheduled maintenance agent, golden regression fixtures, and property
tests — specified in waypoint's own
`docs/superpowers/specs/2026-09-14-agentic-automation-design.md`, adapting a
personal survey on running AI coding agents safely and productively down from
a ~10-person GitLab/compliance context to a solo-maintained project.

This spec ports the parts of that pattern that fit degenebet today, and
explicitly defers the parts that don't yet.

## Goal

Before any more code lands on `master`: every change — human- or
agent-authored — passes one command, run identically locally and in CI,
before it can merge. No direct pushes to `master`, even from the repo owner.

## Non-goals (deferred, not decided against)

- **Golden regression fixtures / property-based tests.** waypoint pins
  numeric output for its analytics surfaces because it has core numeric
  logic worth protecting from silent drift. degenebet doesn't yet — the
  data-foundation sub-project is I/O and parsing, not modeling math. This is
  not a "someday" deferral: it is a **required part of the modeling
  sub-project's own spec**, not an optional add-on discovered after the
  fact. See section 2 below — the guidance goes into `AGENTS.md` now, so the
  agent brainstorming that sub-project's spec sees it before scoping the
  work, the same way this project's own CLAUDE.md/AGENTS.md rules get read
  before any brainstorming session starts.
- **Weekly scheduled maintenance agent.** Confirmed by explicit user
  decision (2026-09-15): not being set up in this pass. Worth noting the
  scheduled agent's job would look different for degenebet than for
  waypoint if it's built later — waypoint is a library that deliberately
  doesn't commit `uv.lock`, so its maintenance is version-floor nudges in
  `pyproject.toml`; degenebet is an application that does commit `uv.lock`
  (see data-foundation spec, Global Constraints), so the equivalent job
  would be a lockfile-bump PR (closer to Dependabot/Renovate's role), not a
  floor bump.
- Cost/token ceilings, eval-task suites, ADR logs — team-scale concerns,
  not relevant at this size either.

## 1. Verification loop

**`justfile`** at repo root:

```
check:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/
    uv run mypy
    uv run pytest
```

Same shape as waypoint's. One addition relative to the data-foundation
spec's tooling section: `ruff format --check` wasn't part of that spec (only
`ruff check` was), so this is a new, slightly stricter bar. The existing
data-foundation code (already merged in the `data-foundation` PR by the time
this lands) needs a one-time `uv run ruff format src/ tests/` pass as part
of implementing this — otherwise CI's first run fails on pre-existing
formatting, not anything this spec changed.

**`.github/workflows/ci.yml`** — triggers on push to `master` and on PRs
targeting `master` (waypoint uses `main`; degenebet's default branch is
`master` — set this to match, not copy waypoint's branch name literally):

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - uses: extractions/setup-just@v2
      - run: uv sync --extra dev
      - run: just check
```

This repo has no CI at all yet, so — same as waypoint's framing — this
workflow is what makes the gate real rather than aspirational.

## 2. `AGENTS.md` convention

Rename `CLAUDE.md` to `AGENTS.md` (content unchanged otherwise), and add
`CLAUDE.md` back as a symlink to `AGENTS.md` so Claude Code keeps reading it
under either name. Add one new section, mirroring waypoint's:

- **Automation & Verification** — documents `just check` as the sole
  authoritative gate, states CI runs the identical command, and states two
  things explicitly rather than leaving them as silent gaps:
  - No scheduled automation yet (see the deferred non-goal above).
  - **No golden regression fixtures or property-based tests yet — and the
    modeling sub-project must add them before or alongside its first
    predictive model, not after.** Concretely: pin numeric output for each
    model/analytic surface against a deterministic fixture (mirroring
    waypoint's `tests/analysis/_golden_portfolio.py` +
    `test_*_golden.py` pattern — one fixture family per surface, tight
    numeric tolerance not exact equality, updates are a deliberate reviewed
    act never a reflexive fix for a failing test), and add `hypothesis`
    property tests for the invariants that would be expensive to get
    silently wrong (e.g., predicted probabilities lie in `[0, 1]` and sum
    to 1 across a market's outcomes; backtest P&L reconciles under
    re-aggregation over a date range). This line is the enforcement
    mechanism for the "non-goal, not decided against" framing above: an
    agent brainstorming the modeling sub-project's spec reads this file
    first and scopes the work to include it.

Not adding a "Scheduled Task Scope" section — nothing scheduled exists yet
to scope. Add it when the scheduled-maintenance non-goal above gets
revisited.

## 3. Review/merge gate

GitHub branch protection on `master`, matching waypoint's configuration
exactly:
- Require the CI status check (`check`) to pass before merge, strict mode
  (branch must be up to date).
- Require a pull request — no direct pushes, including from the repo owner
  (`enforce_admins: true`).
- No force pushes, no deletions.
- 0 required approving reviews (solo-maintained, same as waypoint) — the PR
  requirement plus the CI gate are the actual guardrail, not a review
  headcount.

## Sequencing

Corrected 2026-09-15: `master` currently has no Python project on it at all
(no `pyproject.toml`, `src/`, or `tests/` — those exist only on the not-yet-
merged `data-foundation` branch). A CI workflow with nothing to check can
never go green, so branch protection could never be turned on if this
landed first. Order is therefore:

1. Merge the already-reviewed `data-foundation` PR to `master` first — this
   puts a real `pyproject.toml`, `src/`, `tests/`, and `CLAUDE.md` on
   `master` for CI to actually check.
2. Then open this spec's guardrails as their own PR on top: a new
   branch/worktree off `master` carries the `justfile`,
   `.github/workflows/ci.yml`, the `CLAUDE.md` → `AGENTS.md` rename (plain
   rename now — `CLAUDE.md` already exists on `master` post-step-1, so this
   is a `git mv` plus adding the symlink back and the new "Automation &
   Verification" section content, not a merge/rebase reconciliation), and
   the one-time `ruff format` normalization pass.
3. Branch protection is a repo-settings change (via `gh api`), applied once
   this second PR's CI has run green at least once (a required status check
   can't be required before it exists).
