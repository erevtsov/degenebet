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
  data-foundation sub-project is I/O and parsing, not modeling math. Revisit
  once the modeling sub-project exists.
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
  authoritative gate, states CI runs the identical command, and notes there
  are no golden fixtures or scheduled automation yet (so a future addition
  of either updates this section, rather than the omission looking like an
  oversight).

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

This lands as its own PR against `master`, before the open `data-foundation`
PR merges — so branch protection and the CI gate are already active by the
time the first real code PR lands, and that PR gets to prove the gate works
end-to-end (per user decision, 2026-09-15). Concretely: a new branch/worktree
off `master` carries the `justfile`, `.github/workflows/ci.yml`,
`AGENTS.md`/`CLAUDE.md` rename, and the one-time `ruff format` normalization
pass; branch protection itself is a repo-settings change (via `gh api`),
applied once CI has run green at least once on this new branch's PR (a
required status check can't be required before it exists).
