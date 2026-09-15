# Agentic Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give degenebet the same verification-loop + AGENTS.md + branch-protection guardrails that waypoint already runs, adapted for degenebet's current size (no golden fixtures/property tests yet — those are deferred to the modeling sub-project) and scope (no scheduled maintenance agent yet).

**Architecture:** A `justfile` with one `check` recipe wraps the existing lint/format/typecheck/test commands into a single authoritative gate; a GitHub Actions workflow runs that same recipe on every push/PR to `master`; `CLAUDE.md` becomes `AGENTS.md` (with a symlink back) carrying a new "Automation & Verification" section that documents the gate and — critically — commits the modeling sub-project to adding golden fixtures/property tests. Branch protection on `master` is a repo-settings change applied after this PR's CI has run green once, not a file in the repo.

**Tech Stack:** `just` (command runner), GitHub Actions, existing `uv`/ruff/mypy/pytest toolchain (no new Python dependencies).

**Spec:** `docs/superpowers/specs/2026-09-15-agentic-automation-design.md`

## Global Constraints

- `justfile`'s `check` recipe, in this exact order, fail-fast: `uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/`, `uv run mypy`, `uv run pytest`.
- CI workflow triggers on `push` to `master` and `pull_request` targeting `master` (not `main` — degenebet's default branch is `master`); installs `uv` (`astral-sh/setup-uv@v3`) and `just` (`extractions/setup-just@v2`); runs `uv sync --extra dev` then `just check`.
- `CLAUDE.md` → `AGENTS.md`: a plain rename (`git mv`), not a merge — `data-foundation` is already merged to `master`, so there's exactly one `CLAUDE.md` to rename, no reconciliation needed. `CLAUDE.md` is recreated as a symlink to `AGENTS.md` so Claude Code keeps reading it.
- `ruff format --check` is a new bar the existing code hasn't been held to — a one-time `uv run ruff format src/ tests/` normalization pass is required before committing the justfile, or CI's first run fails on pre-existing formatting rather than anything this plan changes.
- Branch protection (required status check `check`, strict mode; PR required including for the repo owner via `enforce_admins: true`; no force-push; no deletions; 0 required approving reviews) is applied via `gh api` after this plan's PR has run CI green at least once — this is a post-implementation step the controller performs directly, not a task a subagent implements or tests.
- No golden regression fixtures, property-based tests, or scheduled maintenance agent in this plan — explicitly deferred (see spec's Non-goals). The AGENTS.md content added here must still commit the *future* modeling sub-project to adding golden fixtures/property tests, in the exact terms the spec's section 2 specifies.

---

## File Structure

```
degenebet/
  justfile                       # new
  .github/workflows/ci.yml        # new
  AGENTS.md                        # renamed from CLAUDE.md, + new section
  CLAUDE.md                         # replaced: was a file, now a symlink -> AGENTS.md
  src/degenebet/**, tests/**         # unchanged in content, reformatted only if ruff format finds drift
```

---

### Task 1: Verification loop (justfile + CI workflow)

**Files:**
- Create: `justfile`
- Create: `.github/workflows/ci.yml`
- Modify: any file under `src/` or `tests/` that `uv run ruff format` reformats (unknown until run — likely none or few, since the code was written under `ruff check` already)

**Interfaces:**
- Consumes: nothing new — wraps the existing `uv run ruff check`, `uv run mypy`, `uv run pytest` commands already documented in `CLAUDE.md`'s Commands section.
- Produces: `just check` as a runnable local command; a `check` CI job GitHub Actions can report status for. Task 2 and the post-implementation branch-protection step both depend on this job being named `check` (the `jobs.check` key in the workflow YAML) — do not rename it.

- [ ] **Step 1: Normalize formatting first, before anything depends on `ruff format --check` passing**

Run: `uv run ruff format src/ tests/`
Then: `git status --porcelain` — note whether this reformatted anything. If it did, those files are part of this task's diff; if not, nothing to stage here.

- [ ] **Step 2: Create `justfile`**

```
# Run the full verification suite: lint, format check, type-check, tests.
check:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/
    uv run mypy
    uv run pytest
```

- [ ] **Step 3: Verify `just check` passes locally**

Run: `just check`
Expected: all four steps pass, ending in pytest's coverage summary (same as the existing `uv run pytest` output) with exit code 0. If `ruff format --check` fails here, Step 1 wasn't run first (or missed a file) — go back and rerun `uv run ruff format src/ tests/`, do not skip past a red `just check`.

- [ ] **Step 4: Create `.github/workflows/ci.yml`**

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

- [ ] **Step 5: Commit**

```bash
git add justfile .github/workflows/ci.yml
# Add any files Step 1 reformatted, e.g.: git add src/ tests/
git commit -m "chore: add justfile and CI verification loop"
```

---

### Task 2: `AGENTS.md` convention

**Files:**
- Modify (rename): `CLAUDE.md` → `AGENTS.md` (via `git mv`)
- Create: `CLAUDE.md` (new content: a symlink to `AGENTS.md`)

**Interfaces:**
- Consumes: nothing from Task 1 directly, but references `just check` and `.github/workflows/ci.yml` by name in its new prose — must match Task 1's actual filenames/recipe name exactly.
- Produces: `AGENTS.md` as the canonical project-instructions file; `CLAUDE.md` as a working symlink to it. No other task depends on this one.

- [ ] **Step 1: Rename the file**

Run: `git mv CLAUDE.md AGENTS.md`

- [ ] **Step 2: Update the `## Commands` section in `AGENTS.md`**

Insert one new bullet immediately after the `uv sync --extra dev` line (exact wording, matching waypoint's phrasing):

```markdown
- `just check` — the authoritative verification gate: ruff lint, ruff format check, mypy, pytest (identical to what CI runs on every PR)
```

So the section reads:

```markdown
## Commands
- `uv sync --extra dev` — install all dev dependencies
- `just check` — the authoritative verification gate: ruff lint, ruff format check, mypy, pytest (identical to what CI runs on every PR)
- `uv run pytest` — run tests with coverage
- `uv run ruff check src/ tests/` — lint
- `uv run mypy` — type-check
- `uv run degenebet fetch <table>` — warm the local cache
```

- [ ] **Step 3: Insert a new `## Automation & Verification` section**

Insert this new section immediately after `## Commands` and before `## Architecture`:

```markdown
## Automation & Verification
- `just check` is the single CI-authoritative gate — `.github/workflows/ci.yml` runs this exact command, nothing more, nothing less. Locally and in CI it always means: `ruff check src/ tests/`, `ruff format --check src/ tests/`, `mypy`, `pytest`, in that order, fail-fast.
- `master` is branch-protected: the `check` CI status must be green and every change must go through a pull request, with no exception for the repo owner — there is no direct-push path around the gate.
- No scheduled maintenance automation yet — see `docs/superpowers/specs/2026-09-15-agentic-automation-design.md` for the deferred design and why.
- No golden regression fixtures or property-based tests yet, and this is not optional to skip: **the modeling sub-project must add them before or alongside its first predictive model.** Pin numeric output for each model/analytic surface against a deterministic fixture (one fixture family per surface, tight numeric tolerance not exact equality, updates are a deliberate reviewed act never a reflexive fix for a failing test), and add `hypothesis` property tests for invariants that would be expensive to get silently wrong (e.g. predicted probabilities lie in `[0, 1]` and sum to 1 across a market's outcomes; backtest P&L reconciles under re-aggregation over a date range).
```

- [ ] **Step 4: Recreate `CLAUDE.md` as a symlink**

Run: `ln -s AGENTS.md CLAUDE.md`

- [ ] **Step 5: Verify the symlink resolves and content is intact**

Run: `cat CLAUDE.md | head -5`
Expected: prints `AGENTS.md`'s actual content (starting with `# degenebet`), proving the symlink resolves — not an error and not literal symlink-target text.

Run: `git status --porcelain`
Expected: shows `AGENTS.md` (renamed from `CLAUDE.md`, with modifications) and `CLAUDE.md` (new symlink) — no other files touched.

- [ ] **Step 6: Commit**

```bash
git add AGENTS.md CLAUDE.md
git commit -m "docs: rename CLAUDE.md to AGENTS.md, add automation section"
```

---

## Post-Implementation (controller performs directly — not a subagent task)

Once both tasks are reviewed and merged into this plan's branch:

1. Push the branch and open a PR against `master` (per the spec's Sequencing section — CI needs to actually run on a real PR before branch protection can require its status check).
2. Wait for the `check` CI job to complete on the PR and confirm it's green. If it's red, fix forward on the branch before proceeding — do not apply branch protection against a red or not-yet-run check.
3. Apply branch protection to `master`:
   ```bash
   gh api repos/erevtsov/degenebet/branches/master/protection -X PUT --input - <<'EOF'
   {
     "required_status_checks": {"strict": true, "contexts": ["check"]},
     "enforce_admins": true,
     "required_pull_request_reviews": {"required_approving_review_count": 0},
     "restrictions": null,
     "allow_force_pushes": false,
     "allow_deletions": false
   }
   EOF
   ```
4. Verify: `gh api repos/erevtsov/degenebet/branches/master/protection` and confirm the response matches (required check `check`, `enforce_admins.enabled: true`, `allow_force_pushes.enabled: false`, `allow_deletions.enabled: false`).
5. Merge the PR (same finishing-a-development-branch flow as `data-foundation`) — this will now go through the newly-active gate itself, proving it end-to-end.

## Self-Review Notes

- **Spec coverage:** verification loop (Task 1) ✅, AGENTS.md convention including the golden-fixtures/property-tests commitment (Task 2) ✅, branch protection (Post-Implementation section, since it's not file-based/testable code) ✅. Non-goals (golden fixtures, property tests, scheduled agent) correctly have no task — only a documentation commitment in Task 2 to require them later.
- **Placeholder scan:** no TBD/TODO; every step has literal file content or an exact command.
- **Type consistency:** the CI job name `check` (Task 1, Step 4: `jobs.check`) matches the branch-protection `contexts: ["check"]` (Post-Implementation, Step 3) and AGENTS.md's prose (Task 2, Step 3: "the `check` CI status"). The `justfile` recipe name `check` (Task 1, Step 2) matches what both `.github/workflows/ci.yml` (Task 1, Step 4: `run: just check`) and AGENTS.md (Task 2, Steps 2–3) reference.
