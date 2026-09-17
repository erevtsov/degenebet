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
