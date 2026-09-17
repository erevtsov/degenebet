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
    assert calls[1] == ["git", "archive", "--format=tar", "origin/data", "cache"]
    assert calls[2] == [
        "tar",
        "-x",
        "--strip-components=1",
        "-C",
        str(Path(str(tmp_path))),
    ]


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
    assert calls[1] == [
        "git",
        "archive",
        "--format=tar",
        "upstream/scheduled-data",
        "cache",
    ]
