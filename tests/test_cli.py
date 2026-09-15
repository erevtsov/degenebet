from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from typer.testing import CliRunner

from degenebet.cli import app

runner = CliRunner()


def test_fetch_schedules_reports_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame({"season": [2023, 2024]})
    monkeypatch.setattr("degenebet.cli.data.load_schedules", lambda seasons: frame)

    result = runner.invoke(app, ["fetch", "schedules"])

    assert result.exit_code == 0
    assert "2 games loaded" in result.stdout


def test_fetch_schedules_parses_seasons_option(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake(seasons: list[int] | None) -> pl.DataFrame:
        captured["seasons"] = seasons
        return pl.DataFrame({"season": [2022]})

    monkeypatch.setattr("degenebet.cli.data.load_schedules", fake)

    runner.invoke(app, ["fetch", "schedules", "--seasons", "2021,2022"])

    assert captured["seasons"] == [2021, 2022]


def test_fetch_player_stats_reports_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame({"season": [2024]})
    monkeypatch.setattr("degenebet.cli.data.load_player_stats", lambda seasons: frame)

    result = runner.invoke(app, ["fetch", "player-stats"])

    assert result.exit_code == 0
    assert "1 rows loaded" in result.stdout


def test_fetch_team_stats_reports_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame({"season": [2024]})
    monkeypatch.setattr("degenebet.cli.data.load_team_stats", lambda seasons: frame)

    result = runner.invoke(app, ["fetch", "team-stats"])

    assert result.exit_code == 0
    assert "1 rows loaded" in result.stdout


def test_fetch_rosters_reports_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame({"season": [2024]})
    monkeypatch.setattr("degenebet.cli.data.load_rosters", lambda seasons: frame)

    result = runner.invoke(app, ["fetch", "rosters"])

    assert result.exit_code == 0
    assert "1 rows loaded" in result.stdout


def test_fetch_odds_reports_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame(
        {
            "sportsbook": ["draftkings", "fanduel"],
            "pulled_at": [datetime.now(UTC), datetime.now(UTC)],
        }
    )
    monkeypatch.setattr("degenebet.cli.data.load_odds", lambda force_refresh: frame)

    result = runner.invoke(app, ["fetch", "odds"])

    assert result.exit_code == 0
    assert "2 odds rows from 2 sportsbook" in result.stdout


def test_fetch_odds_force_refresh_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake(force_refresh: bool) -> pl.DataFrame:
        captured["force_refresh"] = force_refresh
        return pl.DataFrame({"sportsbook": [], "pulled_at": []})

    monkeypatch.setattr("degenebet.cli.data.load_odds", fake)

    runner.invoke(app, ["fetch", "odds", "--force-refresh"])

    assert captured["force_refresh"] is True
