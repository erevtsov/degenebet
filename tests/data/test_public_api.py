from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

import degenebet.data as data


def test_load_schedules_is_nflverse_load_schedules() -> None:
    assert data.load_schedules is data.nflverse.load_schedules


def test_load_player_stats_is_nflverse_load_player_stats() -> None:
    assert data.load_player_stats is data.nflverse.load_player_stats


def test_load_team_stats_is_nflverse_load_team_stats() -> None:
    assert data.load_team_stats is data.nflverse.load_team_stats


def test_load_rosters_is_nflverse_load_rosters() -> None:
    assert data.load_rosters is data.nflverse.load_rosters


def test_load_odds_delegates_to_cache_with_sharpapi_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_load_or_fetch(
        provider: object,
        source: str,
        *,
        max_age: timedelta,
        force_refresh: bool,
    ) -> pl.DataFrame:
        captured["source"] = source
        captured["max_age"] = max_age
        captured["force_refresh"] = force_refresh
        return pl.DataFrame({"a": [1]})

    monkeypatch.setattr(data.cache, "load_or_fetch", fake_load_or_fetch)

    result = data.load_odds(force_refresh=True)

    assert captured["source"] == "sharpapi"
    assert captured["force_refresh"] is True
    assert result.height == 1
