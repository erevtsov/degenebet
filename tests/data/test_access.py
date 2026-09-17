from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from degenebet.data.access import SharpApiSource


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEGENEBET_CACHE_DIR", str(tmp_path))


def _write_snapshot(tmp_path: object, rows: list[dict[str, object]]) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "sharpapi"
    path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path / "sharpapi_20260917T000000Z.parquet")


def _spread_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": "e1",
        "home_team": "Chicago Bears",
        "away_team": "Minnesota Vikings",
        "market_type": "spread",
        "selection_type": "home",
        "line": -2.5,
        "event_start_time": "2026-09-20T17:00:00Z",
        "pulled_at": datetime(2026, 9, 17, tzinfo=UTC),
    }
    row.update(overrides)
    return row


def test_fetch_converts_sign_home_favored(tmp_path: object) -> None:
    _write_snapshot(tmp_path, [_spread_row(line=-2.5)])

    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    row = result.filter((pl.col("home_team") == "chi") & (pl.col("away_team") == "min"))
    assert row["spread_line"][0] == pytest.approx(2.5)


def test_fetch_converts_sign_away_favored(tmp_path: object) -> None:
    _write_snapshot(tmp_path, [_spread_row(line=3.0)])

    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    row = result.filter((pl.col("home_team") == "chi") & (pl.col("away_team") == "min"))
    assert row["spread_line"][0] == pytest.approx(-3.0)


def test_fetch_takes_median_across_sportsbooks(tmp_path: object) -> None:
    _write_snapshot(
        tmp_path,
        [
            _spread_row(line=-2.5, event_id="e1"),
            _spread_row(line=-3.0, event_id="e1"),
            _spread_row(line=-1.0, event_id="e1"),
        ],
    )

    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    row = result.filter((pl.col("home_team") == "chi") & (pl.col("away_team") == "min"))
    assert row["spread_line"][0] == pytest.approx(2.5)  # median of 2.5/3.0/1.0


def test_fetch_ignores_non_spread_and_away_selection_rows(tmp_path: object) -> None:
    _write_snapshot(
        tmp_path,
        [
            _spread_row(),
            _spread_row(market_type="moneyline", selection_type="home", line=None),
            _spread_row(selection_type="away", line=2.5),
        ],
    )

    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    assert result.height == 1


def test_fetch_filters_to_date_range(tmp_path: object) -> None:
    _write_snapshot(tmp_path, [_spread_row(event_start_time="2026-10-15T17:00:00Z")])

    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    assert result.height == 0


def test_fetch_raises_on_unknown_team_name(tmp_path: object) -> None:
    _write_snapshot(tmp_path, [_spread_row(home_team="Springfield Isotopes")])

    with pytest.raises(Exception):  # noqa: B017 -- polars' replace_strict error type
        SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))


def test_fetch_returns_empty_frame_when_nothing_cached() -> None:
    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    assert result.height == 0


def test_sharpapi_crosswalk_covers_every_canonical_team() -> None:
    from degenebet.data import access, teams

    teams.assert_maps_to_canonical_teams(access._SHARPAPI_TEAM_CROSSWALK)  # does not raise
