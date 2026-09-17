from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from degenebet.data import nflverse_cache
from degenebet.data.access import DataAccess, NflverseSource, SharpApiSource


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


def test_fetch_returns_one_row_per_snapshot_not_collapsed_across_snapshots(
    tmp_path: object,
) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "sharpapi"
    path.mkdir(parents=True, exist_ok=True)
    t1 = datetime(2026, 9, 15, tzinfo=UTC)
    t2 = datetime(2026, 9, 17, tzinfo=UTC)
    pl.DataFrame([_spread_row(line=-2.5, pulled_at=t1)]).write_parquet(
        path / "sharpapi_20260915T000000Z.parquet"
    )
    pl.DataFrame([_spread_row(line=-7.5, pulled_at=t2)]).write_parquet(
        path / "sharpapi_20260917T000000Z.parquet"
    )

    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    assert result.height == 2
    assert sorted(result["pulled_at"].to_list()) == [t1, t2]
    assert sorted(result["spread_line"].to_list()) == [pytest.approx(2.5), pytest.approx(7.5)]


def test_fetch_uses_eastern_gameday_for_evening_kickoff(tmp_path: object) -> None:
    # 2026-09-19T00:20:00Z is 2026-09-18 8:20pm ET -- a Thursday night game.
    _write_snapshot(tmp_path, [_spread_row(event_start_time="2026-09-19T00:20:00Z")])

    result = SharpApiSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    assert result["gameday"][0] == "2026-09-18"


def test_sharpapi_crosswalk_covers_every_canonical_team() -> None:
    from degenebet.data import access, teams

    teams.assert_maps_to_canonical_teams(access._SHARPAPI_TEAM_CROSSWALK)  # does not raise


def _schedule_row(**overrides: object) -> dict[str, object]:
    """A DataSource.fetch() row already in canonical form (lowercase team
    codes) -- used to build _FakeSource fixtures standing in for a
    DataSource's output, not for raw merged-store content."""
    row: dict[str, object] = {
        "game_id": "2026_02_MIN_CHI",
        "season": 2026,
        "week": 2,
        "gameday": "2026-09-20",
        "home_team": "chi",
        "away_team": "min",
        "result": None,
        "spread_line": None,
    }
    row.update(overrides)
    return row


def _raw_nflverse_schedule_row(**overrides: object) -> dict[str, object]:
    """A row as nflreadpy actually returns it (uppercase team codes) --
    used only to test NflverseSource's own raw-to-canonical transform."""
    row: dict[str, object] = {
        "game_id": "2026_02_MIN_CHI",
        "season": 2026,
        "week": 2,
        "gameday": "2026-09-20",
        "home_team": "CHI",
        "away_team": "MIN",
        "result": None,
        "spread_line": None,
    }
    row.update(overrides)
    return row


class _FakeSource:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame

    def fetch(self, start_date: date, end_date: date) -> pl.DataFrame:
        return self.frame


def test_nflverse_source_reads_merged_store_filtered_to_range() -> None:
    nflverse_cache.load_or_merge(
        pl.DataFrame(
            [
                _raw_nflverse_schedule_row(gameday="2026-09-06"),
                _raw_nflverse_schedule_row(gameday="2026-10-06"),
            ]
        ),
        name="schedules",
        key_columns=["game_id"],
        group_column="season",
    )

    result = NflverseSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    assert result.height == 1
    assert result["gameday"][0] == "2026-09-06"


def test_nflverse_source_lowercases_team_codes() -> None:
    nflverse_cache.load_or_merge(
        pl.DataFrame([_raw_nflverse_schedule_row()]),
        name="schedules",
        key_columns=["game_id"],
        group_column="season",
    )

    result = NflverseSource().fetch(date(2026, 9, 1), date(2026, 9, 30))

    assert result["home_team"][0] == "chi"
    assert result["away_team"][0] == "min"


def test_get_team_data_prefers_current_for_unplayed_game_even_with_stale_historical_line() -> None:
    historical = _FakeSource(pl.DataFrame([_schedule_row(result=None, spread_line=1.0)]))
    current = _FakeSource(
        pl.DataFrame(
            {
                "home_team": ["chi"],
                "away_team": ["min"],
                "gameday": ["2026-09-20"],
                "spread_line": [2.5],
                "pulled_at": [datetime(2026, 9, 17, tzinfo=UTC)],
            }
        )
    )

    result = DataAccess(historical, current).get_team_data(
        date(2026, 9, 1), date(2026, 9, 30), as_of_date=date(2026, 9, 17)
    )

    assert result["spread_line"][0] == pytest.approx(2.5)


def test_get_team_data_historical_wins_for_settled_result() -> None:
    historical = _FakeSource(pl.DataFrame([_schedule_row(result=7, spread_line=1.0)]))
    current = _FakeSource(
        pl.DataFrame(
            {
                "home_team": ["chi"],
                "away_team": ["min"],
                "gameday": ["2026-09-20"],
                "spread_line": [2.5],
                "pulled_at": [datetime(2026, 9, 17, tzinfo=UTC)],
            }
        )
    )

    result = DataAccess(historical, current).get_team_data(
        date(2026, 9, 1), date(2026, 9, 30), as_of_date=date(2026, 9, 17)
    )

    assert result["spread_line"][0] == pytest.approx(1.0)


def test_get_team_data_falls_back_to_historical_when_current_missing_game() -> None:
    historical = _FakeSource(pl.DataFrame([_schedule_row(result=None, spread_line=1.0)]))
    current = _FakeSource(
        pl.DataFrame(
            schema={
                "home_team": pl.Utf8,
                "away_team": pl.Utf8,
                "gameday": pl.Utf8,
                "spread_line": pl.Float64,
                "pulled_at": pl.Datetime(time_zone="UTC"),
            }
        )
    )

    result = DataAccess(historical, current).get_team_data(
        date(2026, 9, 1), date(2026, 9, 30), as_of_date=date(2026, 9, 17)
    )

    assert result["spread_line"][0] == pytest.approx(1.0)


def test_get_team_data_warns_when_as_of_before_earliest_history() -> None:
    historical = _FakeSource(
        pl.DataFrame([_schedule_row(gameday="2026-09-06", result=None, spread_line=1.0)])
    )
    # A current snapshot pulled well after the (very early) as_of_date --
    # under the old clamp-forward behavior, as_of_date would be moved to
    # "2026-09-06" (historical's earliest gameday), which is >= this
    # pulled_at, so it would wrongly win over historical's own line. With
    # clamping removed, the original as_of_date (2020-01-01) is used as the
    # cutoff, this snapshot is excluded, and historical's line is the
    # correct, honest answer.
    current = _FakeSource(
        pl.DataFrame(
            {
                "home_team": ["chi"],
                "away_team": ["min"],
                "gameday": ["2026-09-06"],
                "spread_line": [99.0],
                "pulled_at": [datetime(2026, 9, 5, tzinfo=UTC)],
            }
        )
    )

    with pytest.warns(UserWarning, match="predates earliest cached history"):
        result = DataAccess(historical, current).get_team_data(
            date(2026, 9, 1), date(2026, 9, 30), as_of_date=date(2020, 1, 1)
        )

    assert result["spread_line"][0] == pytest.approx(1.0)


def test_get_team_data_selects_snapshot_closest_to_but_not_after_as_of_date(
    tmp_path: object,
) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "sharpapi"
    path.mkdir(parents=True, exist_ok=True)
    t1 = datetime(2026, 9, 15, tzinfo=UTC)
    t2 = datetime(2026, 9, 17, tzinfo=UTC)
    pl.DataFrame([_spread_row(line=-2.5, pulled_at=t1)]).write_parquet(
        path / "sharpapi_20260915T000000Z.parquet"
    )
    pl.DataFrame([_spread_row(line=-7.5, pulled_at=t2)]).write_parquet(
        path / "sharpapi_20260917T000000Z.parquet"
    )
    historical = _FakeSource(pl.DataFrame([_schedule_row(result=None, spread_line=None)]))
    current = SharpApiSource()

    between_t1_and_t2 = DataAccess(historical, current).get_team_data(
        date(2026, 9, 1), date(2026, 9, 30), as_of_date=date(2026, 9, 16)
    )
    at_or_after_t2 = DataAccess(historical, current).get_team_data(
        date(2026, 9, 1), date(2026, 9, 30), as_of_date=date(2026, 9, 18)
    )

    assert between_t1_and_t2["spread_line"][0] == pytest.approx(2.5)
    assert at_or_after_t2["spread_line"][0] == pytest.approx(7.5)


def test_get_team_data_end_to_end_with_real_sources(tmp_path: object) -> None:
    """Exercises the real NflverseSource + SharpApiSource + DataAccess chain
    together -- every other DataAccess test uses a fake DataSource, so this
    is the only test that would have caught C1/C2/C3 (wrong archive path,
    look-ahead-unsafe snapshot blending, UTC-vs-Eastern gameday mismatch)."""
    nflverse_cache.load_or_merge(
        pl.DataFrame(
            [
                _raw_nflverse_schedule_row(
                    game_id="2026_03_MIN_CHI",
                    week=3,
                    gameday="2026-09-18",
                    result=None,
                    spread_line=None,
                )
            ]
        ),
        name="schedules",
        key_columns=["game_id"],
        group_column="season",
    )
    _write_snapshot(
        tmp_path,
        [
            # 2026-09-19T00:20:00Z is 8:20pm ET on 2026-09-18 -- a Thursday
            # night kickoff, chosen deliberately to also cover the C3
            # timezone fix in this end-to-end path.
            _spread_row(
                home_team="Chicago Bears",
                away_team="Minnesota Vikings",
                line=-3.5,
                event_start_time="2026-09-19T00:20:00Z",
                pulled_at=datetime(2026, 9, 17, tzinfo=UTC),
            )
        ],
    )

    result = DataAccess(NflverseSource(), SharpApiSource()).get_team_data(
        date(2026, 9, 1), date(2026, 9, 30), as_of_date=date(2026, 9, 18)
    )

    row = result.filter(pl.col("game_id") == "2026_03_MIN_CHI")
    assert row.height == 1
    assert row["home_team"][0] == "chi"
    assert row["away_team"][0] == "min"
    assert row["spread_line"][0] == pytest.approx(3.5)
