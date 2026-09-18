from __future__ import annotations

import polars as pl
import pytest

from degenebet.data import nflverse_cache


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEGENEBET_CACHE_DIR", str(tmp_path))


def test_merge_frames_key_in_both_takes_new_value() -> None:
    existing = pl.DataFrame({"game_id": ["a"], "result": [10]})
    new = pl.DataFrame({"game_id": ["a"], "result": [14]})

    merged = nflverse_cache.merge_frames(existing, new, key_columns=["game_id"])

    assert merged["result"].to_list() == [14]


def test_merge_frames_key_only_in_existing_survives() -> None:
    existing = pl.DataFrame({"game_id": ["a", "b"], "result": [10, 20]})
    new = pl.DataFrame({"game_id": ["a"], "result": [14]})

    merged = nflverse_cache.merge_frames(existing, new, key_columns=["game_id"])

    assert sorted(merged["game_id"].to_list()) == ["a", "b"]
    assert merged.filter(pl.col("game_id") == "a")["result"][0] == 14
    assert merged.filter(pl.col("game_id") == "b")["result"][0] == 20


def test_merge_frames_key_only_in_new_is_added() -> None:
    existing = pl.DataFrame({"game_id": ["a"], "result": [10]})
    new = pl.DataFrame({"game_id": ["a", "c"], "result": [14, 30]})

    merged = nflverse_cache.merge_frames(existing, new, key_columns=["game_id"])

    assert sorted(merged["game_id"].to_list()) == ["a", "c"]


def test_merge_frames_existing_none_returns_new() -> None:
    new = pl.DataFrame({"game_id": ["a"], "result": [14]})

    merged = nflverse_cache.merge_frames(None, new, key_columns=["game_id"])

    assert merged.equals(new)


def test_read_merged_returns_none_when_absent() -> None:
    assert nflverse_cache.read_merged("schedules") is None


def test_load_or_merge_first_write_has_no_existing_file() -> None:
    new = pl.DataFrame({"game_id": ["a"], "season": [2024], "result": [10]})

    result = nflverse_cache.load_or_merge(
        new, name="schedules", key_columns=["game_id"], group_column="season"
    )

    assert result.equals(new)
    assert nflverse_cache.read_merged("schedules").equals(new)


def test_load_or_merge_persists_and_merges_on_second_call() -> None:
    first = pl.DataFrame({"game_id": ["a"], "season": [2024], "result": [5]})
    second = pl.DataFrame({"game_id": ["a", "b"], "season": [2024, 2024], "result": [10, 20]})

    nflverse_cache.load_or_merge(
        first, name="schedules", key_columns=["game_id"], group_column="season"
    )
    result = nflverse_cache.load_or_merge(
        second, name="schedules", key_columns=["game_id"], group_column="season"
    )

    assert sorted(result["game_id"].to_list()) == ["a", "b"]


def test_load_or_merge_raises_on_shrinkage() -> None:
    first = pl.DataFrame({"game_id": ["a", "b"], "season": [2023, 2024], "result": [10, 20]})
    nflverse_cache.load_or_merge(
        first, name="schedules", key_columns=["game_id"], group_column="season"
    )

    # "a" is reclassified from season 2023 to 2024 in the new fetch. merge_
    # frames doesn't drop the row (key "a" still exists, just under a new
    # season value), but season 2023's row count would drop from 1 to 0 --
    # exactly the shrinkage the guard exists to catch.
    reclassified = pl.DataFrame({"game_id": ["a"], "season": [2024], "result": [10]})

    with pytest.raises(ValueError, match="shrink"):
        nflverse_cache.load_or_merge(
            reclassified, name="schedules", key_columns=["game_id"], group_column="season"
        )


def test_load_or_merge_allows_growth() -> None:
    first = pl.DataFrame({"game_id": ["a"], "season": [2024], "result": [10]})
    grown = pl.DataFrame({"game_id": ["a", "b"], "season": [2024, 2024], "result": [10, 20]})

    nflverse_cache.load_or_merge(
        first, name="schedules", key_columns=["game_id"], group_column="season"
    )
    result = nflverse_cache.load_or_merge(
        grown, name="schedules", key_columns=["game_id"], group_column="season"
    )

    assert result.height == 2


def test_sync_schedules_fetches_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched = pl.DataFrame({"game_id": ["a"], "season": [2024], "result": [10]})
    monkeypatch.setattr(
        "degenebet.data.nflverse_cache.nflverse.load_schedules", lambda seasons: fetched
    )

    result = nflverse_cache.sync_schedules(seasons=[2024])

    assert result.equals(fetched)
    assert nflverse_cache.read_merged("schedules").equals(fetched)


def test_sync_team_stats_fetches_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched = pl.DataFrame({"game_id": ["a"], "team": ["BUF"], "season": [2024]})
    monkeypatch.setattr(
        "degenebet.data.nflverse_cache.nflverse.load_team_stats", lambda seasons: fetched
    )

    result = nflverse_cache.sync_team_stats(seasons=[2024])

    assert result.equals(fetched)
    assert nflverse_cache.read_merged("team_stats").equals(fetched)
