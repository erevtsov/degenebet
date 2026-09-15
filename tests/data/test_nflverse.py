from __future__ import annotations

from collections.abc import Callable

import nflreadpy.config
import polars as pl
import pytest

from degenebet.data import nflverse


def test_nflverse_configures_filesystem_cache() -> None:
    config = nflreadpy.config.get_config()
    assert config.cache_mode == nflreadpy.config.CacheMode.FILESYSTEM
    assert str(config.cache_dir).endswith("nflverse")


_WRAPPERS: list[tuple[Callable[..., pl.DataFrame], str]] = [
    (nflverse.load_schedules, "load_schedules"),
    (nflverse.load_player_stats, "load_player_stats"),
    (nflverse.load_team_stats, "load_team_stats"),
    (nflverse.load_rosters, "load_rosters"),
]


def _fake_frame() -> pl.DataFrame:
    return pl.DataFrame({"season": [2024]})


@pytest.mark.parametrize(("wrapper", "nflreadpy_fn"), _WRAPPERS)
def test_wrapper_translates_none_to_true(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: Callable[..., pl.DataFrame],
    nflreadpy_fn: str,
) -> None:
    calls: list[object] = []

    def fake(seasons: object) -> pl.DataFrame:
        calls.append(seasons)
        return _fake_frame()

    monkeypatch.setattr(f"nflreadpy.{nflreadpy_fn}", fake)

    result = wrapper()

    assert calls == [True]
    assert result.equals(_fake_frame())


@pytest.mark.parametrize(("wrapper", "nflreadpy_fn"), _WRAPPERS)
def test_wrapper_passes_explicit_seasons(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: Callable[..., pl.DataFrame],
    nflreadpy_fn: str,
) -> None:
    calls: list[object] = []

    def fake(seasons: object) -> pl.DataFrame:
        calls.append(seasons)
        return _fake_frame()

    monkeypatch.setattr(f"nflreadpy.{nflreadpy_fn}", fake)

    wrapper([2022, 2023])

    assert calls == [[2022, 2023]]
