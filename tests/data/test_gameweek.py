from __future__ import annotations

from degenebet.data.gameweek import Gameweek


def test_as_int_combines_season_and_week() -> None:
    assert Gameweek(2024, 3).as_int() == 202403


def test_as_int_zero_pads_single_digit_weeks() -> None:
    assert Gameweek(2024, 1).as_int() == 202401


def test_gameweek_sorts_by_season_then_week() -> None:
    assert Gameweek(2023, 22) < Gameweek(2024, 1)
    assert Gameweek(2024, 1) < Gameweek(2024, 2)
    assert Gameweek(2024, 1) == Gameweek(2024, 1)
