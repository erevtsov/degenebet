from __future__ import annotations

import pytest

from degenebet.data import teams


def test_canonical_teams_has_32_teams() -> None:
    assert len(teams.CANONICAL_TEAMS) == 32


def test_assert_maps_to_canonical_teams_passes_for_complete_crosswalk() -> None:
    crosswalk = {f"Vendor Name {i}": code for i, code in enumerate(teams.CANONICAL_TEAMS)}

    teams.assert_maps_to_canonical_teams(crosswalk)  # does not raise


def test_assert_maps_to_canonical_teams_raises_on_missing_team() -> None:
    incomplete = {code: code for code in list(teams.CANONICAL_TEAMS)[:-1]}

    with pytest.raises(ValueError, match="missing"):
        teams.assert_maps_to_canonical_teams(incomplete)


def test_assert_maps_to_canonical_teams_raises_on_unknown_code() -> None:
    crosswalk = {code: code for code in teams.CANONICAL_TEAMS}
    crosswalk["Extra Team"] = "XXX"

    with pytest.raises(ValueError, match="unknown"):
        teams.assert_maps_to_canonical_teams(crosswalk)
