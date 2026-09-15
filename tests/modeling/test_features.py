from __future__ import annotations

import polars as pl

from degenebet.modeling.features import build_model_table, compute_rolling_features


def _team_stats_row(
    *,
    season: int,
    week: int,
    team: str,
    opponent_team: str,
    game_id: str,
    passing_epa: float,
    attempts: int = 10,
    carries: int = 10,
    def_interceptions: int = 0,
    fumble_recovery_opp: int = 0,
    passing_interceptions: int = 0,
    fumbles_lost_total: int = 0,
) -> dict[str, object]:
    return {
        "season": season,
        "week": week,
        "team": team,
        "opponent_team": opponent_team,
        "game_id": game_id,
        "passing_epa": passing_epa,
        "rushing_epa": 0.0,
        "attempts": attempts,
        "carries": carries,
        "def_interceptions": def_interceptions,
        "fumble_recovery_opp": fumble_recovery_opp,
        "passing_interceptions": passing_interceptions,
        "fumbles_lost_total": fumbles_lost_total,
    }


def _four_game_team_stats() -> pl.DataFrame:
    # Team A vs Team B, 4 games. A's offense_epa_per_play (passing_epa/20):
    # G1=1.0, G2=2.0, G3=0.0, G4=3.0. B's: G1=0.5, G2=1.5, G3=2.5, G4=0.0.
    # A's turnover_margin: G1=1, G2=-1, G3=0, G4=2.
    rows = [
        _team_stats_row(
            season=2099,
            week=1,
            team="A",
            opponent_team="B",
            game_id="g1",
            passing_epa=20.0,
            def_interceptions=1,
        ),
        _team_stats_row(
            season=2099, week=1, team="B", opponent_team="A", game_id="g1", passing_epa=10.0
        ),
        _team_stats_row(
            season=2099,
            week=2,
            team="A",
            opponent_team="B",
            game_id="g2",
            passing_epa=40.0,
            passing_interceptions=1,
        ),
        _team_stats_row(
            season=2099, week=2, team="B", opponent_team="A", game_id="g2", passing_epa=30.0
        ),
        _team_stats_row(
            season=2099, week=3, team="A", opponent_team="B", game_id="g3", passing_epa=0.0
        ),
        _team_stats_row(
            season=2099, week=3, team="B", opponent_team="A", game_id="g3", passing_epa=50.0
        ),
        _team_stats_row(
            season=2099,
            week=4,
            team="A",
            opponent_team="B",
            game_id="g4",
            passing_epa=60.0,
            def_interceptions=2,
        ),
        _team_stats_row(
            season=2099, week=4, team="B", opponent_team="A", game_id="g4", passing_epa=0.0
        ),
    ]
    return pl.DataFrame(rows)


def test_compute_rolling_features_excludes_current_game_and_respects_min_history() -> None:
    rolling = compute_rolling_features(_four_game_team_stats(), window=3, min_history=2)

    team_a = rolling.filter(pl.col("team") == "A").sort("week")
    # G1, G2 dropped: fewer than min_history=2 prior games.
    assert team_a["week"].to_list() == [3, 4]

    g3 = team_a.filter(pl.col("week") == 3).row(0, named=True)
    assert g3["rolling_offense_epa_per_play"] == 1.5  # mean(1.0, 2.0)
    assert g3["rolling_defense_epa_allowed_per_play"] == 1.0  # mean(B's 0.5, 1.5)
    assert g3["rolling_turnover_margin"] == 0.0  # mean(1, -1)

    g4 = team_a.filter(pl.col("week") == 4).row(0, named=True)
    assert g4["rolling_offense_epa_per_play"] == 1.0  # mean(1.0, 2.0, 0.0), window=3
    assert g4["rolling_defense_epa_allowed_per_play"] == 1.5  # mean(B's 0.5, 1.5, 2.5)
    assert g4["rolling_turnover_margin"] == 0.0  # mean(1, -1, 0)


def test_build_model_table_joins_home_and_away_features() -> None:
    rolling_features = pl.DataFrame(
        {
            "season": [2099, 2099],
            "week": [5, 5],
            "team": ["A", "B"],
            "game_id": ["g5", "g5"],
            "rolling_offense_epa_per_play": [1.2, 0.8],
            "rolling_defense_epa_allowed_per_play": [0.5, 1.1],
            "rolling_turnover_margin": [0.3, -0.2],
        }
    )
    schedules = pl.DataFrame(
        {
            "game_id": ["g5"],
            "season": [2099],
            "week": [5],
            "home_team": ["A"],
            "away_team": ["B"],
            "result": [7],
            "spread_line": [-2.5],
        }
    )

    table = build_model_table(schedules, rolling_features)

    assert table.height == 1
    row = table.row(0, named=True)
    assert row["home_offense_epa"] == 1.2
    assert row["home_defense_epa_allowed"] == 0.5
    assert row["home_turnover_margin"] == 0.3
    assert row["away_offense_epa"] == 0.8
    assert row["away_defense_epa_allowed"] == 1.1
    assert row["away_turnover_margin"] == -0.2
    assert row["result"] == 7
    assert row["spread_line"] == -2.5


def test_build_model_table_drops_games_missing_either_teams_features() -> None:
    rolling_features = pl.DataFrame(
        {
            "season": [2099],
            "week": [5],
            "team": ["A"],
            "game_id": ["g5"],
            "rolling_offense_epa_per_play": [1.2],
            "rolling_defense_epa_allowed_per_play": [0.5],
            "rolling_turnover_margin": [0.3],
        }
    )
    schedules = pl.DataFrame(
        {
            "game_id": ["g5"],
            "season": [2099],
            "week": [5],
            "home_team": ["A"],
            "away_team": ["B"],  # B has no rolling_features row
            "result": [7],
            "spread_line": [-2.5],
        }
    )

    table = build_model_table(schedules, rolling_features)

    assert table.height == 0
