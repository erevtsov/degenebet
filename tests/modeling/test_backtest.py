from __future__ import annotations

import polars as pl

from degenebet.modeling.backtest import run_backtest

_FEATURE_COLUMNS = [
    "home_offense_epa",
    "home_defense_epa_allowed",
    "home_turnover_margin",
    "away_offense_epa",
    "away_defense_epa_allowed",
    "away_turnover_margin",
]


def _model_table() -> pl.DataFrame:
    # Train: result = 10 * home_offense_epa exactly (season 2098).
    # Test (season 2099): home_offense_epa chosen so predicted edge clearly
    # crosses the default edge_threshold=1.0 for some games, not others.
    train_home_offense = [0.0, 0.5, 1.0, -0.5, -1.0, 0.25, -0.25, 0.75]
    zeros_train = [0.0] * len(train_home_offense)
    train = pl.DataFrame(
        {
            "game_id": [f"train_{i}" for i in range(len(train_home_offense))],
            "season": [2098] * len(train_home_offense),
            "week": list(range(1, len(train_home_offense) + 1)),
            "home_offense_epa": train_home_offense,
            "home_defense_epa_allowed": zeros_train,
            "home_turnover_margin": zeros_train,
            "away_offense_epa": zeros_train,
            "away_defense_epa_allowed": zeros_train,
            "away_turnover_margin": zeros_train,
            "result": [10.0 * v for v in train_home_offense],
            "spread_line": [0.0] * len(train_home_offense),
        }
    )

    # Test games: predicted_result ~= 10 * home_offense_epa.
    # g_home_bet: home_offense_epa=0.5 -> predicted~5; spread_line=-1
    #   -> edge~4 > 1.0 -> bet home; result=6 -> covers (6-1=5>0)
    # g_away_bet: home_offense_epa=-0.5 -> predicted~-5; spread_line=1
    #   -> edge~-4 < -1.0 -> bet away; result=-6 -> away covers (-6+1=-5<0)
    # g_no_bet: home_offense_epa=0.05 -> predicted~0.5; spread_line=0
    #   -> edge~0.5, within threshold -> no bet
    # g_push: home_offense_epa=0.5 -> predicted~5; spread_line=-5 -> edge~0
    #   no bet either (backtest naturally excludes it)
    test = pl.DataFrame(
        {
            "game_id": ["g_home_bet", "g_away_bet", "g_no_bet"],
            "season": [2099, 2099, 2099],
            "week": [1, 1, 1],
            "home_offense_epa": [0.5, -0.5, 0.05],
            "home_defense_epa_allowed": [0.0, 0.0, 0.0],
            "home_turnover_margin": [0.0, 0.0, 0.0],
            "away_offense_epa": [0.0, 0.0, 0.0],
            "away_defense_epa_allowed": [0.0, 0.0, 0.0],
            "away_turnover_margin": [0.0, 0.0, 0.0],
            "result": [6.0, -6.0, 1.0],
            "spread_line": [-1.0, 1.0, 0.0],
        }
    )
    return pl.concat([train, test])


def _model_table_with_push() -> pl.DataFrame:
    # Same training data and base test games as _model_table(), plus g_push.
    # g_push: home_offense_epa=0.5 -> predicted~5; spread_line=-3.0
    #   -> edge~2.0 > 1.0 -> bet home (bet IS placed)
    #   actual result=3.0 -> home_cover_margin = 3.0 + (-3.0) = 0.0 (push)
    train_home_offense = [0.0, 0.5, 1.0, -0.5, -1.0, 0.25, -0.25, 0.75]
    zeros_train = [0.0] * len(train_home_offense)
    train = pl.DataFrame(
        {
            "game_id": [f"train_{i}" for i in range(len(train_home_offense))],
            "season": [2098] * len(train_home_offense),
            "week": list(range(1, len(train_home_offense) + 1)),
            "home_offense_epa": train_home_offense,
            "home_defense_epa_allowed": zeros_train,
            "home_turnover_margin": zeros_train,
            "away_offense_epa": zeros_train,
            "away_defense_epa_allowed": zeros_train,
            "away_turnover_margin": zeros_train,
            "result": [10.0 * v for v in train_home_offense],
            "spread_line": [0.0] * len(train_home_offense),
        }
    )

    test = pl.DataFrame(
        {
            "game_id": ["g_home_bet", "g_away_bet", "g_no_bet", "g_push"],
            "season": [2099, 2099, 2099, 2099],
            "week": [1, 1, 1, 1],
            "home_offense_epa": [0.5, -0.5, 0.05, 0.5],
            "home_defense_epa_allowed": [0.0, 0.0, 0.0, 0.0],
            "home_turnover_margin": [0.0, 0.0, 0.0, 0.0],
            "away_offense_epa": [0.0, 0.0, 0.0, 0.0],
            "away_defense_epa_allowed": [0.0, 0.0, 0.0, 0.0],
            "away_turnover_margin": [0.0, 0.0, 0.0, 0.0],
            "result": [6.0, -6.0, 1.0, 3.0],
            "spread_line": [-1.0, 1.0, 0.0, -3.0],
        }
    )
    return pl.concat([train, test])


def test_run_backtest_places_bets_only_when_edge_exceeds_threshold() -> None:
    result = run_backtest(
        _model_table(),
        train_seasons=[2098],
        test_seasons=[2099],
        edge_threshold=1.0,
    )

    assert result.bets_placed == 2
    assert set(result.bets["game_id"].to_list()) == {"g_home_bet", "g_away_bet"}


def test_run_backtest_scores_wins_correctly() -> None:
    result = run_backtest(
        _model_table(),
        train_seasons=[2098],
        test_seasons=[2099],
        edge_threshold=1.0,
    )

    # both placed bets covered, per the hand-derived comments above
    assert result.ats_win_rate == 1.0
    # two wins at +1.0 unit each
    assert result.units_won == 2.0
    assert result.roi_pct > 0


def test_run_backtest_with_no_qualifying_bets_returns_zeroed_result() -> None:
    result = run_backtest(
        _model_table(),
        train_seasons=[2098],
        test_seasons=[2099],
        edge_threshold=100.0,
    )

    assert result.bets_placed == 0
    assert result.ats_win_rate == 0.0
    assert result.units_won == 0.0
    assert result.roi_pct == 0.0


def test_run_backtest_scores_push_correctly() -> None:
    result = run_backtest(
        _model_table_with_push(),
        train_seasons=[2098],
        test_seasons=[2099],
        edge_threshold=1.0,
    )

    # Three bets placed: g_home_bet, g_away_bet, and g_push (bet IS placed
    # because edge~2.0 > 1.0, despite the push outcome).
    assert result.bets_placed == 3
    # Push contributes 0.0 units; the two covered bets contribute +1.0 each.
    assert result.units_won == 2.0
    # Push is excluded from win_rate denominator; 2 wins / 2 decided bets.
    assert result.ats_win_rate == 1.0
    # Positive ROI: 2.0 / (3 * 1.1) * 100 = 60.6%.
    assert result.roi_pct > 0
    # Verify g_push is in the bets DataFrame.
    assert "g_push" in result.bets["game_id"].to_list()
    # Verify the push bet has units == 0.0.
    push_row = result.bets.filter(pl.col("game_id") == "g_push")
    assert push_row["units"][0] == 0.0
