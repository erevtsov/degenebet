from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from degenebet.modeling.backtest import run_backtest
from degenebet.modeling.spread_model import SpreadModel

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
    # edge = predicted_result - spread_line; home_cover_margin = result - spread_line.
    # g_home_bet: home_offense_epa=0.5 -> predicted~5; spread_line=-1
    #   -> edge = 5-(-1) = 6 > 1.0 -> bet home; result=6
    #   -> home_cover_margin = 6-(-1) = 7 > 0 -> home covers (bet wins)
    # g_away_bet: home_offense_epa=-0.5 -> predicted~-5; spread_line=1
    #   -> edge = -5-1 = -6 < -1.0 -> bet away; result=-6
    #   -> home_cover_margin = -6-1 = -7 < 0 -> away covers (bet wins)
    # g_no_bet: home_offense_epa=0.05 -> predicted~0.5; spread_line=0
    #   -> edge = 0.5-0 = 0.5, within threshold -> no bet
    # g_push: home_offense_epa=0.5 -> predicted~5; spread_line=-5
    #   -> edge = 5-(-5) = 10 > 1.0 -> bet WOULD be placed here (unlike the
    #   old wrong-convention formula, where edge~0 meant no bet); kept as a
    #   no-bet fixture would now require a different spread_line, so this
    #   game is left out of _model_table() entirely and a genuine push case
    #   is exercised separately in _model_table_with_push() below.
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
    # edge = predicted_result - spread_line; home_cover_margin = result - spread_line.
    # g_push: home_offense_epa=0.5 -> predicted~5; spread_line=3.0
    #   -> edge = 5-3 = 2.0 > 1.0 -> bet home (bet IS placed)
    #   actual result=3.0 -> home_cover_margin = 3.0-3.0 = 0.0 (push)
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
            "spread_line": [-1.0, 1.0, 0.0, 3.0],
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


def _model_table_home_favorite_narrow_loss() -> pl.DataFrame:
    # Constructed so the two sign conventions disagree: home is a wide
    # favorite (spread_line=10.0, correct convention: positive = home
    # favored) and the model, trained on result = 10 * home_offense_epa,
    # predicts home wins by 15 -- edge = 15 - 10 = 5 > 1.0, so a bet is
    # placed on home under BOTH the correct ("-") and the old buggy ("+")
    # formula (edge would be 15 + 10 = 25 under the bug, same side), which
    # isolates the scoring bug alone rather than conflating it with a
    # side-selection difference.
    #
    # The actual result is a narrow home win by 2 points, well under the
    # 10-point spread, so home does NOT cover:
    #   correct:  home_cover_margin = 2 - 10 = -8 < 0 -> away covers -> bet LOSES
    #   old bug:  home_cover_margin = 2 + 10 = 12 > 0 -> home covers -> bet WINS
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
            "game_id": ["g_narrow_loss"],
            "season": [2099],
            "week": [1],
            "home_offense_epa": [1.5],
            "home_defense_epa_allowed": [0.0],
            "home_turnover_margin": [0.0],
            "away_offense_epa": [0.0],
            "away_defense_epa_allowed": [0.0],
            "away_turnover_margin": [0.0],
            "result": [2.0],
            "spread_line": [10.0],
        }
    )
    return pl.concat([train, test])


def test_run_backtest_home_favorite_narrow_loss_scored_as_away_cover() -> None:
    result = run_backtest(
        _model_table_home_favorite_narrow_loss(),
        train_seasons=[2098],
        test_seasons=[2099],
        edge_threshold=1.0,
    )

    assert result.bets_placed == 1
    row = result.bets.filter(pl.col("game_id") == "g_narrow_loss")
    assert row["side"][0] == "home"
    assert row["home_cover_margin"][0] == -8.0
    assert row["won"][0] is False
    assert row["units"][0] == -1.1
    assert result.units_won == -1.1


class _ConstantRegressor:
    """Test double predicting a fixed constant regardless of features --
    used to prove run_backtest actually delegates to an injected
    SpreadModel rather than silently hardcoding a fresh SpreadModel()."""

    def __init__(self, constant: float) -> None:
        self._constant = constant

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> _ConstantRegressor:
        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.full(X.shape[0], self._constant)


def _model_table_for_injected_model() -> pl.DataFrame:
    zeros = [0.0, 0.0, 0.0]
    train = pl.DataFrame(
        {
            "game_id": ["train_0", "train_1", "train_2"],
            "season": [3000, 3000, 3000],
            "week": [1, 2, 3],
            "home_offense_epa": zeros,
            "home_defense_epa_allowed": zeros,
            "home_turnover_margin": zeros,
            "away_offense_epa": zeros,
            "away_defense_epa_allowed": zeros,
            "away_turnover_margin": zeros,
            "result": [1.0, 5.0, -2.0],
            "spread_line": [0.0, 0.0, 0.0],
        }
    )
    test = pl.DataFrame(
        {
            "game_id": ["g_win", "g_loss"],
            "season": [3001, 3001],
            "week": [1, 1],
            "home_offense_epa": [0.0, 0.0],
            "home_defense_epa_allowed": [0.0, 0.0],
            "home_turnover_margin": [0.0, 0.0],
            "away_offense_epa": [0.0, 0.0],
            "away_defense_epa_allowed": [0.0, 0.0],
            "away_turnover_margin": [0.0, 0.0],
            "result": [50.0, -60.0],
            "spread_line": [0.0, 0.0],
        }
    )
    return pl.concat([train, test])


def test_run_backtest_uses_injected_model_not_ignored() -> None:
    # Default LinearRegression fit on this train set would predict ~0 for
    # zero-valued features -- nowhere near the constant 20.0 below. If
    # run_backtest ignored the injected model, predicted_result would not
    # equal 20.0 and the win/loss split below would not hold.
    injected = SpreadModel(model=_ConstantRegressor(constant=20.0))

    result = run_backtest(
        _model_table_for_injected_model(),
        train_seasons=[3000],
        test_seasons=[3001],
        model=injected,
    )

    assert result.bets_placed == 2
    assert set(result.bets["predicted_result"].to_list()) == {20.0}
    # g_win: home_cover_margin = 50 - 0 = 50 > 0 -> home covers -> win
    # g_loss: home_cover_margin = -60 - 0 = -60 < 0 -> away covers -> loss (bet was on home)
    win_row = result.bets.filter(pl.col("game_id") == "g_win")
    loss_row = result.bets.filter(pl.col("game_id") == "g_loss")
    assert win_row["won"][0] is True
    assert loss_row["won"][0] is False
    assert result.units_won == pytest.approx(1.0 - 1.1)


def test_run_backtest_raises_on_empty_train_seasons() -> None:
    with pytest.raises(ValueError, match="train_seasons"):
        run_backtest(_model_table(), train_seasons=[1900], test_seasons=[2099])


def test_run_backtest_returns_zeroed_result_for_empty_test_seasons() -> None:
    result = run_backtest(_model_table(), train_seasons=[2098], test_seasons=[1900])

    assert result.bets_placed == 0
    assert result.ats_win_rate == 0.0
    assert result.units_won == 0.0
    assert result.roi_pct == 0.0


def test_run_backtest_raises_on_overlapping_train_and_test_seasons() -> None:
    with pytest.raises(ValueError, match="overlap"):
        run_backtest(_model_table(), train_seasons=[2098, 2099], test_seasons=[2099])


def test_run_backtest_drops_null_result_rows_from_test_instead_of_scoring_as_loss() -> None:
    # g_home_bet would clearly place a winning home bet (see _model_table());
    # a null-result row mixed into the same test season must not be scored
    # as a loss via the `.otherwise()` fallthrough -- it should simply be
    # excluded from the bets entirely.
    table = _model_table()
    null_row = pl.DataFrame(
        {
            "game_id": ["g_unplayed"],
            "season": [2099],
            "week": [1],
            "home_offense_epa": [0.9],
            "home_defense_epa_allowed": [0.0],
            "home_turnover_margin": [0.0],
            "away_offense_epa": [0.0],
            "away_defense_epa_allowed": [0.0],
            "away_turnover_margin": [0.0],
            "result": [None],
            "spread_line": [-5.0],
        },
        schema=table.schema,
    )
    table_with_unplayed_game = pl.concat([table, null_row])

    result = run_backtest(
        table_with_unplayed_game,
        train_seasons=[2098],
        test_seasons=[2099],
        edge_threshold=1.0,
    )

    assert "g_unplayed" not in result.bets["game_id"].to_list()
    # Unaffected vs. the original _model_table() result: still 2 qualifying bets.
    assert result.bets_placed == 2
    assert result.ats_win_rate == 1.0


def test_run_backtest_edge_exactly_at_threshold_places_no_bet() -> None:
    # Strict >/< per the docstring's "exceeds": edge == edge_threshold exactly
    # must not place a bet.
    at_threshold = SpreadModel(model=_ConstantRegressor(constant=5.0))
    train = pl.DataFrame(
        {
            "game_id": ["train_0", "train_1"],
            "season": [4000, 4000],
            "week": [1, 2],
            "home_offense_epa": [0.0, 0.0],
            "home_defense_epa_allowed": [0.0, 0.0],
            "home_turnover_margin": [0.0, 0.0],
            "away_offense_epa": [0.0, 0.0],
            "away_defense_epa_allowed": [0.0, 0.0],
            "away_turnover_margin": [0.0, 0.0],
            "result": [1.0, -1.0],
            "spread_line": [0.0, 0.0],
        }
    )
    test = pl.DataFrame(
        {
            "game_id": ["g_at_threshold"],
            "season": [4001],
            "week": [1],
            "home_offense_epa": [0.0],
            "home_defense_epa_allowed": [0.0],
            "home_turnover_margin": [0.0],
            "away_offense_epa": [0.0],
            "away_defense_epa_allowed": [0.0],
            "away_turnover_margin": [0.0],
            "result": [3.0],
            "spread_line": [4.0],  # predicted (5.0) - spread_line (4.0) == edge_threshold (1.0)
        }
    )
    table = pl.concat([train, test])

    result = run_backtest(
        table, train_seasons=[4000], test_seasons=[4001], edge_threshold=1.0, model=at_threshold
    )

    assert result.bets_placed == 0
