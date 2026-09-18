import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import date

    import altair as alt
    import marimo as mo
    import polars as pl

    from degenebet.data.access import DataAccess, NflverseSource, SharpApiSource
    from degenebet.data.gameweek import Gameweek
    from degenebet.modeling.backtest import Backtest, FlatSizing, compute_bankroll_trajectory
    from degenebet.modeling.features import compute_rolling_features
    from degenebet.modeling.splits import WalkForwardSplit, iterate_folds
    from degenebet.modeling.spread_model import SpreadModel

    return (
        Backtest,
        DataAccess,
        FlatSizing,
        Gameweek,
        NflverseSource,
        SharpApiSource,
        SpreadModel,
        WalkForwardSplit,
        alt,
        compute_bankroll_trajectory,
        compute_rolling_features,
        date,
        iterate_folds,
        mo,
        pl,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Backtest: real odds, walk-forward, compounding bankroll

    Runs `Backtest` + `WalkForwardSplit` against a `SpreadModel` candidate:
    one fold per gameweek after a warmup period, each scored at that
    game's own real American odds (`home_spread_odds`/`away_spread_odds`
    -- not a hardcoded -110). `FlatSizing` divides a normalized pool of
    `1.0` evenly across each week's bets; the top-level `roi_pct` is a
    pooled, **equal-weighted average across folds** (each gameweek counts
    the same, whether it had 2 games or 14) -- a different, still useful,
    question from the compounding dollar trajectory below (`what's my
    actual bankroll if I reinvest it every week`). `ats_win_rate`, by
    contrast, is **bet-count-weighted, not week-weighted** -- each bet
    counts equally regardless of which week it fell in -- which is the
    more useful framing when comparing it to `Efficacy.directional_accuracy`
    on a per-bet basis.

    Reads from the local cache only, via `DataAccess`; run
    `degenebet fetch schedules`/`degenebet fetch team-stats` (or
    `degenebet sync`) first.
    """)
    return


@app.cell
def _():
    SEASONS = [2022, 2023, 2024]
    return (SEASONS,)


@app.cell
def _(
    DataAccess,
    Gameweek,
    NflverseSource,
    SEASONS,
    SharpApiSource,
    compute_rolling_features,
    date,
    pl,
):
    access = DataAccess(NflverseSource(), SharpApiSource())
    start_week = Gameweek(min(SEASONS), 1)
    end_week = Gameweek(max(SEASONS), 22)  # NFL seasons run through the Super Bowl
    as_of_date = date.today()

    team_data = access.get_team_data(start_week, end_week, as_of_date=as_of_date)
    rolling = compute_rolling_features(team_data)
    model_table = (
        access.get_game_data(start_week, end_week, as_of_date=as_of_date, team_data=rolling)
        .drop_nulls(
            subset=[
                "home_offense_epa",
                "home_defense_epa_allowed",
                "home_turnover_margin",
                "away_offense_epa",
                "away_defense_epa_allowed",
                "away_turnover_margin",
            ]
        )
        # Explicit opt-in, per Backtest's null-handling contract: only
        # played, priced games reach Backtest -- it never filters these
        # itself.
        .filter(
            pl.col("result").is_not_null()
            & pl.col("spread_line").is_not_null()
            & pl.col("home_spread_odds").is_not_null()
            & pl.col("away_spread_odds").is_not_null()
        )
    )
    model_table.head()
    return (model_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Walk-forward backtest
    """)
    return


@app.cell
def _(Backtest, FlatSizing, SpreadModel, WalkForwardSplit, iterate_folds, model_table):
    split_strategy = WalkForwardSplit(warmup_seasons=1)
    folds = iterate_folds(model_table, split_strategy, SpreadModel)
    backtest = Backtest(sizing_strategy=FlatSizing(), edge_threshold=1.0)
    result = backtest.run_folds(folds)

    print(
        f"bets_placed={result.bets_placed} "
        f"ats_win_rate={result.ats_win_rate:.3f} "
        f"units_won={result.units_won:.3f} "
        f"roi_pct={result.roi_pct:.1f}%"
    )
    result.by_fold
    return (result,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Bankroll trajectory

    Compounding: each fold's stake is sized off the *actual* bankroll
    remaining after every prior fold, not a fixed amount. This is
    full-reinvestment sizing -- the whole current bankroll is staked
    every week, none held in reserve -- so a single week where every bet
    loses (`units_won == -1.0` exactly) wipes the bankroll to `$0`, which
    then correctly stays `$0` regardless of how well later weeks would
    have gone. That's not a bug; it's the real risk of this sizing rule,
    and it does happen in this real 2022-2024 walk-forward run below.
    """)
    return


@app.cell
def _(compute_bankroll_trajectory, result):
    trajectory = compute_bankroll_trajectory(result.by_fold, starting_bankroll=1000.0)

    print(
        f"starting_bankroll={trajectory.starting_bankroll:.2f} "
        f"ending_bankroll={trajectory.ending_bankroll:.2f} "
        f"total_pnl={trajectory.total_pnl:.2f}"
    )
    trajectory.by_fold
    return (trajectory,)


@app.cell
def _(alt, mo, trajectory):
    _chart = (
        alt.Chart(trajectory.by_fold)
        .mark_line(point=True)
        .encode(
            x=alt.X("fold:Q", title="Fold (gameweek, in order)"),
            y=alt.Y("bankroll_after:Q", title="Bankroll ($)"),
            tooltip=["fold", "bankroll_before", "pnl", "bankroll_after"],
        )
        .properties(title="Compounding bankroll trajectory", width=500, height=300)
    )
    mo.ui.altair_chart(_chart)
    return


if __name__ == "__main__":
    app.run()
