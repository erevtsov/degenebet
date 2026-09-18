import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import date

    import altair as alt
    import marimo as mo
    import polars as pl
    from sklearn.linear_model import Ridge

    from degenebet.data.access import DataAccess, NflverseSource, SharpApiSource
    from degenebet.data.gameweek import Gameweek
    from degenebet.modeling.efficacy import Efficacy
    from degenebet.modeling.features import compute_rolling_features
    from degenebet.modeling.splits import SingleSplit, iterate_folds
    from degenebet.modeling.spread_model import SpreadModel

    return (
        DataAccess,
        Efficacy,
        Gameweek,
        NflverseSource,
        Ridge,
        SharpApiSource,
        SingleSplit,
        SpreadModel,
        alt,
        compute_rolling_features,
        date,
        iterate_folds,
        mo,
        pl,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Model selection: comparing candidates with Efficacy

    Runs a few candidate `SpreadModel` configs through `SingleSplit` +
    `iterate_folds` + `Efficacy.evaluate_folds`, comparing their
    out-of-sample prediction quality (`predicted_result` vs. `result`
    only -- `spread_line` never enters `Efficacy`; that's a `Backtest`
    concern). This is `architecture-notes.md`'s "step 1: model/
    hyperparameter selection" workflow: loop over candidates, score each,
    pick one by human judgment -- `Efficacy` never picks for you.

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
):
    access = DataAccess(NflverseSource(), SharpApiSource())
    start_week = Gameweek(min(SEASONS), 1)
    end_week = Gameweek(max(SEASONS), 22)  # NFL seasons run through the Super Bowl
    as_of_date = date.today()

    team_data = access.get_team_data(start_week, end_week, as_of_date=as_of_date)
    rolling = compute_rolling_features(team_data)
    model_table = access.get_game_data(
        start_week, end_week, as_of_date=as_of_date, team_data=rolling
    ).drop_nulls(
        subset=[
            "home_offense_epa",
            "home_defense_epa_allowed",
            "home_turnover_margin",
            "away_offense_epa",
            "away_defense_epa_allowed",
            "away_turnover_margin",
        ]
    )
    model_table.head()
    return (model_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Candidate models
    """)
    return


@app.cell
def _(Ridge, SpreadModel):
    candidates = {
        "linear": lambda: SpreadModel(),
        "ridge (alpha=1.0)": lambda: SpreadModel(model=Ridge(alpha=1.0)),
        "ridge (alpha=10.0)": lambda: SpreadModel(model=Ridge(alpha=10.0)),
    }
    return (candidates,)


@app.cell
def _(Efficacy, SingleSplit, candidates, iterate_folds, model_table, pl):
    split_strategy = SingleSplit(train_seasons=[2022, 2023], test_seasons=[2024])
    efficacy = Efficacy()

    rows = []
    for name, model_factory in candidates.items():
        folds = iterate_folds(model_table, split_strategy, model_factory)
        result = efficacy.evaluate_folds(folds)
        rows.append({"candidate": name, **result.metrics})

    comparison = pl.DataFrame(rows)
    comparison
    return (comparison,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Out-of-sample RMSE by candidate

    Lower is better -- RMSE is in points of margin.
    """)
    return


@app.cell
def _(alt, comparison, mo):
    _chart = (
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            x=alt.X("candidate:N", title="Candidate"),
            y=alt.Y("rmse:Q", title="Out-of-sample RMSE"),
            tooltip=["candidate", "rmse", "r_squared", "directional_accuracy"],
        )
        .properties(title="Candidate comparison: out-of-sample RMSE", width=400, height=300)
    )
    mo.ui.altair_chart(_chart)
    return


if __name__ == "__main__":
    app.run()
