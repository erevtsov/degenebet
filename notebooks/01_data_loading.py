import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import date

    import altair as alt
    import marimo as mo
    import polars as pl

    from degenebet.data import nflverse_cache
    from degenebet.data.access import DataAccess, NflverseSource, SharpApiSource
    from degenebet.modeling.features import build_model_table, compute_rolling_features

    return (
        DataAccess,
        NflverseSource,
        SharpApiSource,
        alt,
        build_model_table,
        compute_rolling_features,
        date,
        mo,
        nflverse_cache,
        pl,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Loading and exploring NFL data

    Schedules load through `DataAccess` — the production point-in-time read
    layer, stitching historical (`nflreadpy`) and current (SharpAPI) lines.
    Team stats have no `DataAccess` source of their own (by design: future
    games have no stats to stitch against), so those read straight from the
    local merged cache. Neither path ever calls `nflreadpy` or SharpAPI live;
    run `degenebet fetch schedules`/`degenebet fetch team-stats` (or
    `degenebet sync` to pull the scheduled-fetch workflow's data) first to
    populate the cache.
    """)
    return


@app.cell
def _():
    SEASONS = [2022, 2023, 2024]
    return (SEASONS,)


@app.cell
def _(DataAccess, NflverseSource, SEASONS, SharpApiSource, date, pl):
    _access = DataAccess(NflverseSource(), SharpApiSource())
    _schedules = _access.get_team_data(
        start_date=date(min(SEASONS), 1, 1),
        end_date=date(max(SEASONS) + 1, 3, 1),  # NFL seasons run into Feb/March
        as_of_date=date.today(),
    )
    schedules = _schedules.filter(pl.col("season").is_in(SEASONS))
    schedules.head()
    return (schedules,)


@app.cell
def _(SEASONS, nflverse_cache, pl):
    _team_stats = nflverse_cache.read_merged("team_stats")
    if _team_stats is None:
        raise RuntimeError(
            "No team_stats in the local cache -- run `degenebet fetch team-stats` "
            "or `degenebet sync` first."
        )
    # DataAccess/NflverseSource lowercase team codes at their read boundary
    # (teams.CANONICAL_TEAMS convention); match that here so the join in
    # build_model_table lines up against schedules' home_team/away_team.
    team_stats = _team_stats.filter(pl.col("season").is_in(SEASONS)).with_columns(
        pl.col("team").str.to_lowercase(),
        pl.col("opponent_team").str.to_lowercase(),
    )
    team_stats.head()
    return (team_stats,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Feature engineering (reusing the production pipeline)
    """)
    return


@app.cell
def _(compute_rolling_features, team_stats):
    rolling = compute_rolling_features(team_stats)
    rolling.head()
    return (rolling,)


@app.cell
def _(build_model_table, rolling, schedules):
    model_table = build_model_table(schedules, rolling)
    model_table.head()
    return (model_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Simple analysis
    """)
    return


@app.cell
def _(model_table, pl):
    analysis_table = model_table.with_columns(
        (pl.col("result") - pl.col("spread_line")).alias("home_cover_margin"),
        (pl.col("home_offense_epa") - pl.col("away_offense_epa")).alias("net_offense_epa_edge"),
    )
    analysis_table.select(
        "result", "spread_line", "home_cover_margin", "net_offense_epa_edge"
    ).describe()
    return (analysis_table,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualizations

    `spread_line` sign convention: **positive means home favored**. So
    `home_cover_margin = result - spread_line` is positive when the home
    team beat the spread.
    """)
    return


@app.cell
def _(alt, analysis_table, mo):
    _scatter = (
        alt.Chart(analysis_table)
        .mark_circle(opacity=0.5)
        .encode(
            x=alt.X("net_offense_epa_edge", title="Net offensive EPA/play edge (home - away)"),
            y=alt.Y("result", title="Actual margin (home - away)"),
            tooltip=["season", "week", "home_team", "away_team", "result"],
        )
        .properties(
            title="Does offensive EPA edge predict the actual margin?", width=500, height=350
        )
    )
    _trend = _scatter.transform_regression("net_offense_epa_edge", "result").mark_line(
        color="red"
    )
    mo.ui.altair_chart(_scatter + _trend)
    return


@app.cell
def _(alt, analysis_table, mo, pl):
    _hist = (
        alt.Chart(analysis_table)
        .mark_bar()
        .encode(
            x=alt.X(
                "home_cover_margin",
                bin=alt.Bin(maxbins=40),
                title="Home cover margin (result - spread_line)",
            ),
            y=alt.Y("count()", title="Games"),
        )
        .properties(title="Distribution of home cover margin (ATS)", width=500, height=300)
    )
    _rule = (
        alt.Chart(pl.DataFrame({"x": [0]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(x="x")
    )
    mo.ui.altair_chart(_hist + _rule)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Rolling EPA/play by team, interactively
    """)
    return


@app.cell
def _(mo, rolling):
    team_dropdown = mo.ui.dropdown(
        options=sorted(rolling["team"].unique().to_list()),
        value="kc",
        label="Team",
    )
    team_dropdown
    return (team_dropdown,)


@app.cell
def _(alt, mo, pl, rolling, team_dropdown):
    _team_rolling = rolling.filter(pl.col("team") == team_dropdown.value).sort(["season", "week"])
    _chart = (
        alt.Chart(_team_rolling)
        .transform_fold(
            ["rolling_offense_epa_per_play", "rolling_defense_epa_allowed_per_play"],
            as_=["metric", "value"],
        )
        .mark_line(point=True)
        .encode(
            x=alt.X("week:O", title="Week"),
            y=alt.Y("value:Q", title="Rolling EPA/play"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["season", "week", "metric:N", "value:Q"],
        )
        .properties(width=220, height=300)
        .facet(column=alt.Column("season:N", title=None))
        .properties(title=f"{team_dropdown.value}: rolling offense/defense EPA per play")
    )
    mo.ui.altair_chart(_chart)
    return


if __name__ == "__main__":
    app.run()
