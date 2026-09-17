"""degenebet CLI — on-demand commands to warm the local data cache."""

from __future__ import annotations

import typer

from degenebet import data

app = typer.Typer(no_args_is_help=True)
fetch_app = typer.Typer(no_args_is_help=True)
app.add_typer(fetch_app, name="fetch", help="Fetch and cache NFL data")


def _parse_seasons(seasons: str | None) -> list[int] | None:
    if seasons is None:
        return None
    try:
        return [int(s) for s in seasons.split(",")]
    except ValueError as exc:
        msg = f"Seasons must be comma-separated years, got: {seasons!r}"
        raise typer.BadParameter(msg) from exc


_SEASONS_OPTION = typer.Option(None, help="Comma-separated seasons, e.g. 2022,2023")


@fetch_app.command("schedules")
def fetch_schedules(seasons: str | None = _SEASONS_OPTION) -> None:
    """Load NFL schedules/games (including historical closing lines)."""
    frame = data.sync_schedules(_parse_seasons(seasons))
    typer.echo(f"{frame.height} games loaded")


@fetch_app.command("player-stats")
def fetch_player_stats(seasons: str | None = _SEASONS_OPTION) -> None:
    """Load weekly player stats."""
    frame = data.load_player_stats(_parse_seasons(seasons))
    typer.echo(f"{frame.height} rows loaded")


@fetch_app.command("team-stats")
def fetch_team_stats(seasons: str | None = _SEASONS_OPTION) -> None:
    """Load weekly team stats."""
    frame = data.sync_team_stats(_parse_seasons(seasons))
    typer.echo(f"{frame.height} rows loaded")


@fetch_app.command("rosters")
def fetch_rosters(seasons: str | None = _SEASONS_OPTION) -> None:
    """Load seasonal roster data."""
    frame = data.load_rosters(_parse_seasons(seasons))
    typer.echo(f"{frame.height} rows loaded")


@fetch_app.command("odds")
def fetch_odds(
    force_refresh: bool = typer.Option(False, "--force-refresh"),
) -> None:
    """Load current NFL odds (moneyline, spread, total) from SharpAPI."""
    frame = data.load_odds(force_refresh=force_refresh)
    books = frame["sportsbook"].n_unique()
    pulled_at = str(frame["pulled_at"].max())
    typer.echo(f"{frame.height} odds rows from {books} sportsbook(s), pulled at {pulled_at}")
