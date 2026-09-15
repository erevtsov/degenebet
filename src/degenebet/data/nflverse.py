"""Thin wrappers over nflreadpy's load_* functions.

Kept separate from direct nflreadpy calls elsewhere in degenebet so there is
one stable, typed interface that would only need to change here if the
underlying library's API changes. ``seasons=None`` means "all available
seasons" — nflreadpy's own sentinel for that is ``seasons=True``, so each
wrapper translates ``None`` to ``True``.
"""

from __future__ import annotations

import nflreadpy
import nflreadpy.config
import polars as pl

from degenebet.config import cache_dir

# update_config mutates a process-global config, read once here at import
# time — a later DEGENEBET_CACHE_DIR change within the same process won't
# re-apply. Acceptable for a short-lived CLI invocation.
nflreadpy.config.update_config(cache_mode="filesystem", cache_dir=str(cache_dir() / "nflverse"))


def _seasons_arg(seasons: list[int] | None) -> list[int] | bool:
    return True if seasons is None else seasons


def load_schedules(seasons: list[int] | None = None) -> pl.DataFrame:
    """Return NFL schedules/games, including historical closing lines."""
    return nflreadpy.load_schedules(seasons=_seasons_arg(seasons))  # type: ignore[no-any-return]


def load_player_stats(seasons: list[int] | None = None) -> pl.DataFrame:
    """Return weekly player stats."""
    return nflreadpy.load_player_stats(seasons=_seasons_arg(seasons))  # type: ignore[no-any-return]


def load_team_stats(seasons: list[int] | None = None) -> pl.DataFrame:
    """Return weekly team stats."""
    return nflreadpy.load_team_stats(seasons=_seasons_arg(seasons))  # type: ignore[no-any-return]


def load_rosters(seasons: list[int] | None = None) -> pl.DataFrame:
    """Return seasonal roster data."""
    return nflreadpy.load_rosters(seasons=_seasons_arg(seasons))  # type: ignore[no-any-return]
