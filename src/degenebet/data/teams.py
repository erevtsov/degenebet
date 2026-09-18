"""Canonical NFL team identifiers every vendor's data gets translated into.

nflreadpy's own abbreviations are the canonical set: nflverse is this
project's primary/authoritative source (schedules, team stats), so any
other vendor's team identifiers (e.g. SharpAPI's "Buffalo Bills" full
names) have to line up with these codes for DataAccess to join across
sources at all. See access.py's module docstring for the full
cross-vendor join contract (HistoricalDataSource/CurrentDataSource) this
is one piece of.

Codes are lowercase (nflreadpy's own raw data is uppercase, e.g. "BUF") --
this project's naming convention is lower_snake_case throughout, so
normalization happens once, at the merge-cache write boundary
(nflverse_cache.sync_schedules/sync_team_stats), rather than the
canonical set matching the vendor's casing.
"""

from __future__ import annotations

CANONICAL_TEAMS: frozenset[str] = frozenset(
    {
        "ari",
        "atl",
        "bal",
        "buf",
        "car",
        "chi",
        "cin",
        "cle",
        "dal",
        "den",
        "det",
        "gb",
        "hou",
        "ind",
        "jax",
        "kc",
        "la",
        "lac",
        "lv",
        "mia",
        "min",
        "ne",
        "no",
        "nyg",
        "nyj",
        "phi",
        "pit",
        "sf",
        "sea",
        "tb",
        "ten",
        "was",
    }
)


def assert_maps_to_canonical_teams(crosswalk: dict[str, str]) -> None:
    """Raise ValueError if `crosswalk`'s values don't exactly cover
    CANONICAL_TEAMS -- catches a typo'd code, a missing team, or an
    unrecognized code in a vendor's crosswalk generically."""
    mapped = set(crosswalk.values())
    missing = CANONICAL_TEAMS - mapped
    unknown = mapped - CANONICAL_TEAMS
    if missing or unknown:
        raise ValueError(
            f"Crosswalk doesn't map exactly onto CANONICAL_TEAMS -- "
            f"missing: {sorted(missing)}, unknown: {sorted(unknown)}"
        )
