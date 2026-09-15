"""SharpAPI provider — current NFL odds (moneyline, spread, total)."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import polars as pl

from degenebet.config import sharpapi_key

_BASE_URL = "https://api.sharpapi.io/api/v1/odds"
_MARKETS = "moneyline,spread,total"
_COLUMNS = [
    "event_id",
    "home_team",
    "away_team",
    "market_type",
    "selection",
    "selection_type",
    "odds_american",
    "odds_decimal",
    "odds_probability",
    "line",
    "event_start_time",
    "sportsbook",
]


class SharpAPIProvider:
    """Fetches current odds from SharpAPI, following pagination to completion."""

    def __init__(self, league: str = "nfl") -> None:
        self.league = league

    def fetch_raw(self) -> pl.DataFrame:
        """Return every current main-line odds row for ``self.league``.

        Adds a ``pulled_at`` column (UTC, one instant for the whole call)
        recording when this fetch ran.
        """
        pulled_at = datetime.now(UTC)
        rows: list[dict[str, object]] = []
        cursor: str | None = None

        with httpx.Client(headers={"X-API-Key": sharpapi_key()}, timeout=30.0) as client:
            while True:
                params: dict[str, str] = {"league": self.league, "market": _MARKETS}
                if cursor is not None:
                    params["cursor"] = cursor

                response = client.get(_BASE_URL, params=params)
                if response.status_code == 401:
                    raise RuntimeError(
                        "SharpAPI rejected the API key (401). Check SHARPAPI_KEY."
                    )
                if response.status_code == 429:
                    raise RuntimeError(
                        "SharpAPI rate limit exceeded (429). "
                        "Free tier allows 12 requests/minute."
                    )
                response.raise_for_status()

                payload = response.json()
                rows.extend(row for row in payload["data"] if row["is_main_line"])

                pagination = payload["pagination"]
                if not pagination["has_more"]:
                    break
                cursor = pagination["next_cursor"]

        if rows:
            frame = pl.DataFrame(rows).select(_COLUMNS)
        else:
            frame = pl.DataFrame(schema={c: pl.Utf8 for c in _COLUMNS})

        return frame.with_columns(pl.lit(pulled_at).alias("pulled_at"))
