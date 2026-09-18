from __future__ import annotations

import httpx
import polars as pl
import pytest
import respx

from degenebet.data.providers.sharpapi import SharpAPIProvider

_URL = "https://api.sharpapi.io/api/v1/odds"


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "draftkings_1_moneyline_PHI",
        "sportsbook": "draftkings",
        "event_id": "1",
        "sport": "football",
        "league": "nfl",
        "home_team": "PHI Eagles",
        "away_team": "DAL Cowboys",
        "market_type": "moneyline",
        "selection": "PHI Eagles",
        "selection_type": "home",
        "odds_american": -150,
        "odds_decimal": 1.667,
        "odds_probability": 0.6,
        "line": None,
        "is_main_line": True,
        "is_alternate_line": False,
        "event_start_time": "2026-09-21T17:00:00Z",
        "timestamp": "2026-09-14T02:10:24.125Z",
        "is_live": False,
    }
    row.update(overrides)
    return row


def _payload(
    rows: list[dict[str, object]],
    *,
    has_more: bool = False,
    next_cursor: str | None = None,
) -> dict[str, object]:
    return {
        "data": rows,
        "pagination": {
            "limit": 50,
            "offset": 0,
            "count": len(rows),
            "has_more": has_more,
            "next_cursor": next_cursor,
        },
        "updated_at": "2026-09-14T02:10:37.846Z",
    }


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARPAPI_KEY", "test-key")


@respx.mock
def test_fetch_raw_parses_single_page() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(200, json=_payload([_row()])))

    frame = SharpAPIProvider().fetch_raw()

    assert frame.height == 1
    assert frame["home_team"][0] == "PHI Eagles"
    assert frame["odds_american"][0] == -150
    assert "pulled_at" in frame.columns


@respx.mock
def test_fetch_raw_follows_pagination() -> None:
    route = respx.get(_URL)
    route.side_effect = [
        httpx.Response(200, json=_payload([_row(event_id="1")], has_more=True, next_cursor="abc")),
        httpx.Response(200, json=_payload([_row(event_id="2")])),
    ]

    frame = SharpAPIProvider().fetch_raw()

    assert sorted(frame["event_id"].to_list()) == ["1", "2"]


@respx.mock
def test_fetch_raw_drops_non_main_lines() -> None:
    rows = [_row(), _row(id="alt", is_main_line=False, line=-3.5)]
    respx.get(_URL).mock(return_value=httpx.Response(200, json=_payload(rows)))

    frame = SharpAPIProvider().fetch_raw()

    assert frame.height == 1


@respx.mock
def test_fetch_raw_raises_on_unauthorized() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(401, json={"error": "invalid key"}))

    with pytest.raises(RuntimeError, match="401"):
        SharpAPIProvider().fetch_raw()


@respx.mock
def test_fetch_raw_raises_on_rate_limit() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(429, json={"error": "rate limited"}))

    with pytest.raises(RuntimeError, match="429"):
        SharpAPIProvider().fetch_raw()


@respx.mock
def test_fetch_raw_handles_moneylines_before_spread_line() -> None:
    moneylines = [_row(event_id=str(i), id=f"ml_{i}") for i in range(100)]
    spread = _row(
        event_id="spread-1",
        id="spread_1",
        market_type="spread",
        selection_type="home",
        line=-3.5,
    )
    rows = [*moneylines, spread]
    respx.get(_URL).mock(return_value=httpx.Response(200, json=_payload(rows)))

    frame = SharpAPIProvider().fetch_raw()

    assert frame.height == 101
    spread_row = frame.filter(pl.col("event_id") == "spread-1")
    assert spread_row["line"][0] == -3.5


@respx.mock
def test_fetch_raw_handles_empty_response() -> None:
    respx.get(_URL).mock(return_value=httpx.Response(200, json=_payload([])))

    frame = SharpAPIProvider().fetch_raw()

    assert frame.height == 0
    assert "pulled_at" in frame.columns


@respx.mock
def test_fetch_raw_stops_on_has_more_with_null_cursor() -> None:
    route = respx.get(_URL)
    route.mock(
        return_value=httpx.Response(200, json=_payload([_row()], has_more=True, next_cursor=None))
    )

    frame = SharpAPIProvider().fetch_raw()

    assert frame.height == 1
    assert route.call_count == 1


@respx.mock
def test_fetch_raw_raises_on_missing_pagination_key() -> None:
    payload = {"data": [_row()]}
    respx.get(_URL).mock(return_value=httpx.Response(200, json=payload))

    with pytest.raises(RuntimeError, match="pagination"):
        SharpAPIProvider().fetch_raw()


def _headers(remaining: int, reset_at: float) -> dict[str, str]:
    return {
        "x-ratelimit-limit": "12",
        "x-ratelimit-remaining": str(remaining),
        "x-ratelimit-reset": str(reset_at),
    }


@respx.mock
def test_fetch_raw_sleeps_when_rate_limit_nearly_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_at = 1_700_000_100.0
    monkeypatch.setattr("degenebet.data.providers.sharpapi.time.time", lambda: 1_700_000_095.0)
    sleeps: list[float] = []
    monkeypatch.setattr("degenebet.data.providers.sharpapi.time.sleep", sleeps.append)

    route = respx.get(_URL)
    route.side_effect = [
        httpx.Response(
            200,
            json=_payload([_row(event_id="1")], has_more=True, next_cursor="abc"),
            headers=_headers(remaining=1, reset_at=reset_at),
        ),
        httpx.Response(
            200,
            json=_payload([_row(event_id="2")]),
            headers=_headers(remaining=11, reset_at=reset_at),
        ),
    ]

    frame = SharpAPIProvider().fetch_raw()

    assert sorted(frame["event_id"].to_list()) == ["1", "2"]
    assert sleeps == [5.0]  # reset_at (…100) - fake now (…095)


@respx.mock
def test_fetch_raw_does_not_sleep_with_budget_remaining(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("degenebet.data.providers.sharpapi.time.sleep", sleeps.append)

    respx.get(_URL).mock(
        return_value=httpx.Response(
            200, json=_payload([_row()]), headers=_headers(remaining=11, reset_at=9_999_999_999.0)
        )
    )

    SharpAPIProvider().fetch_raw()

    assert sleeps == []


@respx.mock
def test_fetch_raw_tolerates_missing_rate_limit_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("degenebet.data.providers.sharpapi.time.sleep", sleeps.append)

    respx.get(_URL).mock(return_value=httpx.Response(200, json=_payload([_row()])))

    frame = SharpAPIProvider().fetch_raw()

    assert frame.height == 1
    assert sleeps == []
