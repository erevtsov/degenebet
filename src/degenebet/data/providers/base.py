"""Provider protocol — the interface data/cache.py's load_or_fetch expects."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class Provider(Protocol):
    """Fetches a fresh, full-replacement snapshot from a vendor.

    ``fetch_raw()`` takes no arguments beyond what the implementation was
    constructed with (e.g. league, API key). The returned DataFrame must
    include a ``pulled_at`` column (``pl.Datetime``, UTC) recording when
    the snapshot was taken — ``data/cache.py`` uses it to judge freshness.
    """

    def fetch_raw(self) -> pl.DataFrame:
        """Return a fresh DataFrame snapshot from the vendor."""
        ...
