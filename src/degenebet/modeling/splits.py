"""Split/fold orchestration: SplitStrategy implementations partition data
into (train, test) pairs; iterate_folds (Task 2) is the one shared
fit/predict loop Efficacy and, later, Backtest both consume. See
docs/superpowers/specs/2026-09-18-modeling-efficacy-foundation-design.md.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple, Protocol

import polars as pl


class Split(NamedTuple):
    """One (train, test) partition of a model table."""

    train: pl.DataFrame
    test: pl.DataFrame


class SplitStrategy(Protocol):
    def splits(self, data: pl.DataFrame) -> Iterator[Split]: ...


class SingleSplit:
    """One (train, test) pair by season list. Mirrors run_backtest's
    existing train_seasons/test_seasons filtering and leakage guards,
    extracted here rather than reimplemented."""

    def __init__(self, train_seasons: list[int], test_seasons: list[int]) -> None:
        self.train_seasons = train_seasons
        self.test_seasons = test_seasons

    def splits(self, data: pl.DataFrame) -> Iterator[Split]:
        """Raises ValueError if train_seasons and test_seasons overlap
        (leakage), or if the train filter is empty (nothing to fit on).
        An empty test filter is not an error (e.g. an in-progress season
        with no played games yet) -- yields a zero-row test frame."""
        overlap = set(self.train_seasons) & set(self.test_seasons)
        if overlap:
            raise ValueError(
                f"train_seasons and test_seasons overlap: {sorted(overlap)} — "
                "this would leak test-season data into training and inflate results."
            )

        train = data.filter(pl.col("season").is_in(self.train_seasons))
        test = data.filter(pl.col("season").is_in(self.test_seasons))

        if train.height == 0:
            raise ValueError(f"No rows in data for train_seasons={self.train_seasons}")

        yield Split(train=train, test=test)
