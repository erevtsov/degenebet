"""Split/fold orchestration: SplitStrategy implementations partition data
into (train, test) pairs; iterate_folds (Task 2) is the one shared
fit/predict loop Efficacy and, later, Backtest both consume. See
docs/superpowers/specs/2026-09-18-modeling-efficacy-foundation-design.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
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


class Model(Protocol):
    def fit(self, train_data: pl.DataFrame) -> None: ...
    def predict(self, data: pl.DataFrame) -> pl.DataFrame: ...


class FoldPredictions(NamedTuple):
    """One fold's fit model plus its predictions on both sides of the
    split. `in_sample`/`out_of_sample` are the precomputed,
    single-source-of-truth prediction frames -- Efficacy (and, later,
    Backtest) score these directly rather than re-deriving predictions
    themselves. `model` is for whatever predictions alone can't answer
    (residuals, learned weights, any other diagnostic); it costs nothing
    extra to expose since it's already been created and fit either way."""

    model: Model
    in_sample: pl.DataFrame
    out_of_sample: pl.DataFrame


def iterate_folds(
    data: pl.DataFrame,
    split_strategy: SplitStrategy,
    model_factory: Callable[[], Model],
) -> Iterator[FoldPredictions]:
    """The one shared fit/predict orchestration -- neither Efficacy nor
    (later) Backtest owns fitting or fold-looping itself. model_factory
    (not a fixed Model instance) matters because a stateful Model like
    SpreadModel needs a fresh instance per fold so fitted state never
    leaks across folds."""
    for split in split_strategy.splits(data):
        model = model_factory()
        model.fit(split.train)
        yield FoldPredictions(
            model=model,
            in_sample=model.predict(split.train),
            out_of_sample=model.predict(split.test),
        )
