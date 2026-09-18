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
    """Produces one or more (train, test) partitions of a model table."""

    def splits(self, data: pl.DataFrame) -> Iterator[Split]: ...


class SingleSplit:
    """One (train, test) pair by season list, with a leakage guard
    against overlapping train/test seasons."""

    def __init__(self, train_seasons: list[int], test_seasons: list[int]) -> None:
        self.train_seasons = train_seasons
        self.test_seasons = test_seasons

    def splits(self, data: pl.DataFrame) -> Iterator[Split]:
        """Raises ValueError if train_seasons and test_seasons overlap
        (leakage), or if the train filter is empty (nothing to fit on).
        An empty test filter is not an error here -- yields a zero-row
        test frame (e.g. for an in-progress season with no played games
        yet). Note: iterate_folds does not special-case this -- a
        zero-row test frame is passed straight to model.predict(), whose
        behavior on empty input is up to the concrete Model (e.g.
        SpreadModel raises, via sklearn, since LinearRegression requires
        at least one sample)."""
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


class WalkForwardSplit:
    """One (train, test) pair per gameweek from the first post-warmup
    season onward, expanding window (train grows every fold, never
    shrinks or slides). Does not filter unplayed rows -- same rule as
    SingleSplit; a fold that walks into a partially-played or future week
    just has null-result rows in its test, which Backtest's own guard
    catches if that fold ever reaches run()/run_folds() unfiltered."""

    def __init__(self, warmup_seasons: int) -> None:
        self.warmup_seasons = warmup_seasons

    def splits(self, data: pl.DataFrame) -> Iterator[Split]:
        seasons = sorted(data["season"].unique().to_list())
        first_test_season = seasons[0] + self.warmup_seasons
        gameweeks = sorted(
            data.filter(pl.col("season") >= first_test_season)["gameweek"].unique().to_list()
        )
        for gw in gameweeks:
            train = data.filter(pl.col("gameweek") < gw)
            test = data.filter(pl.col("gameweek") == gw)
            if train.height == 0:
                raise ValueError(
                    f"No rows in data before gameweek={gw} "
                    f"(warmup_seasons={self.warmup_seasons} too small?)"
                )
            yield Split(train=train, test=test)


class Model(Protocol):
    """Minimal fit/predict interface iterate_folds depends on -- any
    concrete predictor (e.g. SpreadModel) satisfies this structurally,
    with no explicit inheritance required."""

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
