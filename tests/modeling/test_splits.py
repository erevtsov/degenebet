from __future__ import annotations

from collections.abc import Iterator

import polars as pl
import pytest

from degenebet.modeling.splits import SingleSplit, Split, iterate_folds


def test_singlesplit_partitions_by_season() -> None:
    data = pl.DataFrame({"season": [2020, 2020, 2021, 2022], "value": [1, 2, 3, 4]})
    strategy = SingleSplit(train_seasons=[2020], test_seasons=[2021])

    splits = list(strategy.splits(data))

    assert len(splits) == 1
    assert splits[0].train["value"].to_list() == [1, 2]
    assert splits[0].test["value"].to_list() == [3]


def test_singlesplit_raises_on_overlapping_seasons() -> None:
    data = pl.DataFrame({"season": [2020, 2021], "value": [1, 2]})
    strategy = SingleSplit(train_seasons=[2020, 2021], test_seasons=[2021])

    with pytest.raises(ValueError, match="overlap"):
        list(strategy.splits(data))


def test_singlesplit_raises_on_empty_train() -> None:
    data = pl.DataFrame({"season": [2020], "value": [1]})
    strategy = SingleSplit(train_seasons=[2099], test_seasons=[2020])

    with pytest.raises(ValueError, match="No rows"):
        list(strategy.splits(data))


def test_singlesplit_yields_zero_row_test_frame_when_test_seasons_have_no_data() -> None:
    data = pl.DataFrame({"season": [2020], "value": [1]})
    strategy = SingleSplit(train_seasons=[2020], test_seasons=[2099])

    splits = list(strategy.splits(data))

    assert len(splits) == 1
    assert splits[0].test.height == 0


class _TwoFoldSplitStrategy:
    """Test double yielding two fixed (train, test) splits from a `fold`
    column -- standing in for a real multi-fold strategy like the future
    WalkForwardSplit, which doesn't exist yet. Exists only to prove
    iterate_folds creates a fresh model per fold rather than reusing one
    across folds; SingleSplit itself always yields exactly one fold, so it
    can't exercise that behavior."""

    def splits(self, data: pl.DataFrame) -> Iterator[Split]:
        yield Split(train=data.filter(pl.col("fold") == 0), test=data.filter(pl.col("fold") == 1))
        yield Split(train=data.filter(pl.col("fold") == 1), test=data.filter(pl.col("fold") == 0))


class _RecordingModel:
    """Fake Model recording fit() calls and tagging predict() output with
    this instance's identity, so a test can assert iterate_folds passes
    the right frames and creates a fresh instance per fold."""

    def __init__(self) -> None:
        self.fit_calls: list[pl.DataFrame] = []

    def fit(self, train_data: pl.DataFrame) -> None:
        self.fit_calls.append(train_data)

    def predict(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns(pl.lit(id(self)).alias("model_id"))


def test_iterate_folds_uses_a_fresh_model_per_fold() -> None:
    data = pl.DataFrame({"fold": [0, 1], "x": [1, 2]})

    results = list(iterate_folds(data, _TwoFoldSplitStrategy(), _RecordingModel))

    assert len(results) == 2
    assert results[0].model is not results[1].model
    assert isinstance(results[0].model, _RecordingModel)
    assert results[0].model.fit_calls[0].equals(data.filter(pl.col("fold") == 0))
    assert results[1].model.fit_calls[0].equals(data.filter(pl.col("fold") == 1))
    assert results[0].in_sample["model_id"].to_list() == [id(results[0].model)]
    assert results[0].out_of_sample["model_id"].to_list() == [id(results[0].model)]
