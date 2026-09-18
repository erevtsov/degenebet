from __future__ import annotations

from collections.abc import Iterator

import polars as pl
import pytest

from degenebet.modeling.splits import SingleSplit, Split, WalkForwardSplit, iterate_folds


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


def _walk_forward_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2020, 2020, 2020, 2021, 2021, 2021],
            "week": [1, 2, 3, 1, 2, 3],
            "gameweek": [202001, 202002, 202003, 202101, 202102, 202103],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )


def test_walk_forward_split_produces_one_fold_per_post_warmup_gameweek() -> None:
    strategy = WalkForwardSplit(warmup_seasons=1)

    splits = list(strategy.splits(_walk_forward_data()))

    assert [s.test["gameweek"].to_list() for s in splits] == [[202101], [202102], [202103]]


def test_walk_forward_split_train_expands_and_never_includes_current_or_future_gameweek() -> None:
    strategy = WalkForwardSplit(warmup_seasons=1)

    splits = list(strategy.splits(_walk_forward_data()))

    assert splits[0].train["gameweek"].to_list() == [202001, 202002, 202003]
    assert splits[1].train["gameweek"].to_list() == [202001, 202002, 202003, 202101]
    assert splits[2].train["gameweek"].to_list() == [202001, 202002, 202003, 202101, 202102]
    # Never includes the test gameweek itself or anything after it.
    for split in splits:
        assert split.train["gameweek"].max() < split.test["gameweek"][0]


def test_walk_forward_split_first_test_fold_is_first_gameweek_of_correct_season() -> None:
    strategy = WalkForwardSplit(warmup_seasons=1)

    first = next(iter(strategy.splits(_walk_forward_data())))

    assert first.test["gameweek"].to_list() == [202101]


def test_walk_forward_split_raises_when_warmup_leaves_no_train_data() -> None:
    strategy = WalkForwardSplit(warmup_seasons=0)

    with pytest.raises(ValueError, match="No rows in data before gameweek"):
        list(strategy.splits(_walk_forward_data()))
