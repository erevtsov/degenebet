from __future__ import annotations

import polars as pl
import pytest

from degenebet.modeling.splits import SingleSplit


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
