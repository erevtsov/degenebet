from __future__ import annotations

import polars as pl
import pytest

from degenebet.modeling.efficacy import Efficacy
from degenebet.modeling.splits import FoldPredictions, SingleSplit, iterate_folds
from degenebet.modeling.spread_model import SpreadModel


def test_evaluate_computes_rmse_r_squared_directional_accuracy_and_ic() -> None:
    predictions = pl.DataFrame(
        {"predicted_result": [3.0, -2.0, 5.0, 0.0], "result": [4.0, -1.0, 5.0, 1.0]}
    )

    result = Efficacy().evaluate(predictions)

    assert result.metrics["rmse"] == pytest.approx(0.8660254037844386)
    assert result.metrics["r_squared"] == pytest.approx(0.8681318681318682)
    assert result.metrics["directional_accuracy"] == pytest.approx(0.75)
    assert result.metrics["information_coefficient"] == pytest.approx(0.9927741970308563)
    assert result.by_fold is None


def test_evaluate_raises_on_null_result() -> None:
    predictions = pl.DataFrame(
        {"predicted_result": [1.0, 2.0], "result": [1.0, None]},
        schema={"predicted_result": pl.Float64, "result": pl.Float64},
    )

    with pytest.raises(ValueError, match="null values"):
        Efficacy().evaluate(predictions)


def test_evaluate_raises_on_empty_frame() -> None:
    predictions = pl.DataFrame(
        {"predicted_result": [], "result": []},
        schema={"predicted_result": pl.Float64, "result": pl.Float64},
    )

    with pytest.raises(ValueError, match="empty"):
        Efficacy().evaluate(predictions)


def _fold(
    *,
    in_sample_predicted: list[float],
    in_sample_result: list[float],
    out_of_sample_predicted: list[float],
    out_of_sample_result: list[float],
) -> FoldPredictions:
    # SpreadModel() here is an unfitted placeholder -- evaluate_folds never
    # reads FoldPredictions.model, only .in_sample/.out_of_sample, so its
    # exact identity doesn't matter for these tests.
    return FoldPredictions(
        model=SpreadModel(),
        in_sample=pl.DataFrame(
            {"predicted_result": in_sample_predicted, "result": in_sample_result}
        ),
        out_of_sample=pl.DataFrame(
            {"predicted_result": out_of_sample_predicted, "result": out_of_sample_result}
        ),
    )


def _two_fixed_folds() -> list[FoldPredictions]:
    # Deliberately different-sized out_of_sample frames (2 rows, 4 rows) --
    # a fold that mis-weights by averaging per-fold metrics instead of
    # pooling would produce a different (and wrong) rmse/r_squared/ic than
    # concatenating both frames and scoring once.
    fold_1 = _fold(
        in_sample_predicted=[5.0, 6.0, 7.0],
        in_sample_result=[5.0, 6.5, 6.0],
        out_of_sample_predicted=[1.0, 2.0],
        out_of_sample_result=[1.0, 3.0],
    )
    fold_2 = _fold(
        in_sample_predicted=[50.0, 60.0],
        in_sample_result=[55.0, 58.0],
        out_of_sample_predicted=[10.0, 20.0, 30.0, 40.0],
        out_of_sample_result=[12.0, 18.0, 33.0, 41.0],
    )
    return [fold_1, fold_2]


def test_evaluate_folds_pools_out_of_sample_metrics_not_averages_per_fold() -> None:
    result = Efficacy().evaluate_folds(_two_fixed_folds())

    # Pooled (concatenate both out_of_sample frames, score once). NOT the
    # naive average of each fold's own out_of_sample rmse (0.7071067811865476
    # for fold 1, 2.1213203435596424 for fold 2 -> average 1.414213562373095),
    # which would mis-weight the 2-row fold equally against the 4-row fold.
    assert result.metrics["rmse"] == pytest.approx(1.7795130420052185)
    assert result.metrics["r_squared"] == pytest.approx(0.9854294478527608)
    assert result.metrics["directional_accuracy"] == pytest.approx(1.0)
    assert result.metrics["information_coefficient"] == pytest.approx(0.9945095645744224)


def test_evaluate_folds_builds_long_format_by_fold_table() -> None:
    result = Efficacy().evaluate_folds(_two_fixed_folds())

    assert result.by_fold is not None
    assert result.by_fold.height == 4  # 2 folds x (in_sample, out_of_sample)
    assert sorted(result.by_fold["fold"].to_list()) == [0, 0, 1, 1]
    assert sorted(result.by_fold["sample"].unique().to_list()) == [
        "in_sample",
        "out_of_sample",
    ]
    fold_0_out_of_sample = result.by_fold.filter(
        (pl.col("fold") == 0) & (pl.col("sample") == "out_of_sample")
    ).row(0, named=True)
    assert fold_0_out_of_sample["rmse"] == pytest.approx(0.7071067811865476)
    fold_1_in_sample = result.by_fold.filter(
        (pl.col("fold") == 1) & (pl.col("sample") == "in_sample")
    ).row(0, named=True)
    assert fold_1_in_sample["rmse"] == pytest.approx(3.8078865529319543)


def test_evaluate_folds_raises_on_empty_folds_stream() -> None:
    with pytest.raises(ValueError, match="empty"):
        Efficacy().evaluate_folds([])


def test_evaluate_folds_on_a_single_fold_matches_evaluate_on_its_out_of_sample_frame() -> None:
    model_table = pl.DataFrame(
        {
            "season": [2000, 2000, 2000, 2001, 2001],
            "week": [1, 2, 3, 1, 2],
            "home_offense_epa": [0.1, -0.2, 0.3, 0.2, -0.1],
            "home_defense_epa_allowed": [0.0, 0.1, -0.1, 0.0, 0.2],
            "home_turnover_margin": [1.0, -1.0, 0.0, 1.0, -1.0],
            "away_offense_epa": [-0.1, 0.2, -0.3, -0.2, 0.1],
            "away_defense_epa_allowed": [0.1, 0.0, 0.2, -0.1, 0.0],
            "away_turnover_margin": [-1.0, 1.0, 0.0, -1.0, 1.0],
            "result": [3.0, -5.0, 7.0, 2.0, -4.0],
        }
    )
    split_strategy = SingleSplit(train_seasons=[2000], test_seasons=[2001])

    folds = list(iterate_folds(model_table, split_strategy, SpreadModel))
    assert len(folds) == 1

    via_evaluate_folds = Efficacy().evaluate_folds(folds).metrics
    via_evaluate = Efficacy().evaluate(folds[0].out_of_sample).metrics

    assert via_evaluate_folds == via_evaluate
