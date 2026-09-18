from __future__ import annotations

import polars as pl
import pytest

from degenebet.modeling.efficacy import Efficacy


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
