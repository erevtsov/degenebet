"""Efficacy: pure prediction-quality scoring -- predicted_result vs.
result only, never spread_line (that's exclusively a Backtest concern).
See docs/superpowers/specs/2026-09-18-modeling-efficacy-foundation-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


@dataclass(frozen=True)
class EfficacyResult:
    metrics: dict[str, float]
    by_fold: pl.DataFrame | None


def _compute_metrics(predictions: pl.DataFrame) -> dict[str, float]:
    """predicted_result vs. result only -- spread_line never enters this
    computation. Raises ValueError on an empty frame or a null result
    (loud, not silent NaN metrics), same philosophy as SpreadModel's
    null-feature guard."""
    if predictions.height == 0:
        raise ValueError("Cannot evaluate an empty predictions frame.")
    if predictions["result"].null_count() > 0:
        raise ValueError(
            "predictions has null values in 'result' -- can't score against missing "
            "ground truth. Filter to played games before calling evaluate()."
        )

    predicted = predictions["predicted_result"].to_numpy()
    actual = predictions["result"].to_numpy()

    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    r_squared = float(r2_score(actual, predicted))
    # sign(0) == 0, so a tied game (result == 0, rare but legal in the NFL
    # regular season) only counts as a correct call if predicted_result is
    # exactly 0.0 too -- not worth special-casing for how rare it is.
    directional_accuracy = float(np.mean(np.sign(predicted) == np.sign(actual)))
    information_coefficient = float(pearsonr(predicted, actual)[0])

    return {
        "rmse": rmse,
        "r_squared": r_squared,
        "directional_accuracy": directional_accuracy,
        "information_coefficient": information_coefficient,
    }


class Efficacy:
    """Pure scorer -- no fitting, no fold-looping, no multi-configuration
    search. predicted_result vs. result only; spread_line never enters
    this class (that's exclusively a Backtest concern)."""

    def evaluate(self, predictions: pl.DataFrame) -> EfficacyResult:
        """Scores one frame. by_fold is always None here -- a single frame
        has no fold concept, so there's nothing to build a breakdown
        from."""
        return EfficacyResult(metrics=_compute_metrics(predictions), by_fold=None)
