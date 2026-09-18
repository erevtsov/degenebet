"""Efficacy: pure prediction-quality scoring -- predicted_result vs.
result only, never spread_line (that's exclusively a Backtest concern).
See docs/superpowers/specs/2026-09-18-modeling-efficacy-foundation-design.md.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from degenebet.modeling.splits import FoldPredictions


@dataclass(frozen=True)
class EfficacyResult:
    """metrics is pooled and out-of-sample only (never averaged across
    folds); by_fold is long format (one row per fold x sample) and the
    only place in-sample figures appear, for the walk-forward-efficiency
    comparison."""

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

    def evaluate_folds(self, folds: Iterable[FoldPredictions]) -> EfficacyResult:
        """Consumes iterate_folds's FoldPredictions stream. by_fold is long
        format: one row per (fold, sample) pair. metrics is computed by
        concatenating every fold's out_of_sample frame and scoring once --
        never by averaging each fold's already-computed metrics, so an
        uneven fold size (e.g. from a future WalkForwardSplit) doesn't
        mis-weight the result."""
        materialized = list(folds)
        if not materialized:
            raise ValueError("Cannot evaluate an empty folds stream.")

        by_fold_rows: list[dict[str, float | int | str]] = []
        out_of_sample_frames: list[pl.DataFrame] = []
        for fold_index, fold in enumerate(materialized):
            by_fold_rows.append(
                {"fold": fold_index, "sample": "in_sample", **_compute_metrics(fold.in_sample)}
            )
            by_fold_rows.append(
                {
                    "fold": fold_index,
                    "sample": "out_of_sample",
                    **_compute_metrics(fold.out_of_sample),
                }
            )
            out_of_sample_frames.append(fold.out_of_sample)

        pooled = pl.concat(out_of_sample_frames, how="vertical")
        return EfficacyResult(metrics=_compute_metrics(pooled), by_fold=pl.DataFrame(by_fold_rows))
