"""Linear regression (by default) predicting NFL game margin from rolling
team features. SpreadModel is composed with a RegressorProtocol rather than
hardcoding LinearRegression, so a future sub-project can swap in a
different sklearn-compatible estimator without changing this class."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

_FEATURE_COLUMNS = [
    "home_offense_epa",
    "home_defense_epa_allowed",
    "home_turnover_margin",
    "away_offense_epa",
    "away_defense_epa_allowed",
    "away_turnover_margin",
]


class RegressorProtocol(Protocol):
    """Minimal sklearn-compatible regressor interface SpreadModel depends on."""

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> object: ...
    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...


def _raise_if_null_features(model_table: pl.DataFrame) -> None:
    """DataAccess's team_data widening always left-joins (see access.py's
    _widen_with_team_data), so a game where one team lacks enough rolling
    history comes through with null feature columns rather than being
    dropped -- by design, so the drop decision is the caller's explicit
    choice (e.g. model_table.drop_nulls(subset=[...])), not something
    silently baked into a shared widening/join step. Raising here (rather
    than letting sklearn's generic "Input X contains NaN" surface, or
    fitting/predicting on NaN silently) makes that unmade choice loud and
    specific instead of a confusing downstream failure."""
    null_counts = model_table.select(_FEATURE_COLUMNS).null_count()
    null_columns = [c for c in _FEATURE_COLUMNS if null_counts[c][0] > 0]
    if null_columns:
        raise ValueError(
            f"model_table has null values in feature columns {null_columns} -- "
            "these rows can't be fit/predicted on. Filter them out explicitly "
            "(e.g. model_table.drop_nulls(subset=[...])) before calling "
            "SpreadModel.fit()/predict()."
        )


class SpreadModel:
    """Predicts `result` from 6 rolling team features via an injected
    regressor (LinearRegression by default), plus a cover probability
    derived from the training residual spread."""

    def __init__(self, model: RegressorProtocol | None = None) -> None:
        self.model: RegressorProtocol = model if model is not None else LinearRegression()
        self._residual_std: float | None = None

    def fit(self, model_table: pl.DataFrame) -> None:
        """Fits the injected regressor on the 6 feature columns against `result`."""
        _raise_if_null_features(model_table)
        features = model_table.select(_FEATURE_COLUMNS).to_numpy()
        target = model_table["result"].to_numpy()
        self.model.fit(features, target)
        residuals = target - self.model.predict(features)
        residual_std = float(np.std(residuals, ddof=1))
        # A non-finite (e.g. NaN from a 1-row train set, ddof=1) or
        # non-positive (e.g. ~0 from a perfect/degenerate fit) residual_std
        # would silently produce NaN/inf home_cover_probability later.
        if not np.isfinite(residual_std) or residual_std <= 0:
            raise ValueError(
                f"SpreadModel.fit() produced a degenerate residual_std={residual_std!r}; "
                "training data is likely too small or perfectly collinear."
            )
        self._residual_std = residual_std

    def predict(self, model_table: pl.DataFrame) -> pl.DataFrame:
        """Returns model_table with a new `predicted_result` column."""
        _raise_if_null_features(model_table)
        features = model_table.select(_FEATURE_COLUMNS).to_numpy()
        predicted = self.model.predict(features)
        return model_table.with_columns(pl.Series("predicted_result", predicted))

    def cover_probability(self, model_table: pl.DataFrame) -> pl.DataFrame:
        """Requires `spread_line` present. Adds `home_cover_probability` via
        normal_cdf((predicted_result - spread_line) / residual_std)."""
        if self._residual_std is None:
            raise RuntimeError("SpreadModel.fit() must be called before cover_probability().")
        if "predicted_result" not in model_table.columns:
            model_table = self.predict(model_table)
        edge = (model_table["predicted_result"] - model_table["spread_line"]) / self._residual_std
        probabilities = norm.cdf(edge.to_numpy())
        return model_table.with_columns(pl.Series("home_cover_probability", probabilities))
