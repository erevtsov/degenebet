from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from degenebet.modeling.spread_model import SpreadModel

_FEATURE_COLUMNS = [
    "home_offense_epa",
    "home_defense_epa_allowed",
    "home_turnover_margin",
    "away_offense_epa",
    "away_defense_epa_allowed",
    "away_turnover_margin",
]


def _noiseless_table() -> pl.DataFrame:
    # result = 10 * home_offense_epa exactly; every other feature is 0.
    home_offense = [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, 2.5, 0.25, -0.25]
    zeros = [0.0] * len(home_offense)
    return pl.DataFrame(
        {
            "home_offense_epa": home_offense,
            "home_defense_epa_allowed": zeros,
            "home_turnover_margin": zeros,
            "away_offense_epa": zeros,
            "away_defense_epa_allowed": zeros,
            "away_turnover_margin": zeros,
            "result": [10.0 * v for v in home_offense],
        }
    )


def _noisy_table_with_spread() -> pl.DataFrame:
    table = _noiseless_table()
    # Add small deterministic noise so residual_std is nonzero.
    noise = [0.3, -0.2, 0.1, -0.4, 0.2, -0.1, 0.3, -0.3, 0.1, -0.2]
    table = table.with_columns((pl.col("result") + pl.Series(noise)).alias("result"))
    return table.with_columns(pl.Series("spread_line", [0.0] * table.height))


def test_predict_matches_noiseless_linear_relationship() -> None:
    model = SpreadModel()
    table = _noiseless_table()
    model.fit(table)

    predicted = model.predict(table)

    for expected, actual in zip(table["result"].to_list(), predicted["predicted_result"].to_list()):
        assert actual == pytest.approx(expected, abs=1e-6)


def test_cover_probability_is_bounded_and_present() -> None:
    model = SpreadModel()
    table = _noisy_table_with_spread()
    model.fit(table)

    result = model.cover_probability(table)

    assert "home_cover_probability" in result.columns
    for p in result["home_cover_probability"].to_list():
        assert 0.0 <= p <= 1.0


def test_cover_probability_before_fit_raises() -> None:
    model = SpreadModel()
    table = _noisy_table_with_spread()

    with pytest.raises(RuntimeError, match="fit"):
        model.cover_probability(table)


def test_cover_probability_calls_predict_internally_if_missing() -> None:
    model = SpreadModel()
    table = _noisy_table_with_spread()
    model.fit(table)

    result = model.cover_probability(table.select(_FEATURE_COLUMNS + ["spread_line"]))

    assert "predicted_result" in result.columns
    assert "home_cover_probability" in result.columns


class _ConstantRegressor:
    """Trivial regressor always predicting a fixed constant — a test double
    proving SpreadModel actually delegates to an injected regressor rather
    than hardcoding LinearRegression. Matches RegressorProtocol's shape
    exactly (ndarray in, ndarray out) so it type-checks where a
    RegressorProtocol is expected."""

    def __init__(self, constant: float) -> None:
        self._constant = constant

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> _ConstantRegressor:
        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.full(X.shape[0], self._constant)


def test_spread_model_accepts_injected_regressor() -> None:
    model = SpreadModel(model=_ConstantRegressor(constant=3.5))
    table = _noiseless_table()
    model.fit(table)

    predicted = model.predict(table)

    for value in predicted["predicted_result"].to_list():
        assert value == pytest.approx(3.5)


def test_spread_model_defaults_to_linear_regression() -> None:
    from sklearn.linear_model import LinearRegression

    model = SpreadModel()

    assert isinstance(model.model, LinearRegression)


class _PerfectRegressor:
    """Test double that memorizes and replays the exact fit targets,
    forcing residuals of exactly 0.0 — used to force a degenerate
    (residual_std == 0) fit deterministically, without relying on
    floating-point luck from a "nearly perfect" real regression."""

    def __init__(self) -> None:
        self._y: npt.NDArray[np.float64] | None = None

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> _PerfectRegressor:
        self._y = y
        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        assert self._y is not None
        return self._y


def test_fit_raises_on_zero_residual_std() -> None:
    model = SpreadModel(model=_PerfectRegressor())
    table = _noiseless_table()

    with pytest.raises(ValueError, match="degenerate residual_std"):
        model.fit(table)


def test_fit_raises_on_single_row_train_set() -> None:
    # np.std(residuals, ddof=1) on a 1-row train set is NaN (n - ddof = 0);
    # numpy warns about this expected division before we raise on it.
    model = SpreadModel()
    table = _noiseless_table().head(1)

    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="degenerate residual_std"):
        model.fit(table)


def test_fit_raises_on_null_feature_column() -> None:
    model = SpreadModel()
    table = _noiseless_table()
    table = table.with_columns(
        pl.Series("home_offense_epa", [None, *table["home_offense_epa"].to_list()[1:]])
    )

    with pytest.raises(ValueError, match="null values in feature columns.*home_offense_epa"):
        model.fit(table)


def test_predict_raises_on_null_feature_column() -> None:
    model = SpreadModel()
    table = _noiseless_table()
    model.fit(table)
    table = table.with_columns(
        pl.Series("away_turnover_margin", [None, *table["away_turnover_margin"].to_list()[1:]])
    )

    with pytest.raises(ValueError, match="null values in feature columns.*away_turnover_margin"):
        model.predict(table)


def test_residual_std_is_none_before_fit() -> None:
    model = SpreadModel()

    assert model.residual_std is None


def test_residual_std_matches_the_value_fit_computed() -> None:
    model = SpreadModel()
    table = _noisy_table_with_spread()

    model.fit(table)

    assert model.residual_std is not None
    assert model.residual_std > 0.0
