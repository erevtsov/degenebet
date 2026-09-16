"""Golden regression test: pins SpreadModel's fitted output against a fixed
synthetic dataset. A failure here means the math actually changed — verify
the new values are intentional before updating the pinned constants below;
never update them reflexively just to make the test pass.
"""

from __future__ import annotations

import pytest

from degenebet.modeling.spread_model import SpreadModel
from tests.modeling._golden_spread_fixture import golden_model_table

# Pinned from a real run against the fixed-seed fixture (see plan Task 4 Step 3).
_EXPECTED_COEF = [
    7.448124339554307,
    -5.828832058450803,
    2.4571239988894535,
    -8.623437367603884,
    5.937314628509586,
    -3.4153290098833335,
]
_EXPECTED_INTERCEPT = -0.032212551343548945
_EXPECTED_RESIDUAL_STD = 1.9837770914273822
_EXPECTED_FIRST_5_PREDICTED = [
    0.9703443080571101,
    -23.229413776204183,
    18.975206856577557,
    -14.114939210917491,
    -13.403376561399295,
]
_EXPECTED_FIRST_5_COVER_PROB = [
    0.9581609298386047,
    0.008887840931952315,
    0.9977172861304908,
    0.930140185614296,
    0.0023877966037175323,
]


def test_spread_model_golden_fit_and_predict() -> None:
    table = golden_model_table()
    model = SpreadModel()
    model.fit(table)

    for actual, expected in zip(model.model.coef_.tolist(), _EXPECTED_COEF):
        assert actual == pytest.approx(expected, abs=1e-6)
    assert float(model.model.intercept_) == pytest.approx(_EXPECTED_INTERCEPT, abs=1e-6)
    assert model._residual_std == pytest.approx(_EXPECTED_RESIDUAL_STD, abs=1e-6)

    predicted = model.predict(table.head(5))
    predicted_values = predicted["predicted_result"].to_list()
    for actual, expected in zip(predicted_values, _EXPECTED_FIRST_5_PREDICTED):
        assert actual == pytest.approx(expected, abs=1e-6)

    cover = model.cover_probability(table.head(5))
    cover_values = cover["home_cover_probability"].to_list()
    for actual, expected in zip(cover_values, _EXPECTED_FIRST_5_COVER_PROB):
        assert actual == pytest.approx(expected, abs=1e-6)
