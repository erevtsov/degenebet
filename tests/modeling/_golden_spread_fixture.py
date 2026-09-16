"""Deterministic synthetic model_table for golden regression tests.

Fixed-seed, not real fetched data — avoids any dependency on nflverse data
potentially changing over time and keeps golden values exactly
reproducible. Mirrors waypoint's np.random.default_rng(seed=42) convention
for reproducible test data.
"""

from __future__ import annotations

import numpy as np
import polars as pl

_FEATURE_COLUMNS = [
    "home_offense_epa",
    "home_defense_epa_allowed",
    "home_turnover_margin",
    "away_offense_epa",
    "away_defense_epa_allowed",
    "away_turnover_margin",
]


def golden_model_table() -> pl.DataFrame:
    rng = np.random.default_rng(seed=42)
    n = 40
    features = {col: rng.normal(loc=0.0, scale=1.0, size=n) for col in _FEATURE_COLUMNS}
    # A fixed, made-up "true" relationship plus noise, so the fit is
    # well-conditioned and reproducible — not meant to reflect real football.
    true_coefs = np.array([8.0, -6.0, 3.0, -8.0, 6.0, -3.0])
    design = np.column_stack([features[c] for c in _FEATURE_COLUMNS])
    noise = rng.normal(loc=0.0, scale=2.0, size=n)
    result = design @ true_coefs + noise
    # A noisy "market" around the true result.
    spread_line = -result + rng.normal(loc=0.0, scale=3.0, size=n)

    data = {**features, "result": result, "spread_line": spread_line}
    return pl.DataFrame(data)
