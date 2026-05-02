"""Lightweight verification for the portfolio refactor."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_assessment_fairness import (  # noqa: E402
    assessment_bias_by_price_tier,
    engineer_features,
    prop_overestimated,
    rmse,
)


def main() -> None:
    df = pd.DataFrame(
        {
            "Sale Price": [100_000, 150_000, 225_000, 300_000, 450_000, 600_000],
            "Estimate (Building)": [70_000, 110_000, 160_000, 210_000, 300_000, 400_000],
            "Building Square Feet": [900, 1100, 1300, 1600, 2100, 2600],
            "Full Baths": [1, 1, 2, 2, 3, 3],
            "Half Baths": [0, 1, 0, 1, 0, 1],
        }
    )

    x, y = engineer_features(df, remove_sale_price_outliers=False)
    assert list(x.columns) == [
        "Log Estimate (Building)",
        "Bathrooms",
        "Log Building Square Feet",
    ]
    assert len(x) == len(y) == len(df)

    design = np.column_stack([np.ones(len(x)), x.to_numpy()])
    weights, *_ = np.linalg.lstsq(design, y.to_numpy(), rcond=None)
    predictions = design @ weights
    assert np.isfinite(rmse(y, predictions))
    assert 0 <= prop_overestimated(y, predictions) <= 1

    diagnostics = assessment_bias_by_price_tier(y, predictions)
    assert {"n", "rmse_log_price", "prop_overestimated"}.issubset(diagnostics.columns)
    assert len(diagnostics) == 2

    test_x = engineer_features(df.drop(columns=["Sale Price"]), is_test_set=True)
    assert test_x.shape == x.shape

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
