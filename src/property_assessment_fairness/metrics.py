"""Evaluation metrics and fairness diagnostics for property valuation models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Root mean squared error."""
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual_arr - predicted_arr) ** 2)))


def mape(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Mean absolute percentage error, ignoring zero-valued actuals."""
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    mask = actual_arr != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual_arr[mask] - predicted_arr[mask]) / actual_arr[mask])))


def prop_overestimated(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> float:
    """Share of observations where the model prediction is above the actual value."""
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    return float(np.mean(predicted_arr > actual_arr))


def assessment_bias_by_price_tier(
    actual_log_price: np.ndarray | pd.Series,
    predicted_log_price: np.ndarray | pd.Series,
    *,
    tiers: int = 2,
) -> pd.DataFrame:
    """Summarize error patterns across sale-price tiers.

    The returned table is designed for fairness review. For tax assessment,
    systematic overestimation in lower-priced tiers can indicate regressive
    burden even when aggregate model RMSE looks acceptable.
    """
    df = pd.DataFrame(
        {
            "actual_log_price": np.asarray(actual_log_price, dtype=float),
            "predicted_log_price": np.asarray(predicted_log_price, dtype=float),
        }
    ).dropna()

    df["actual_price"] = np.exp(df["actual_log_price"])
    df["predicted_price"] = np.exp(df["predicted_log_price"])
    df["tier"] = pd.qcut(df["actual_log_price"], q=tiers, duplicates="drop")

    summary = (
        df.groupby("tier", observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "n": len(group),
                    "rmse_sale_price": rmse(group["actual_price"], group["predicted_price"]),
                    "rmse_log_price": rmse(
                        group["actual_log_price"], group["predicted_log_price"]
                    ),
                    "mape_sale_price": mape(group["actual_price"], group["predicted_price"]),
                    "prop_overestimated": prop_overestimated(
                        group["actual_price"], group["predicted_price"]
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    summary["tier"] = summary["tier"].astype(str)
    return summary
