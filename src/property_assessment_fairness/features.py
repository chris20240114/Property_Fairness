"""Feature engineering for Cook County property assessment modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "Log Estimate (Building)",
    "Bathrooms",
    "Log Building Square Feet",
]


def remove_outliers_iqr(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return rows inside the 1.5 IQR range for a numeric column."""
    values = pd.to_numeric(data[column], errors="coerce")
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(values >= lower) & (values <= upper)].copy()


def _safe_log(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.log(values.where(values > 0, np.nan))


def _add_bathrooms(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    if {"Full Baths", "Half Baths"}.issubset(df.columns):
        full = pd.to_numeric(df["Full Baths"], errors="coerce").fillna(0)
        half = pd.to_numeric(df["Half Baths"], errors="coerce").fillna(0)
        df["Bathrooms"] = full + 0.5 * half
        return df

    if "Bathrooms" in df.columns:
        df["Bathrooms"] = pd.to_numeric(df["Bathrooms"], errors="coerce").fillna(0)
        return df

    if "Description" in df.columns:
        extracted = (
            df["Description"]
            .astype(str)
            .str.extract(r"(\d*\.?\d+)\s+of\s+which\s+are\s+bathrooms")
        )
        df["Bathrooms"] = pd.to_numeric(extracted[0], errors="coerce").fillna(0)
        return df

    df["Bathrooms"] = 0.0
    return df


def _fill_with_median(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = data.copy()
    for column in columns:
        median = df[column].median()
        if pd.isna(median):
            median = 0.0
        df[column] = df[column].fillna(median)
    return df


def engineer_features(
    data: pd.DataFrame,
    *,
    is_test_set: bool = False,
    remove_sale_price_outliers: bool = True,
) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
    """Create the design matrix used by the portfolio model.

    When ``is_test_set`` is false, returns ``(X, y)`` where ``y`` is log sale
    price. When ``is_test_set`` is true, returns only ``X`` and never filters
    rows, matching the real assessment use case where every parcel needs a
    prediction.
    """
    df = data.copy()

    if "Estimate (Building)" not in df.columns:
        raise KeyError("Expected column 'Estimate (Building)' in input data.")
    if "Building Square Feet" not in df.columns:
        raise KeyError("Expected column 'Building Square Feet' in input data.")

    df["Log Estimate (Building)"] = _safe_log(df["Estimate (Building)"])
    df["Log Building Square Feet"] = _safe_log(df["Building Square Feet"])
    df = _add_bathrooms(df)
    df = _fill_with_median(df, DEFAULT_FEATURES)

    if is_test_set:
        return df[DEFAULT_FEATURES].copy()

    if "Sale Price" not in df.columns:
        raise KeyError("Training data must include 'Sale Price'.")

    df["Log Sale Price"] = _safe_log(df["Sale Price"])
    df = df.dropna(subset=["Log Sale Price", *DEFAULT_FEATURES]).copy()

    if remove_sale_price_outliers:
        df = remove_outliers_iqr(df, "Log Sale Price")

    return df[DEFAULT_FEATURES].copy(), df["Log Sale Price"].copy()
