"""Train and evaluate the Cook County assessment fairness model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
except ModuleNotFoundError:
    LinearRegression = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from property_assessment_fairness import (  # noqa: E402
    assessment_bias_by_price_tier,
    engineer_features,
    rmse,
)


class NumpyLinearRegression:
    """Small least-squares fallback used when scikit-learn is unavailable."""

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "NumpyLinearRegression":
        design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
        self.weights_, *_ = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
        return design @ self.weights_


def make_model():
    if LinearRegression is not None:
        return LinearRegression()
    return NumpyLinearRegression()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a property value model and report fairness diagnostics."
    )
    parser.add_argument("--train", required=True, help="Path to training CSV with Sale Price.")
    parser.add_argument("--test", help="Optional path to test CSV without Sale Price.")
    parser.add_argument("--output", help="Optional CSV path for test predictions.")
    parser.add_argument("--holdout-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--folds", type=int, default=4)
    return parser.parse_args()


def train_test_split_frame(
    data: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))
    valid_count = max(1, int(round(len(data) * test_size)))
    valid_idx = indices[:valid_count]
    train_idx = indices[valid_count:]
    return data.iloc[train_idx].copy(), data.iloc[valid_idx].copy()


def kfold_indices(n_rows: int, folds: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_rows)
    fold_sizes = np.full(folds, n_rows // folds, dtype=int)
    fold_sizes[: n_rows % folds] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        valid_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        current = stop
        yield train_idx, valid_idx


def cross_validated_rmse(x: pd.DataFrame, y: pd.Series, folds: int, seed: int) -> list[float]:
    errors: list[float] = []
    for train_idx, valid_idx in kfold_indices(len(x), folds, seed):
        x_train = x.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        model = make_model().fit(x_train, y_train)
        errors.append(rmse(y_valid, model.predict(x_valid)))
    return errors


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    train_df = pd.read_csv(train_path)

    train_part, valid_part = train_test_split_frame(
        train_df, test_size=args.holdout_size, seed=args.seed
    )

    x_train, y_train = engineer_features(train_part)
    x_valid, y_valid = engineer_features(valid_part)

    model = make_model().fit(x_train, y_train)
    train_predictions = model.predict(x_train)
    valid_predictions = model.predict(x_valid)

    print("Model evaluation")
    print(f"Train RMSE (log price):   {rmse(y_train, train_predictions):.4f}")
    print(f"Holdout RMSE (log price): {rmse(y_valid, valid_predictions):.4f}")

    cv_errors = cross_validated_rmse(x_train, y_train, args.folds, args.seed)
    cv_mean = sum(cv_errors) / len(cv_errors)
    print(f"{args.folds}-fold CV RMSE:        {cv_mean:.4f}")
    print("Fold RMSE values:         " + ", ".join(f"{err:.4f}" for err in cv_errors))

    print()
    print("Fairness diagnostics by price tier")
    print(assessment_bias_by_price_tier(y_valid, valid_predictions).to_string(index=False))

    if args.test:
        if not args.output:
            raise SystemExit("--output is required when --test is provided.")

        full_x, full_y = engineer_features(train_df)
        final_model = make_model().fit(full_x, full_y)
        test_df = pd.read_csv(args.test)
        test_x = engineer_features(test_df, is_test_set=True)
        predictions = final_model.predict(test_x)

        id_column = "Unnamed: 0" if "Unnamed: 0" in test_df.columns else None
        ids = test_df[id_column] if id_column else test_df.index
        output = pd.DataFrame({"Id": ids, "Value": predictions})
        output.to_csv(args.output, index=False)
        print()
        print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
