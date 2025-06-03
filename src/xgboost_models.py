"""XGBoost with explicit GPU training (using gpu_hist and gpu_id) to leverage your RTX 4070.

We remove the invalid `device="cuda"` argument from DMatrix, and instead set `tree_method="gpu_hist"`
and `gpu_id=0` in the params dictionary. This ensures full GPU utilization without “mismatched devices”.
"""

from __future__ import annotations

import itertools
from typing import Dict, List
import warnings

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Parameters: \\{ \"predictor\" \\} are not used.*")

DATA_PATH = "data\OrangeJuiceX25.csv"


def load_processed() -> pd.DataFrame:
    """Load and preprocess the dataset using utilities from preprocessing."""
    from Processing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)
    return df


def _tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, float]:
    """Grid search over a broader hyperparameter space using GPU‐friendly params."""
    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.5, 0.7, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
    }

    # Create CPU‐resident DMatrix for train/val
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    best_params: Dict[str, float] | None = None
    best_rmse = float("inf")

    for md, lr, subs, col, mcw, gm in itertools.product(
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["subsample"],
        param_grid["colsample_bytree"],
        param_grid["min_child_weight"],
        param_grid["gamma"],
    ):
        params = {
            "tree_method":      "gpu_hist",
            "gpu_id":            0,
            "objective":        "reg:squarederror",
            "eval_metric":      "rmse",
            "max_depth":        md,
            "learning_rate":    lr,
            "subsample":        subs,
            "colsample_bytree": col,
            "min_child_weight": mcw,
            "gamma":            gm,
            "verbosity":        0,
        }

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dval, "validation")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        preds = bst.predict(dval)
        mse = mean_squared_error(y_val.to_numpy(), preds)
        rmse = np.sqrt(mse)

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {
                "max_depth":        md,
                "learning_rate":    lr,
                "subsample":        subs,
                "colsample_bytree": col,
                "min_child_weight": mcw,
                "gamma":            gm,
            }

    assert best_params is not None
    return best_params


def xgb_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """Rolling‐window XGBoost for one store–brand pair on log‐sales, using GPU training."""
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week")

    # Compute log‐sales target
    subset["y_next_log"] = np.log1p(subset["sales"].shift(-1))
    subset.dropna(subset=["y_next_log"], inplace=True)

    # Need at least 70 rows (60 warmup + 10 validation)
    if len(subset) <= 70:
        raise ValueError(
            f"Not enough data for store={store}, brand={brand} ({len(subset)} rows)."
        )

    lag_cols   = [f"sales_lag{lag}" for lag in [1, 2, 3]]
    price_cols = [c for c in subset.columns if c.startswith("price")]
    feature_cols = lag_cols + price_cols + ["deal", "feat", "sin_week", "cos_week"]

    preds: List[float] = []
    actuals: List[float] = []

    for end in range(70, len(subset)):
        window   = subset.iloc[:end]
        test_row = subset.iloc[end : end + 1]

        # Hold out last 10 rows of window for validation
        train_part = window.iloc[:-10]
        val_part   = window.iloc[-10:]

        X_train = train_part[feature_cols]
        y_train = train_part["y_next_log"]
        X_val   = val_part[feature_cols]
        y_val   = val_part["y_next_log"]

        # Tune hyperparameters using GPU training
        best_params = _tune_xgb(X_train, y_train, X_val, y_val)

        params_full = {
            "tree_method":      "gpu_hist",
            "gpu_id":            0,
            "objective":        "reg:squarederror",
            "eval_metric":      "rmse",
            "max_depth":        best_params["max_depth"],
            "learning_rate":    best_params["learning_rate"],
            "subsample":        best_params["subsample"],
            "colsample_bytree": best_params["colsample_bytree"],
            "min_child_weight": best_params["min_child_weight"],
            "gamma":            best_params["gamma"],
            "verbosity":        0,
        }

        # Train final model on entire window (log‐target), GPU‐enabled
        dfull = xgb.DMatrix(window[feature_cols], label=window["y_next_log"])
        bst = xgb.train(params_full, dfull, num_boost_round=300, verbose_eval=False)

        # Predict on log‐scale, then back‐transform
        dtest   = xgb.DMatrix(test_row[feature_cols])
        pred_log = bst.predict(dtest)[0]
        pred = np.expm1(pred_log)

        true = test_row["sales"].values[0]

        preds.append(pred)
        actuals.append(true)

    mse = mean_squared_error(actuals, preds)
    return np.sqrt(mse)


def xgb_pooled(df: pd.DataFrame) -> float:
    """Rolling‐window pooled XGBoost on all series with GPU training, log‐sales target."""
    df = df.sort_values(["store", "brand", "week"])
    df["y_next_log"] = df.groupby(["store", "brand"])["sales"].shift(-1).apply(np.log1p)
    df = df.dropna(subset=["y_next_log"])

    lag_cols   = [f"sales_lag{lag}" for lag in [1, 2, 3]]
    price_cols = [c for c in df.columns if c.startswith("price")]

    store_dummies = pd.get_dummies(df["store_id"], prefix="store")
    brand_dummies = pd.get_dummies(df["brand_id"], prefix="brand")
    df = pd.concat([df, store_dummies, brand_dummies], axis=1)
    dummy_cols = list(store_dummies.columns) + list(brand_dummies.columns)

    feature_cols = (
        lag_cols
        + price_cols
        + ["deal", "feat", "sin_week", "cos_week"]
        + dummy_cols
    )

    preds: List[float] = []
    actuals: List[float] = []

    weeks = sorted(df["week"].unique())
    if len(weeks) <= 70:
        raise ValueError(f"Not enough total weeks ({len(weeks)}) for pooled XGBoost.")

    for end_week in weeks[70:-1]:
        window = df[df["week"] <= end_week]
        test   = df[df["week"] == end_week + 1]
        if test.empty:
            continue

        # Within window, hold out last 10 weeks for validation
        train_df = window[window["week"] <= end_week - 10]
        val_df   = window[window["week"] > end_week - 10]
        if len(train_df) < 20 or val_df.empty:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["y_next_log"]
        X_val   = val_df[feature_cols]
        y_val   = val_df["y_next_log"]

        best_params = _tune_xgb(X_train, y_train, X_val, y_val)

        params_full = {
            "tree_method":      "gpu_hist",
            "gpu_id":            0,
            "objective":        "reg:squarederror",
            "eval_metric":      "rmse",
            "max_depth":        best_params["max_depth"],
            "learning_rate":    best_params["learning_rate"],
            "subsample":        best_params["subsample"],
            "colsample_bytree": best_params["colsample_bytree"],
            "min_child_weight": best_params["min_child_weight"],
            "gamma":            best_params["gamma"],
            "verbosity":        0,
        }

        # Train on train + val (log‐target) with GPU
        full_train = pd.concat([train_df, val_df], axis=0)
        dfull = xgb.DMatrix(full_train[feature_cols], label=full_train["y_next_log"])
        bst = xgb.train(params_full, dfull, num_boost_round=400, verbose_eval=False)

        # Predict on original scale
        dtest = xgb.DMatrix(test[feature_cols])
        preds_log = bst.predict(dtest)
        preds_orig = np.expm1(preds_log)

        preds.extend(preds_orig.tolist())
        actuals.extend(test["sales"].values.tolist())

    if not preds:
        raise ValueError("No predictions were made in the pooled run.")
    mse = mean_squared_error(actuals, preds)
    return np.sqrt(mse)


def main() -> None:
    df = load_processed()

    # Compute naïve persistence RMSE
    persistence_err = 0.0
    total_count = 0
    for (_, _), group in df.groupby(["store", "brand"]):
        grp = group.sort_values("week")
        grp["y_pred_naive"] = grp["sales"].shift(1)
        valid = grp.dropna(subset=["y_pred_naive"])
        persistence_err += ((valid["sales"] - valid["y_pred_naive"]) ** 2).sum()
        total_count += len(valid)
    rmse_naive = np.sqrt(persistence_err / total_count)
    print(f"Naïve persistence RMSE: {rmse_naive:.2f}")

    # Pick a sample (store,brand) with ≥ 80 rows
    counts = (
        df.groupby(["store", "brand"])
          .size()
          .reset_index(name="n_rows")
          .sort_values("n_rows", ascending=False)
    )
    sample = counts[counts["n_rows"] > 80].iloc[0]
    s, b = int(sample["store"]), int(sample["brand"])
    print(f"Using (store, brand) = ({s}, {b}) for per‐series run.")

    # Per‐series
    rmse_single = xgb_per_series(df, s, b)
    print(f"XGBoost per‐series RMSE: {rmse_single:.2f}")

    # Pooled
    rmse_pool = xgb_pooled(df)
    print(f"Pooled XGBoost RMSE:   {rmse_pool:.2f}")

    # Relative errors
    mean_sales = df["sales"].mean()
    median_sales = df["sales"].median()
    print(f"Mean weekly sales:   {mean_sales:.2f}")
    print(f"Median weekly sales: {median_sales:.2f}")
    print(f"Per‐series RMSE / mean_sales:   {rmse_single/mean_sales:.2%}")
    print(f"Per‐series RMSE / median_sales: {rmse_single/median_sales:.2%}")
    print(f"Pooled RMSE / mean_sales:       {rmse_pool/mean_sales:.2%}")
    print(f"Pooled RMSE / median_sales:     {rmse_pool/median_sales:.2%}")


if __name__ == "__main__":
    main()
