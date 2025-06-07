from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, Lasso
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from tqdm import tqdm  # For progress bar

DATA_PATH = "OrangeJuiceX25.csv"


def load_processed() -> pd.DataFrame:
    """Load raw data and apply preprocessing."""
    from Processing import load_data, preprocess_features

    df_raw = load_data(DATA_PATH)
    print(f"After load_data: total rows = {len(df_raw)}")

    df_processed = preprocess_features(df_raw)
    print(f"After preprocess_features: total rows = {len(df_processed)}")

    return df_processed


def ridge_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """
    Rolling-window RidgeCV on one store–brand series with a fixed window of 52 weeks.
    Returns RMSE of one-step-ahead forecasts using Ridge with built-in CV.
    """
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week")

    # Create next-week target, drop rows without lags/target
    subset["y_next"] = subset["sales"].shift(-1)
    subset.dropna(subset=["y_next"], inplace=True)
    subset.dropna(subset=[f"sales_lag{l}" for l in (1, 2, 3)], inplace=True)

    # Now we need at least 52 for the first training window + 1 for test → 53 rows total
    if len(subset) < 53:
        raise ValueError(
            f"Not enough data for Ridge (store={store}, brand={brand}): "
            f"found {len(subset)} rows, need ≥53."
        )

    price_cols = [c for c in subset.columns if c.startswith("price")]
    lag_cols = [f"sales_lag{l}" for l in (1, 2, 3)]
    feature_cols = (
        price_cols
        + lag_cols
        + ["deal", "feat", "store_id", "brand_id", "sin_week", "cos_week"]
    )

    preds, actuals = [], []

    # For each end index from 52 up to len(subset)-1, train on exactly 52 prior rows
    for end in range(52, len(subset)):
        train_window = subset.iloc[end - 52 : end]
        test_row = subset.iloc[end : end + 1]

        X_train = train_window[feature_cols].astype(float)
        y_train = train_window["y_next"].astype(float)
        X_test = test_row[feature_cols].astype(float)
        y_test = test_row["y_next"].astype(float)

        # RidgeCV over a small grid of alphas
        ridge_alphas = np.logspace(-3, 3, 10)  # candidates between 1e-3 and 1e3
        model_ridge = RidgeCV(
            alphas=ridge_alphas, cv=5, scoring="neg_mean_squared_error"
        ).fit(X_train, y_train)
        pred = model_ridge.predict(X_test)[0]

        preds.append(pred)
        actuals.append(y_test.values[0])

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    return rmse


def lasso_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """
    Rolling-window LASSO on one store–brand series with a fixed window of 52 weeks.
    Returns RMSE of one-step-ahead forecasts.
    """
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week")

    subset["y_next"] = subset["sales"].shift(-1)
    subset.dropna(subset=["y_next"], inplace=True)
    subset.dropna(subset=[f"sales_lag{l}" for l in (1, 2, 3)], inplace=True)

    # Now we need at least 52 for the first training window + 1 for test → 53 rows total
    if len(subset) < 53:
        raise ValueError(
            f"Not enough data for LASSO (store={store}, brand={brand}): "
            f"found {len(subset)} rows, need ≥53."
        )

    price_cols = [c for c in subset.columns if c.startswith("price")]
    lag_cols = [f"sales_lag{l}" for l in (1, 2, 3)]
    feature_cols = (
        price_cols
        + lag_cols
        + ["deal", "feat", "store_id", "brand_id", "sin_week", "cos_week"]
    )

    preds, actuals = [], []

    for end in range(52, len(subset)):
        train_window = subset.iloc[end - 52 : end]
        test_row = subset.iloc[end : end + 1]

        X_train = train_window[feature_cols].astype(float)
        y_train = train_window["y_next"].astype(float)
        X_test = test_row[feature_cols].astype(float)
        y_test = test_row["y_next"].astype(float)

        model = LassoCV(
            cv=5,
            random_state=0,
            max_iter=5000,
            tol=1e-5,
        ).fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        preds.append(pred)
        actuals.append(y_test.values[0])

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    # If you ever want to inspect which alpha LassoCV picked, uncomment:
    # print(f"Store={store}, Brand={brand}, chosen LASSO alpha = {model.alpha_:.4f}")
    return rmse


def gets_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """
    Rolling-window GETS (General-to-Specific) with OLS + t-ratio pruning,
    using a fixed window of 52 weeks.
    Returns RMSE of one-step-ahead forecasts.
    """
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week")

    subset["y_next"] = subset["sales"].shift(-1)
    subset.dropna(subset=["y_next"], inplace=True)
    subset.dropna(subset=[f"sales_lag{l}" for l in (1, 2, 3)], inplace=True)

    # Now we need at least 52 for the first training window + 1 for test → 53 rows total
    if len(subset) < 53:
        raise ValueError(
            f"Not enough data for GETS (store={store}, brand={brand}): "
            f"found {len(subset)} rows, need ≥53."
        )

    price_cols = [c for c in subset.columns if c.startswith("price")]
    lag_cols = [f"sales_lag{l}" for l in (1, 2, 3)]
    feature_cols = (
        price_cols
        + lag_cols
        + ["deal", "feat", "store_id", "brand_id", "sin_week", "cos_week"]
    )

    preds, actuals = [], []

    for end in range(52, len(subset)):
        train_window = subset.iloc[end - 52 : end]
        test_row = subset.iloc[end : end + 1]

        X = train_window[feature_cols].astype(float)
        y = train_window["y_next"].astype(float)
        X_test = test_row[feature_cols].astype(float)
        y_test = test_row["y_next"].astype(float)

        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        # Prune variables until all remaining |t| ≥ 1.96 or no variables left besides const
        while True:
            tvals = model.tvalues.drop(labels="const", errors="ignore").dropna()
            if tvals.empty:
                break

            min_var = tvals.abs().idxmin()
            if abs(tvals[min_var]) < 1.96:
                X_const = X_const.drop(columns=[min_var])
                X_test = X_test.drop(columns=[min_var])
                model = sm.OLS(y, X_const).fit()
            else:
                break

        pred = model.predict(sm.add_constant(X_test, has_constant="add")).iloc[0]
        preds.append(pred)
        actuals.append(y_test.values[0])

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    return rmse


if __name__ == "__main__":
    # Load processed data
    df_processed = load_processed()

    # Find all unique (store, brand) pairs
    combos = df_processed[["store", "brand"]].drop_duplicates().values
    total_combos = len(combos)
    print(f"Found {total_combos} unique store–brand combinations.\n")

    ridge_rmses = []
    lasso_rmses = []
    gets_rmses = []

    # Loop over each store–brand pair
    for store, brand in tqdm(combos, desc="Processing store-brand combos"):
        # 1) Compute RidgeCV RMSE (rolling window of 52)
        try:
            rmse_r = ridge_per_series(df_processed, store, brand)
            ridge_rmses.append(rmse_r)
        except ValueError:
            # Not enough data → skip
            pass

        # 2) Compute LASSO RMSE (rolling window of 52)
        try:
            rmse_l = lasso_per_series(df_processed, store, brand)
            lasso_rmses.append(rmse_l)
        except ValueError:
            # Not enough data → skip
            pass

        # 3) Compute GETS RMSE (rolling window of 52)
        try:
            rmse_g = gets_per_series(df_processed, store, brand)
            gets_rmses.append(rmse_g)
        except ValueError:
            # Not enough data → skip
            pass

    # Compute average RMSE for each method (if any combos succeeded)
    avg_ridge = np.mean(ridge_rmses) if ridge_rmses else float("nan")
    avg_lasso = np.mean(lasso_rmses) if lasso_rmses else float("nan")
    avg_gets = np.mean(gets_rmses) if gets_rmses else float("nan")

    print(f"\nAverage RMSE across all store–brand combos (Ridge) : {avg_ridge:.4f}")
    print(f"Average RMSE across all store–brand combos (LASSO): {avg_lasso:.4f}")
    print(f"Average RMSE across all store–brand combos (GETS) : {avg_gets:.4f}")
