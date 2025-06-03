from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

DATA_PATH = "data\OrangeJuiceX25.csv"


def load_processed() -> pd.DataFrame:
    """Load raw data and apply preprocessing."""
    from Processing import load_data, preprocess_features

    df_raw = load_data(DATA_PATH)
    print(f"After load_data: total rows = {len(df_raw)}")

    df_processed = preprocess_features(df_raw)
    print(f"After preprocess_features: total rows = {len(df_processed)}")

    return df_processed


def lasso_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """
    Rolling-window LASSO on one store–brand series.
    Returns RMSE of one-step-ahead forecasts.
    """
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week")

    subset["y_next"] = subset["sales"].shift(-1)
    subset.dropna(subset=["y_next"], inplace=True)
    subset.dropna(subset=[f"sales_lag{l}" for l in (1, 2, 3)], inplace=True)

    if len(subset) <= 60:
        raise ValueError(
            f"Not enough data for store={store}, brand={brand}: "
            f"found {len(subset)} rows, need >60."
        )

    price_cols = [c for c in subset.columns if c.startswith("price")]
    lag_cols = [f"sales_lag{l}" for l in (1, 2, 3)]
    feature_cols = (
        price_cols
        + lag_cols
        + ["deal", "feat", "store_id", "brand_id", "sin_week", "cos_week"]
    )

    if debug and debug_end is None:
        debug_end = np.random.randint(60, len(subset))

    preds, actuals = [], []
    for end in range(60, len(subset)):
        train = subset.iloc[:end]
        test = subset.iloc[end : end + 1]

        X_train = train[feature_cols].astype(float)
        y_train = train["y_next"].astype(float)
        X_test = test[feature_cols].astype(float)
        y_test = test["y_next"].astype(float)

        model = LassoCV(cv=5, random_state=0, max_iter=5000, tol=1e-5).fit(
            X_train, y_train
        )
        pred = model.predict(X_test)[0]

        preds.append(pred)
        actuals.append(y_test.values[0])

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    return rmse


def gets_per_series(
    df: pd.DataFrame,
    store: int,
    brand: int,
    *,
    debug: bool = False,
    debug_end: int | None = None,
    return_residuals: bool = False,
) -> float | tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling-window GETS (General-to-Specific).

    Parameters
    ----------
    debug:
        If ``True`` print the first/last training rows and the test row for a
        randomly chosen window.  ``debug_end`` can specify the window index.
    return_residuals:
        If ``True`` also return arrays of residuals, predictions and actuals.
    """
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week")

    subset["y_next"] = subset["sales"].shift(-1)
    subset.dropna(subset=["y_next"], inplace=True)
    subset.dropna(subset=[f"sales_lag{l}" for l in (1, 2, 3)], inplace=True)

    if len(subset) <= 60:
        raise ValueError(
            f"Not enough data for GETS (store={store}, brand={brand}): "
            f"found {len(subset)} rows, need >60."
        )

    price_cols = [c for c in subset.columns if c.startswith("price")]
    lag_cols = [f"sales_lag{l}" for l in (1, 2, 3)]
    feature_cols = (
        price_cols
        + lag_cols
        + ["deal", "feat", "store_id", "brand_id", "sin_week", "cos_week"]
    )

    preds, actuals = [], []
    for end in range(60, len(subset)):
        train = subset.iloc[:end]
        test = subset.iloc[end : end + 1]

        X = train[feature_cols].astype(float)
        y = train["y_next"].astype(float)
        X_test = test[feature_cols].astype(float)
        y_test = test["y_next"].astype(float)

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

        if debug and end == debug_end:
            print("Debug window index:", end)
            print("Train first row:\n", train.iloc[0])
            print("Train last row:\n", train.iloc[-1])
            print("Test row:\n", test.iloc[0])

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    if return_residuals:
        residuals = np.asarray(actuals) - np.asarray(preds)
        return rmse, residuals, np.asarray(preds), np.asarray(actuals)
    return rmse


def pooled_lasso(df: pd.DataFrame) -> float:
    """
    Pooled LASSO across all store-brand combinations.
    Returns RMSE of one-step-ahead forecasts.
    """
    df = df.sort_values(["store", "brand", "week"])
    df["y_next"] = df.groupby(["store", "brand"])["sales"].shift(-1)
    df.dropna(subset=["y_next"], inplace=True)
    df.dropna(subset=[f"sales_lag{l}" for l in (1, 2, 3)], inplace=True)

    price_cols = [c for c in df.columns if c.startswith("price")]
    lag_cols = [f"sales_lag{l}" for l in (1, 2, 3)]
    feature_cols = (
        price_cols
        + lag_cols
        + ["deal", "feat", "store_id", "brand_id", "sin_week", "cos_week"]
    )

    df = df.dropna(subset=feature_cols)

    weeks = sorted(df["week"].unique())
    if len(weeks) <= 60:
        raise ValueError(
            f"Not enough distinct weeks in the pooled data (found {len(weeks)}, need >60)."
        )

    preds, actuals = [], []
    for end_week in weeks[59:-1]:
        train = df[df["week"] <= end_week].copy()
        test = df[df["week"] == end_week + 1].copy()

        X_train = train[feature_cols].astype(float)
        y_train = train["y_next"].astype(float)
        X_test = test[feature_cols].astype(float)
        y_test = test["y_next"].astype(float)

        model = LassoCV(cv=5, random_state=0, max_iter=5000, tol=1e-5).fit(
            X_train, y_train
        )
        pred = model.predict(X_test)

        preds.extend(pred)
        actuals.extend(y_test)

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    return rmse


if __name__ == "__main__":
    df_processed = load_processed()

    sample_store, sample_brand = 21, 1
    subset = df_processed[
        (df_processed["store"] == sample_store)
        & (df_processed["brand"] == sample_brand)
    ]
    print(f"Rows for store={sample_store}, brand={sample_brand}: {len(subset)}")

    print("\nLASSO per series RMSE:")
    try:
        print(lasso_per_series(df_processed, sample_store, sample_brand))
    except ValueError as e:
        print("LASSO error:", e)

    print("\nGETS per series RMSE:")
    try:
        print(gets_per_series(df_processed, sample_store, sample_brand))
    except ValueError as e:
        print("GETS error:", e)

    print("\nPooled LASSO RMSE:")
    try:
        print(pooled_lasso(df_processed))
    except ValueError as e:
        print("Pooled LASSO error:", e)
