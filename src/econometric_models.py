# Econometric model prototypes: LASSO, GETS, and pooled LASSO
# These are simple reference implementations using pandas, scikit-learn,
# and statsmodels. They assume features have been preprocessed via
# src/data_ingestion_preprocessing.py.

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

DATA_PATH = "data/OrangeJuiceX25.csv"


# ---------- Helpers ----------

def load_processed() -> pd.DataFrame:
    """Load and preprocess the dataset using existing utilities."""
    from src.data_ingestion_preprocessing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)
    return df


# ---------- 3a. LASSO per store-brand ----------

def lasso_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """Run rolling-window LASSO for a single store-brand pair.

    Returns RMSE of one-step-ahead forecasts.
    """
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week")

    # Target is next week's sales
    subset["y_next"] = subset["sales"].shift(-1)
    subset.dropna(inplace=True)

    lag_cols = [f"sales_lag{lag}" for lag in [1, 2, 3]]
    price_cols = [c for c in subset.columns if c.startswith("price")]
    week_dummies = pd.get_dummies(subset["week"] % 52, prefix="week")
    subset = pd.concat([subset, week_dummies], axis=1)
    feature_cols = lag_cols + price_cols + ["deal", "feat"] + list(week_dummies.columns)

    preds, actuals = [], []
    for end in range(60, len(subset)):
        train = subset.iloc[:end]
        test = subset.iloc[end : end + 1]

        X_train = train[feature_cols]
        y_train = train["y_next"]
        X_test = test[feature_cols]
        y_test = test["y_next"]

        model = LassoCV(cv=5, random_state=0).fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        preds.append(pred)
        actuals.append(y_test.values[0])

    rmse = mean_squared_error(actuals, preds, squared=False)
    return rmse


# ---------- 3b. GETS via OLS with t-ratio pruning ----------

def gets_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """Apply General-to-Specific regression with t-ratio pruning."""
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week")
    subset["y_next"] = subset["sales"].shift(-1)
    subset.dropna(inplace=True)

    lag_cols = [f"sales_lag{lag}" for lag in [1, 2, 3]]
    price_cols = [c for c in subset.columns if c.startswith("price")]
    week_dummies = pd.get_dummies(subset["week"] % 52, prefix="week")
    subset = pd.concat([subset, week_dummies], axis=1)
    feature_cols = lag_cols + price_cols + ["deal", "feat"] + list(week_dummies.columns)

    preds, actuals = [], []
    for end in range(60, len(subset)):
        train = subset.iloc[:end]
        test = subset.iloc[end : end + 1]

        X = train[feature_cols]
        y = train["y_next"]
        X_test = test[feature_cols]
        y_test = test["y_next"]

        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        # Iteratively drop lowest |t| variable until all |t| >= 1.96
        while True:
            tvals = model.tvalues.drop("const")
            min_var = tvals.abs().idxmin()
            if abs(tvals[min_var]) < 1.96:
                X_const = X_const.drop(columns=[min_var])
                X_test = X_test.drop(columns=[min_var])
                model = sm.OLS(y, X_const).fit()
            else:
                break

        pred = model.predict(sm.add_constant(X_test, has_constant="add"))[0]
        preds.append(pred)
        actuals.append(y_test.values[0])

    rmse = mean_squared_error(actuals, preds, squared=False)
    return rmse


# ---------- 3c. Panel-wide LASSO ----------

def pooled_lasso(df: pd.DataFrame) -> float:
    """Pooled LASSO across all store-brand combinations."""
    df = df.sort_values(["store", "brand", "week"])
    df["y_next"] = df.groupby(["store", "brand"])["sales"].shift(-1)
    df = df.dropna(subset=["y_next"])  # drop last week per series

    lag_cols = [f"sales_lag{lag}" for lag in [1, 2, 3]]
    price_cols = [c for c in df.columns if c.startswith("price")]
    week_dummies = pd.get_dummies(df["week"] % 52, prefix="week")
    df = pd.concat([df, week_dummies], axis=1)
    feature_cols = (
        lag_cols
        + price_cols
        + ["deal", "feat", "store_id", "brand_id"]
        + list(week_dummies.columns)
    )

    preds, actuals = [], []
    weeks = sorted(df["week"].unique())
    for end in weeks[59:-1]:  # ensure at least 60 weeks in window
        train = df[df["week"] <= end]
        test = df[df["week"] == end + 1]

        X_train = train[feature_cols]
        y_train = train["y_next"]
        X_test = test[feature_cols]
        y_test = test["y_next"]

        model = LassoCV(cv=5, random_state=0).fit(X_train, y_train)
        pred = model.predict(X_test)
        preds.extend(pred)
        actuals.extend(y_test)

    rmse = mean_squared_error(actuals, preds, squared=False)
    return rmse


if __name__ == "__main__":
    df_processed = load_processed()
    # Example store-brand pair
    sample_store, sample_brand = 1, 1

    print("LASSO per series RMSE:")
    rmse_lasso = lasso_per_series(df_processed, sample_store, sample_brand)
    print(rmse_lasso)

    print("\nGETS per series RMSE:")
    rmse_gets = gets_per_series(df_processed, sample_store, sample_brand)
    print(rmse_gets)

    print("\nPooled LASSO RMSE:")
    rmse_pooled = pooled_lasso(df_processed)
    print(rmse_pooled)
