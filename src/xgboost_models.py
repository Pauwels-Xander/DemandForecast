"""XGBoost prototypes for demand forecasting.

Includes per-series and pooled implementations with simple hyperparameter tuning.
"""

from __future__ import annotations

import itertools
from typing import Dict, List

import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

DATA_PATH = "data/OrangeJuiceX25.csv"


def load_processed() -> pd.DataFrame:
    """Load and preprocess the dataset using utilities from preprocessing."""
    from src.data_ingestion_preprocessing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)
    return df


def _tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, float]:
    """Grid search over a small hyperparameter space."""
    param_grid = {
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }
    best_params: Dict[str, float] | None = None
    best_rmse = float("inf")

    for md, lr, subs, col in itertools.product(
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["subsample"],
        param_grid["colsample_bytree"],
    ):
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=md,
            learning_rate=lr,
            subsample=subs,
            colsample_bytree=col,
            random_state=0,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False,
        )
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {
                "max_depth": md,
                "learning_rate": lr,
                "subsample": subs,
                "colsample_bytree": col,
            }
    assert best_params is not None
    return best_params


def xgb_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """Rolling-window XGBoost for a single store-brand pair."""
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week")
    subset["y_next"] = subset["sales"].shift(-1)
    subset.dropna(inplace=True)

    lag_cols = [f"sales_lag{lag}" for lag in [1, 2, 3]]
    price_cols = [c for c in subset.columns if c.startswith("price")]
    feature_cols = lag_cols + price_cols + ["deal", "feat", "sin_week", "cos_week"]

    preds: List[float] = []
    actuals: List[float] = []

    for end in range(60, len(subset)):
        train = subset.iloc[:end]
        test = subset.iloc[end : end + 1]

        X_train = train.iloc[:-10][feature_cols]
        y_train = train.iloc[:-10]["y_next"]
        X_val = train.iloc[-10:][feature_cols]
        y_val = train.iloc[-10:]["y_next"]

        params = _tune_xgb(X_train, y_train, X_val, y_val)
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            random_state=0,
            **params,
        )
        model.fit(
            train[feature_cols],
            train["y_next"],
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False,
        )
        pred = model.predict(test[feature_cols])[0]
        preds.append(pred)
        actuals.append(test["y_next"].values[0])

    rmse = mean_squared_error(actuals, preds, squared=False)
    return rmse


def xgb_pooled(df: pd.DataFrame) -> float:
    """Rolling-window XGBoost pooling all store-brand series."""
    df = df.sort_values(["store", "brand", "week"])
    df["y_next"] = df.groupby(["store", "brand"])["sales"].shift(-1)
    df = df.dropna(subset=["y_next"])

    lag_cols = [f"sales_lag{lag}" for lag in [1, 2, 3]]
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
    for end_week in weeks[59:-1]:
        window = df[df["week"] <= end_week]
        test = df[df["week"] == end_week + 1]

        train = window[window["week"] <= end_week - 10]
        val = window[window["week"] > end_week - 10]

        X_train = train[feature_cols]
        y_train = train["y_next"]
        X_val = val[feature_cols]
        y_val = val["y_next"]

        params = _tune_xgb(X_train, y_train, X_val, y_val)
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            random_state=0,
            **params,
        )
        model.fit(
            pd.concat([X_train, X_val]),
            pd.concat([y_train, y_val]),
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False,
        )
        pred = model.predict(test[feature_cols])
        preds.extend(pred)
        actuals.extend(test["y_next"])

    rmse = mean_squared_error(actuals, preds, squared=False)
    return rmse


def main() -> None:
    df = load_processed()
    sample_store, sample_brand = 1, 1

    print("XGBoost per-series RMSE:")
    rmse_single = xgb_per_series(df, sample_store, sample_brand)
    print(rmse_single)

    print("\nPooled XGBoost RMSE:")
    rmse_pool = xgb_pooled(df)
    print(rmse_pool)


if __name__ == "__main__":
    main()
