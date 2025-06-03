"""Train and evaluate a pooled XGBoost model using rolling windows."""

from __future__ import annotations

import itertools
from typing import Dict, List

import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from src.data_ingestion_preprocessing import load_data, preprocess_features

DATA_PATH = "data/OrangeJuiceX25.csv"


def load_processed() -> pd.DataFrame:
    """Load and preprocess the dataset."""
    df = load_data(DATA_PATH)
    df = preprocess_features(df)
    return df


def _tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, float]:
    """Small grid search to pick XGBoost hyperparameters."""
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
            n_estimators=300,
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


def run_pooled_xgb(df: pd.DataFrame) -> float:
    """Rolling-window pooled XGBoost across all series."""
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
        preds.extend(model.predict(test[feature_cols]))
        actuals.extend(test["y_next"])

    rmse = mean_squared_error(actuals, preds, squared=False)
    return rmse


def main() -> None:
    df = load_processed()
    rmse = run_pooled_xgb(df)
    print(f"Pooled XGBoost RMSE: {rmse:.3f}")


if __name__ == "__main__":
    main()
