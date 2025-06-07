from __future__ import annotations

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

DATA_PATH = "OrangeJuiceX25.csv"
WINDOW_SIZE = 52  # number of weeks to train on each iteration

# ------------------------------------------------------------------------------
# Example fixed hyperparameters (tuned once offline). You can change these.
FIXED_PARAMS = {
    "tree_method":      "gpu_hist",
    "gpu_id":            0,
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "gamma":            0,
    "verbosity":        0,
}
# ------------------------------------------------------------------------------

# Choose one of:
#   "per_series_all" → run per‐series XGBoost on every (store, brand) and average RMSE
#   "pooled"         → run pooled XGBoost on all store×brand rows by calendar week
#   "both"           → do both, printing both results
EVAL_MODE = "per_series_all"  # options: "per_series_all", "pooled", "both"


def load_processed() -> pd.DataFrame:
    """Load and preprocess the dataset using utilities from preprocessing."""
    from Processing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)
    return df


def xgb_per_series_52(
    df: pd.DataFrame, store: int, brand: int, fixed_params: dict
) -> float:
    """
    Rolling‐window per‐series XGBoost using exactly 52 weeks of history (no validation split).
    Trains on weeks [t-52 ... t-1] to predict week t, for a single store-brand pair.
    Returns RMSE over all such predictions for that series.
    """
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week").reset_index(drop=True)

    # Compute log‐sales target for next week
    subset["y_next_log"] = np.log1p(subset["sales"].shift(-1))
    subset = subset.dropna(subset=["y_next_log"]).reset_index(drop=True)

    # Need at least WINDOW_SIZE + 1 rows to make the first prediction
    if len(subset) < WINDOW_SIZE + 1:
        raise ValueError(
            f"Not enough data for store={store}, brand={brand} "
            f"({len(subset)} rows; need ≥ {WINDOW_SIZE+1})"
        )

    # Define feature columns
    lag_cols   = [f"sales_lag{lag}" for lag in (1, 2, 3)]
    price_cols = [c for c in subset.columns if c.startswith("price")]
    base_feats = lag_cols + price_cols + ["deal", "feat", "sin_week", "cos_week"]

    preds: list[float] = []
    actuals: list[float] = []

    for end in range(WINDOW_SIZE, len(subset)):
        # Train window covers indices [end-52 ... end-1], length = 52
        window_start = end - WINDOW_SIZE
        window_df = subset.iloc[window_start:end]

        # Train on these 52 weeks
        X_train = window_df[base_feats]
        y_train = window_df["y_next_log"]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        bst = xgb.train(fixed_params, dtrain, num_boost_round=300, verbose_eval=False)

        # Predict next week (row at index = end)
        test_row = subset.iloc[end : end + 1]
        dtest = xgb.DMatrix(test_row[base_feats])
        pred_log = bst.predict(dtest)[0]
        pred = np.expm1(pred_log)

        true_val = test_row["sales"].values[0]
        preds.append(pred)
        actuals.append(true_val)

    mse = mean_squared_error(actuals, preds)
    return float(np.sqrt(mse))


def xgb_pooled_52(df: pd.DataFrame, fixed_params: dict) -> float:
    """
    Rolling‐window pooled XGBoost using exactly 52 weeks of history (no validation split).
    Trains on calendar weeks [t-52 ... t-1] (across all store×brand rows) to predict week t.
    Returns RMSE over all such predictions.
    """
    # Sort and compute next‐week target
    df = df.sort_values(["store", "brand", "week"]).reset_index(drop=True)
    df["y_next_log"] = df.groupby(["store", "brand"])["sales"].shift(-1).apply(np.log1p)
    df = df.dropna(subset=["y_next_log"]).reset_index(drop=True)

    # Build feature columns: lags, prices, deal/feat/week‐cycles, and dummies
    lag_cols   = [f"sales_lag{lag}" for lag in (1, 2, 3)]
    price_cols = [c for c in df.columns if c.startswith("price")]
    base_feats = lag_cols + price_cols + ["deal", "feat", "sin_week", "cos_week"]

    # One‐hot encode store_id and brand_id once
    store_dummies = pd.get_dummies(df["store_id"], prefix="store")
    brand_dummies = pd.get_dummies(df["brand_id"], prefix="brand")
    df = pd.concat([df, store_dummies, brand_dummies], axis=1)
    dummy_cols = list(store_dummies.columns) + list(brand_dummies.columns)

    feature_cols = base_feats + dummy_cols

    # All unique calendar weeks (sorted)
    weeks = sorted(df["week"].unique())
    if len(weeks) < WINDOW_SIZE + 1:
        raise ValueError(f"Need ≥ {WINDOW_SIZE+1} distinct weeks; found {len(weeks)}.")

    preds: list[float] = []
    actuals: list[float] = []

    # We start at idx = WINDOW_SIZE, so:
    #   end_week  = weeks[idx-1]  → last week of training block
    #   next_week = weeks[idx]    → week we predict
    for idx in range(WINDOW_SIZE, len(weeks)):
        end_week  = weeks[idx - 1]
        next_week = weeks[idx]

        # Define the 52‐week block: [end_week-51 ... end_week]
        block_start = end_week - (WINDOW_SIZE - 1)
        block_mask  = (df["week"] >= block_start) & (df["week"] <= end_week)
        block_df    = df[block_mask]

        # Verify we have exactly 52 distinct weeks in this block
        distinct_block_weeks = sorted(block_df["week"].unique())
        if len(distinct_block_weeks) != WINDOW_SIZE:
            # If any week in that range has no rows, skip
            continue

        # Train on these 52 calendar weeks (all store×brand rows)
        dtrain = xgb.DMatrix(block_df[feature_cols], label=block_df["y_next_log"])
        bst = xgb.train(fixed_params, dtrain, num_boost_round=400, verbose_eval=False)

        # Predict for next_week across all store×brand rows in that week
        test_mask = df["week"] == next_week
        test_df   = df[test_mask]
        if test_df.empty:
            # No data in that calendar week, skip
            continue

        dtest = xgb.DMatrix(test_df[feature_cols])
        preds_log = bst.predict(dtest)
        preds_orig = np.expm1(preds_log)

        preds.extend(preds_orig.tolist())
        actuals.extend(test_df["sales"].tolist())

    if not preds:
        raise ValueError("No predictions generated; maybe insufficient weekly coverage.")

    mse = mean_squared_error(actuals, preds)
    return float(np.sqrt(mse))


def main() -> None:
    df = load_processed()

    # 1) Compute naïve persistence RMSE for baseline
    persistence_err = 0.0
    total_count = 0
    for (_, _), group in df.groupby(["store", "brand"]):
        grp = group.sort_values("week").reset_index(drop=True)
        grp["y_pred_naive"] = grp["sales"].shift(1)
        valid = grp.dropna(subset=["y_pred_naive"])
        persistence_err += ((valid["sales"] - valid["y_pred_naive"]) ** 2).sum()
        total_count += len(valid)
    rmse_naive = np.sqrt(persistence_err / total_count)
    print(f"Naïve persistence RMSE: {rmse_naive:.2f}\n")

    # 2) Get all unique (store, brand) pairs
    counts = (
        df.groupby(["store", "brand"])
          .size()
          .reset_index(name="n_rows")
          .sort_values("n_rows", ascending=False)
    )
    combos = counts[["store", "brand"]].values

    if EVAL_MODE in ("per_series_all", "both"):
        per_series_rmses: list[float] = []

        # Loop over every store–brand pair
        for store, brand in tqdm(combos, desc="Per‐series XGBoost over all store-brand pairs"):
            try:
                rmse_sb = xgb_per_series_52(df, int(store), int(brand), FIXED_PARAMS)
                per_series_rmses.append(rmse_sb)
            except ValueError:
                # Not enough data → skip
                continue

        if per_series_rmses:
            avg_per_series = float(np.mean(per_series_rmses))
            print(f"Average per-series XGBoost (52-week) RMSE across all pairs: {avg_per_series:.2f}")
        else:
            print("No valid store-brand series had enough data for per-series evaluation.")

    if EVAL_MODE in ("pooled", "both"):
        try:
            rmse_pooled = xgb_pooled_52(df, FIXED_PARAMS)
            print(f"Pooled XGBoost (52-week) RMSE:                 {rmse_pooled:.2f}")
        except ValueError as e:
            print(f"Pooled evaluation skipped: {e}")

    # 3) Show relative errors only if we ran per-series or pooled
    if EVAL_MODE in ("per_series_all", "both") and per_series_rmses:
        mean_sales = df["sales"].mean()
        median_sales = df["sales"].median()
        print(f"\nPer‐series RMSE / mean_sales:   {avg_per_series/mean_sales:.2%}")
        print(f"Per‐series RMSE / median_sales: {avg_per_series/median_sales:.2%}")

    if EVAL_MODE in ("pooled", "both") and "rmse_pooled" in locals():
        mean_sales = df["sales"].mean()
        median_sales = df["sales"].median()
        print(f"Pooled RMSE / mean_sales:       {rmse_pooled/mean_sales:.2%}")
        print(f"Pooled RMSE / median_sales:     {rmse_pooled/median_sales:.2%}")


if __name__ == "__main__":
    main()
