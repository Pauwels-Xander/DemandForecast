from __future__ import annotations

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

DATA_PATH = "OrangeJuiceX25.csv"
WINDOW_SIZE = 52  # number of weeks to train on each iteration

# --------------------------------------------------------------------------
# Example fixed hyperparameters (tuned once offline). You can change these. :)
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
# --------------------------------------------------------------------------

# Choose one of:
#   "per_series_all" -> run per‐series XGBoost on every (store, brand) and average RMSE
#   "pooled"         _> run pooled XGBoost on all store×brand rows by calendar week
#   "both"           -> do both, printing both results
EVAL_MODE = "per_series_all"  # options: "per_series_all", "pooled", "both"

# --------------------------------------------------------------------------
# 1) List of all regressors you want to use
FEATURE_COLS = [
    "store", "brand", "week",
    "OwnLogPrice",
    "LagLogPrice_2",
    "MinCompPrice",
    "CategoryFeatShare",
    "CrossFeatPressure",
    "LagLogSales",
    "DealFlag_L1"
]
# --------------------------------------------------------------------------

def load_processed() -> pd.DataFrame:
    """Load and preprocess the dataset using utilities from preprocessing."""
    from ProcessingOutliers import load_data, preprocess_features

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

    if len(subset) < WINDOW_SIZE + 1:
        raise ValueError(
            f"Not enough data for store={store}, brand={brand} "
            f"({len(subset)} rows; need ≥ {WINDOW_SIZE+1})"
        )

    # Build feature list for this series (drop grouping + raw sales)
    feature_cols = [c for c in FEATURE_COLS if c not in ("store", "brand", "sales")]

    preds: list[float] = []
    actuals: list[float] = []

    for end in range(WINDOW_SIZE, len(subset)):
        window_df = subset.iloc[end - WINDOW_SIZE : end]

        X_train = window_df[feature_cols]
        y_train = window_df["y_next_log"]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        bst = xgb.train(fixed_params, dtrain, num_boost_round=300, verbose_eval=False)

        test_row = subset.iloc[end : end + 1]
        dtest = xgb.DMatrix(test_row[feature_cols])
        pred_log = bst.predict(dtest)[0]
        preds.append(np.expm1(pred_log))
        actuals.append(test_row["sales"].values[0])

    return float(np.sqrt(mean_squared_error(actuals, preds)))

def xgb_pooled_52(df: pd.DataFrame, fixed_params: dict) -> float:
    """
    Rolling‐window pooled XGBoost using exactly 52 weeks of history (no validation split).
    Trains on calendar weeks [t-52 ... t-1] (across all store×brand rows) to predict week t.
    Returns RMSE over all such predictions.
    """
    df = df.sort_values(["store", "brand", "week"]).reset_index(drop=True)
    df["y_next_log"] = (
        df.groupby(["store", "brand"])["sales"]
        .shift(-1)
        .apply(np.log1p)
    )
    df = df.dropna(subset=["y_next_log"]).reset_index(drop=True)

    # 1) One-hot encode store & brand
    df = pd.concat([
        df,
        pd.get_dummies(df["store"], prefix="store"),
        pd.get_dummies(df["brand"], prefix="brand"),
    ], axis=1)

    # Build final feature list: numeric regressors + encoded store/brand
    encoded_cols = [c for c in df.columns if c.startswith("store_") or c.startswith("brand_")]
    numeric_feats = [c for c in FEATURE_COLS if c not in ("store", "brand", "sales")]
    feature_cols = numeric_feats + encoded_cols

    weeks = sorted(df["week"].unique())
    if len(weeks) < WINDOW_SIZE + 1:
        raise ValueError(f"Need ≥ {WINDOW_SIZE+1} distinct weeks; found {len(weeks)}.")

    preds: list[float] = []
    actuals: list[float] = []

    for idx in range(WINDOW_SIZE, len(weeks)):
        end_week = weeks[idx - 1]
        next_week = weeks[idx]

        block_mask = (df["week"] >= end_week - (WINDOW_SIZE - 1)) & (df["week"] <= end_week)
        block_df = df[block_mask]
        if len(set(block_df["week"])) != WINDOW_SIZE:
            continue  # skip if any week missing

        dtrain = xgb.DMatrix(block_df[feature_cols], label=block_df["y_next_log"])
        bst = xgb.train(fixed_params, dtrain, num_boost_round=400, verbose_eval=False)

        test_df = df[df["week"] == next_week]
        if test_df.empty:
            continue

        dtest = xgb.DMatrix(test_df[feature_cols])
        preds.extend(np.expm1(bst.predict(dtest)).tolist())
        actuals.extend(test_df["sales"].tolist())

    if not preds:
        raise ValueError("No predictions generated; maybe insufficient weekly coverage.")

    return float(np.sqrt(mean_squared_error(actuals, preds)))

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
        for store, brand in tqdm(combos, desc="Per‐series XGBoost"):
            try:
                per_series_rmses.append(
                    xgb_per_series_52(df, int(store), int(brand), FIXED_PARAMS)
                )
            except ValueError:
                continue

        if per_series_rmses:
            avg_per_series = float(np.mean(per_series_rmses))
            print(f"Average per-series XGBoost RMSE: {avg_per_series:.2f}")
        else:
            print("No valid series for per-series evaluation.")

    if EVAL_MODE in ("pooled", "both"):
        try:
            rmse_pooled = xgb_pooled_52(df, FIXED_PARAMS)
            print(f"Pooled XGBoost RMSE: {rmse_pooled:.2f}")
        except ValueError as e:
            print(f"Pooled evaluation skipped: {e}")

    # 3) Relative errors
    mean_sales = df["sales"].mean()
    median_sales = df["sales"].median()

    if EVAL_MODE in ("per_series_all", "both") and 'avg_per_series' in locals():
        print(f"\nPer‐series RMSE / mean_sales:   {avg_per_series/mean_sales:.2%}")
        print(f"Per‐series RMSE / median_sales: {avg_per_series/median_sales:.2%}")

    if EVAL_MODE in ("pooled", "both") and 'rmse_pooled' in locals():
        print(f"Pooled RMSE / mean_sales:       {rmse_pooled/mean_sales:.2%}")
        print(f"Pooled RMSE / median_sales:     {rmse_pooled/median_sales:.2%}")

if __name__ == "__main__":
    main()
