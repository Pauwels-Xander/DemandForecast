from __future__ import annotations

import random
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from econometric_models import load_processed, gets_per_series


def check_data_leakage(df: pd.DataFrame) -> None:
    """Run one random GETS window with debug printing."""
    stores = df["store"].unique().tolist()
    brands = df["brand"].unique().tolist()
    store = random.choice(stores)
    brand = random.choice(brands)
    print(f"\nChecking data leakage for store={store}, brand={brand}")
    gets_per_series(df, store, brand, debug=True)


def smearing_transform(log_pred: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Apply a unified log→level transform using Duan's smearing."""
    smear = np.exp((residuals ** 2).mean() / 2)
    return np.exp(log_pred) * smear


def error_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table of per-series RMSE using GETS."""
    results = []
    for s in sorted(df["store"].unique()):
        for b in sorted(df["brand"].unique()):
            try:
                rmse = gets_per_series(df, s, b)
                results.append({"store": s, "brand": b, "rmse": rmse})
            except ValueError:
                continue
    pivot = (
        pd.DataFrame(results)
        .pivot(index="store", columns="brand", values="rmse")
        .sort_index()
    )
    print("\nRMSE heat-map (store x brand):")
    print(pivot)
    return pivot


def residual_tests(df: pd.DataFrame, store: int, brand: int) -> None:
    """Ljung-Box and ARCH tests on GETS residuals."""
    rmse, resid, _, _ = gets_per_series(
        df, store, brand, return_residuals=True
    )
    print(f"\nGETS RMSE for store={store}, brand={brand}: {rmse:.2f}")
    lb = acorr_ljungbox(resid, lags=[8], return_df=True)
    arch = het_arch(resid, nlags=8)
    print("Ljung-Box p-value:", lb["lb_pvalue"].iloc[0])
    print("ARCH p-value:", arch[1])


if __name__ == "__main__":
    df_proc = load_processed()
    check_data_leakage(df_proc)
    error_heatmap(df_proc)
    residual_tests(df_proc, df_proc["store"].iloc[0], df_proc["brand"].iloc[0])

