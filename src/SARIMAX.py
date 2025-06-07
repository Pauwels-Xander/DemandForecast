"""SARIMAX rolling‐window model for OrangeJuiceX25 dataset (window size = 52),
with fixes for frequency and optimizer warnings, and a progress bar."""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm  # For progress bar

DATA_PATH = "OrangeJuiceX25.csv"
WINDOW_SIZE = 52  # rolling window length in weeks


def load_processed() -> pd.DataFrame:
    """
    Load and preprocess the dataset using existing utilities.
    Assumes "load_data" and "preprocess_features" return a DataFrame
    with at least columns: ["week", "store", "brand", "sales", "price...", "deal", "feat"].
    """
    from Processing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)
    return df


def sarimax_per_series(
    subset: pd.DataFrame,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> float:
    """
    Rolling‐window SARIMAX for one store–brand pair, using a 52‐week DateTime index
    and convergence/frequency fixes.

    Parameters
    ----------
    subset : pd.DataFrame
        Preprocessed data for a single (store, brand) pair, containing at least:
        - "week": a date string or period that can be converted to datetime
        - "sales": the target time series
        - exogenous columns: any column starting with "price", plus "deal", "feat"
    order : tuple
        ARIMA(p, d, q) order.
    seasonal_order : tuple
        Seasonal order (P, D, Q, s).

    Returns
    -------
    float
        RMSE across all one‐step forecasts produced by sliding a 52‐week window.
    """
    # 1) Sort by "week", convert to datetime, set as index
    subset = subset.sort_values("week").reset_index(drop=True)
    subset["week"] = pd.to_datetime(subset["week"])
    subset.set_index("week", inplace=True)

    # 2) Infer and set a weekly frequency on the index to avoid ValueWarning
    inferred_freq = pd.infer_freq(subset.index)
    if inferred_freq is None:
        # If inference fails, explicitly set to weekly freq
        subset.index = pd.DatetimeIndex(subset.index.values, freq="W")
    else:
        subset.index = pd.DatetimeIndex(subset.index.values, freq=inferred_freq)

    # 3) Create one‐step‐ahead target column and drop the last row
    subset["y_next"] = subset["sales"].shift(-1)
    subset.dropna(inplace=True)

    # 4) Identify exogenous columns (drop "week" since it's now the index)
    exog_cols = [c for c in subset.columns if c.startswith("price")] + ["deal", "feat"]

    preds: list[float] = []
    actuals: list[float] = []

    # 5) Suppress ConvergenceWarning after applying mitigations
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # 6) Rolling‐window loop: each train set has exactly WINDOW_SIZE observations
    for end in range(WINDOW_SIZE, len(subset)):
        start = end - WINDOW_SIZE
        train = subset.iloc[start:end]
        test = subset.iloc[end : end + 1]

        # 6a) Scale exogenous features based on the training window
        scaler = StandardScaler()
        train_exog_scaled = scaler.fit_transform(train[exog_cols])
        test_exog_scaled = scaler.transform(test[exog_cols])

        # 6b) Fit SARIMAX with increased maxiter and appropriate convergence args
        model = SARIMAX(
            train["sales"],
            order=order,
            seasonal_order=seasonal_order,
            exog=train_exog_scaled,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        res = model.fit(
            disp=False,
            maxiter=500,        # increase iterations
            pgtol=1e-5,         # set gradient tolerance for 'lbfgs'
            method="lbfgs",     # default optimizer: 'lbfgs'
        )

        # 6c) Forecast the next observation
        forecast = res.predict(
            start=len(train),
            end=len(train),
            exog=test_exog_scaled,
        )
        preds.append(float(forecast.iloc[0]))
        actuals.append(float(test["y_next"].values[0]))

    mse = mean_squared_error(actuals, preds)
    return float(np.sqrt(mse))


def evaluate_all_series(
    df: pd.DataFrame,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> pd.DataFrame:
    """
    Compute rolling‐window SARIMAX RMSE for every (store, brand) pair,
    with a progress bar.

    Returns a DataFrame with columns: [store, brand, rmse].
    """
    results = []

    stores = df["store"].unique()
    brands = df["brand"].unique()
    combos = [(s, b) for s in stores for b in brands]

    for store_id, brand_id in tqdm(
        combos,
        total=len(combos),
        desc="Evaluating SARIMAX store-brand combos",
        unit="combo",
    ):
        subset = df[(df["store"] == store_id) & (df["brand"] == brand_id)].copy()

        # Need at least WINDOW_SIZE + 1 rows to train and then forecast
        if len(subset) < WINDOW_SIZE + 1:
            continue

        rmse = sarimax_per_series(
            subset=subset,
            order=order,
            seasonal_order=seasonal_order,
        )
        results.append({"store": store_id, "brand": brand_id, "rmse": rmse})

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Load and preprocess all data
    df = load_processed()

    # Example: ARIMA(1,0,0) with no seasonal terms
    # You can adjust order and seasonal_order as needed.
    all_rmse_df = evaluate_all_series(
        df,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
    )

    print("SARIMAX rolling‐window RMSE (per series):")
    print(all_rmse_df.to_string(index=False))
