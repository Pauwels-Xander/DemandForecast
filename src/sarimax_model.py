"""SARIMAX baseline model for OrangeJuiceX25 dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_PATH = "data/OrangeJuiceX25.csv"


def load_processed() -> pd.DataFrame:
    """Load and preprocess the dataset using existing utilities."""
    from src.data_ingestion_preprocessing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)
    return df


# ---------------------------------------------------------------------------
# Per-series SARIMAX
# ---------------------------------------------------------------------------

def sarimax_per_series(
    df: pd.DataFrame,
    store: int,
    brand: int,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> float:
    """Rolling-window SARIMAX for one store-brand pair.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset.
    store, brand : int
        Identifiers for the series.
    order : tuple
        ARIMA (p,d,q) order.
    seasonal_order : tuple
        Seasonal order (P,D,Q,s).

    Returns
    -------
    float
        RMSE across one-step forecasts.
    """
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week")

    # one-step ahead target
    subset["y_next"] = subset["sales"].shift(-1)
    subset.dropna(inplace=True)

    exog_cols = [c for c in subset.columns if c.startswith("price")] + [
        "deal",
        "feat",
    ]

    preds: list[float] = []
    actuals: list[float] = []

    for end in range(60, len(subset)):
        train = subset.iloc[:end]
        test = subset.iloc[end : end + 1]

        model = SARIMAX(
            train["sales"],
            order=order,
            seasonal_order=seasonal_order,
            exog=train[exog_cols],
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        forecast = model.predict(
            start=len(train),
            end=len(train),
            exog=test[exog_cols],
        )
        preds.append(forecast.iloc[0])
        actuals.append(test["y_next"].values[0])

    rmse = mean_squared_error(actuals, preds, squared=False)
    return rmse


if __name__ == "__main__":
    df = load_processed()
    rmse = sarimax_per_series(df, store=1, brand=1, order=(1, 0, 0))
    print("SARIMAX per-series RMSE:", rmse)
