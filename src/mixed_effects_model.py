"""Linear mixed-effects model for pooled demand forecasting."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

DATA_PATH = "data/OrangeJuiceX25.csv"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_processed() -> pd.DataFrame:
    """Load and preprocess the dataset using utilities from preprocessing."""
    from src.data_ingestion_preprocessing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)
    return df


# ---------------------------------------------------------------------------
# Mixed-effects model across all store-brand series
# ---------------------------------------------------------------------------

def mixedlm_pooled(df: pd.DataFrame) -> float:
    """Rolling-window linear mixed-effects model with random intercepts.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset.

    Returns
    -------
    float
        RMSE across one-step forecasts for all series.
    """
    df = df.sort_values(["store", "brand", "week"])
    df["y_next"] = df.groupby(["store", "brand"])["sales"].shift(-1)
    df = df.dropna(subset=["y_next"])  # last week of each series

    # group identifier for random intercepts
    df["series"] = df["store"].astype(str) + "_" + df["brand"].astype(str)

    lag_cols = [f"sales_lag{lag}" for lag in [1, 2, 3]]
    price_cols = [c for c in df.columns if c.startswith("price") and not c.endswith("was_zero")]
    week_dummies = pd.get_dummies(df["week"] % 52, prefix="week")
    df = pd.concat([df, week_dummies], axis=1)
    feature_cols = lag_cols + price_cols + ["deal", "feat"] + list(week_dummies.columns)

    preds: list[float] = []
    actuals: list[float] = []

    weeks = sorted(df["week"].unique())
    for end_week in weeks[59:-1]:
        train = df[df["week"] <= end_week]
        test = df[df["week"] == end_week + 1]

        formula = "y_next ~ " + " + ".join(feature_cols)
        model = sm.MixedLM.from_formula(formula, groups=train["series"], data=train)
        result = model.fit(method="lbfgs", reml=False)

        pred = result.predict(test)
        preds.extend(pred)
        actuals.extend(test["y_next"])

    rmse = mean_squared_error(actuals, preds, squared=False)
    return rmse


def main() -> None:
    df = load_processed()
    rmse = mixedlm_pooled(df)
    print(f"Mixed-effects pooled RMSE: {rmse:.3f}")


if __name__ == "__main__":
    main()
