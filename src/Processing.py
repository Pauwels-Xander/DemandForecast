"""Data Ingestion, Exploration, and Preprocessing for OrangeJuiceX25 dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

DATA_PATH = "OrangeJuiceX25.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset into a pandas DataFrame."""
    df = pd.read_csv(path)
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Print basic info and create exploration plots."""
    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes)

    price_cols = [c for c in df.columns if c.startswith("price")]
    summary = df[price_cols + ["deal", "feat", "sales"]].describe()
    print("\nSummary statistics:\n", summary)

    # Distribution of sales
    plt.figure(figsize=(6, 4))
    sns.histplot(df["sales"], bins=50)
    plt.title("Distribution of Weekly Sales")
    plt.xlabel("sales")
    plt.tight_layout()
    plt.show()

    # Correlation matrix of price columns
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[price_cols].corr(), cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Correlation Matrix of Price Series")
    plt.tight_layout()
    plt.show()


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features."""
    price_cols = [c for c in df.columns if c.startswith("price")]

    # Replace zero or negative prices with NaN and forward fill
    df[price_cols] = df[price_cols].mask(df[price_cols] <= 0)
    df[price_cols] = df.groupby(["store", "brand"])[price_cols].fillna(method="ffill")

    # Flag where price was originally zero or NaN
    for col in price_cols:
        df[f"{col}_was_zero"] = df[col].isna().astype(int)

    # Create lagged sales features per store-brand
    df = df.sort_values(["store", "brand", "week"])  # ensure chronological order
    for lag in [1, 2, 3]:
        df[f"sales_lag{lag}"] = df.groupby(["store", "brand"])["sales"].shift(lag)

    # Encode store_id and brand_id as categorical codes
    df["store_id"] = df["store"].astype("category").cat.codes
    df["brand_id"] = df["brand"].astype("category").cat.codes

    # Cyclical seasonal indicators (week-of-year modulo 52)
    df["week_mod"] = df["week"] % 52
    df["sin_week"] = np.sin(2 * np.pi * df["week_mod"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_mod"] / 52)
    df.drop(columns="week_mod", inplace=True)

    # Standardize price and lagged sales features
    scaler = StandardScaler()
    cont_cols = price_cols + [f"sales_lag{lag}" for lag in [1, 2, 3]]
    df[cont_cols] = scaler.fit_transform(df[cont_cols])

    return df


def main():
    df = load_data()
    explore_data(df)
    df_processed = preprocess_features(df)
    df_processed.to_csv("processed_data.csv", index=False)
    print("\nProcessed Data Sample:\n", df_processed.head())


if __name__ == "__main__":
    main()
