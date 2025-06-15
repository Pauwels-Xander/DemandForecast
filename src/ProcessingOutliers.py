"""Data Ingestion and Preprocessing for OrangeJuiceX25 dataset."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_PATH = "OrangeJuiceX25.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # map each brand code → its pack size (in oz)
    oz_map = {
        1: 64,   # Tropicana Premium 64 oz
        2: 96,   # Tropicana Premium 96 oz
        3: 64,   # Florida’s Natural 64 oz
        4: 64,   # Tropicana 64 oz
        5: 64,   # Minute Maid 64 oz
        6: 96,   # Minute Maid 96oz
        7: 64,   # Citrus Hill 64 oz
        8: 64,   # Tree Fresh 64 oz
        9: 64,   # Florida Gold 64 oz
       10: 64,   # Dominick’s 64 oz
       11:128    # Dominick’s 128 oz
    }

    # compute “units sold” = ounces sold ÷ oz per pack
    df['sales'] = df['sales'] / df['brand'].map(oz_map)
    print(max(df["sales"]))

    print(
        "▶ Loaded file:", path,
        "| rows:", df.shape[0],
        "| unique stores:", df["store"].nunique(),
        "| unique brands:", df["brand"].nunique(),
        "| unique weeks:", df["week"].nunique()
    )
    return df



IQR_MULTIPLIER = 4

def preprocess_features(df: pd.DataFrame, iqr_mul: float = IQR_MULTIPLIER) -> pd.DataFrame:
    """
    … same docstring …
    """

    # 0. Copy & reset index
    df = df.copy().reset_index(drop=True)

    # --- BEGIN IQR OUTLIER TRIMMING WITH PARAMETER ---
    def cap_iqr(x: pd.Series) -> pd.Series:
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_mul * iqr
        upper = q3 + iqr_mul * iqr
        return x.clip(lower, upper)

    def count_outliers(x: pd.Series) -> int:
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - iqr_mul * iqr, q3 + iqr_mul * iqr
        return ((x < lower) | (x > upper)).sum()

    total_to_cap = (
        df.groupby(["store", "brand"])["sales"]
          .apply(count_outliers)
          .sum()
    )
    print(f"⚠ Will cap a total of {total_to_cap} sales values (IQR × {iqr_mul}).")

    df["sales"] = (
        df.groupby(["store", "brand"])["sales"]
          .transform(cap_iqr)
    )
    # --- END IQR OUTLIER TRIMMING ---

    # 1. Identify all price columns and forward-fill per store
    price_cols = sorted(
        [c for c in df.columns if c.startswith("price")],
        key=lambda x: int(x.replace("price", ""))
    )
    df[price_cols] = (
        df.groupby("store")[price_cols]
          .transform(lambda g: g.fillna(method="ffill"))
    )

    # 2. Flags for feature & deal
    df["FeatFlag"] = df["feat"].astype(int)
    df["DealFlag"] = df["deal"].astype(int)

    # 3. OwnLogPrice and its lags
    prices_arr = df[price_cols].to_numpy()
    brand_idxs = df["brand"].astype(int).to_numpy() - 1
    own_prices = prices_arr[np.arange(len(df)), brand_idxs]
    df["OwnLogPrice"]    = np.log(own_prices + 1e-6)
    df["LagLogPrice_1"]  = df.groupby(["store", "brand"])["OwnLogPrice"].shift(1)
    df["LagLogPrice_2"]  = df.groupby(["store", "brand"])["OwnLogPrice"].shift(2)

    # 4. MinCompPrice & LogMinCompPrice
    comp_arr = prices_arr.copy()
    comp_arr[np.arange(len(df)), brand_idxs] = np.nan
    min_comp = np.nanmin(comp_arr, axis=1)
    df["MinCompPrice"]     = min_comp
    df["LogMinCompPrice"]  = np.log(min_comp + 1e-6)

    # 5. Lagged feature/deal flags
    df["FeatFlag_L1"] = df.groupby(["store", "brand"])["FeatFlag"].shift(1)
    df["DealFlag_L1"] = df.groupby(["store", "brand"])["DealFlag"].shift(1)

    # 6. Category-level feature/deal counts & shares
    grp = df.groupby(["store", "week"])
    df["CategoryFeatCount"] = grp["FeatFlag"].transform("sum")
    df["CategoryFeatShare"] = df["CategoryFeatCount"] / 11
    df["CategoryDealCount"] = grp["DealFlag"].transform("sum")
    df["CategoryDealShare"] = df["CategoryDealCount"] / 11

    # 7. Promo interaction & cross-feature pressure
    df["PromoInteraction"]  = df["FeatFlag"] * df["DealFlag"]
    df["OthersFeatSum"]     = grp["FeatFlag"].transform("sum") - df["FeatFlag"]
    df["CrossFeatPressure"] = df["FeatFlag"] * (df["OthersFeatSum"] > 0).astype(int)

    # 8. Lagged log-sales
    df["LagLogSales"] = (
        df.groupby(["store", "brand"])["sales"]
          .shift(1)
          .pipe(lambda s: np.log(s + 1e-6))
    )

    # 9. Market share log
    df["LogSales"]      = np.log(df["sales"] + 1e-6)
    total_sales_store  = grp["sales"].transform("sum")
    df["MarketShareLog"] = df["LogSales"] - np.log(total_sales_store + 1e-6)

    # 10. Price × Feat, Price × Deal interactions
    df["Price_x_Feat"] = df["OwnLogPrice"] * df["FeatFlag"]
    df["Price_x_Deal"] = df["OwnLogPrice"] * df["DealFlag"]

    # Regressor list (should be 17 after trimming)
    regressors = [
        "OwnLogPrice", "LagLogPrice_1", "LagLogPrice_2",
        "MinCompPrice", "LogMinCompPrice",
        "FeatFlag", "FeatFlag_L1",
        "DealFlag", "DealFlag_L1",
        "CategoryFeatCount", "CategoryFeatShare",
        "CategoryDealCount", "CategoryDealShare",
        "PromoInteraction", "CrossFeatPressure",
        "LagLogSales", "MarketShareLog",
        "Price_x_Feat", "Price_x_Deal", "Holiday"
    ]
    # 11.1 Holidays
    df["Holiday"] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] * 110

    # 11.2 Sort & reset index
    df = df.sort_values(["store", "brand", "week"]).reset_index(drop=True)

    # 12. Standardize regressors
    scaler = StandardScaler()
    df[regressors] = scaler.fit_transform(df[regressors].values)

    # 13. Keep only identifiers + sales + regressors
    keep_cols = ["store", "brand", "week", "sales"] + regressors
    df_final = df[keep_cols].copy()

    return df_final


def main():
    df_raw = load_data()
    raw_n = df_raw.shape[0]

    processed = preprocess_features(df_raw)
    proc_n = processed.shape[0]

    print(f"\n▶ Observations before preprocessing: {raw_n}")
    print(f"▶ Observations after  preprocessing: {proc_n}")
    print(f"▶ Observations removed             : {raw_n - proc_n}\n")

    processed.to_csv("processed_data.csv", index=False)
    print("Processed Data Sample:\n", processed.head())


if __name__ == "__main__":
    main()


# --- Visualization Section ---

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load unitized & capped data
df = load_data("OrangeJuiceX25.csv")

# 2. define IQR capping on units-sold
def cap_iqr(x: pd.Series) -> pd.Series:
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 4*iqr, q3 + 4*iqr
    return x.clip(lower, upper)

# 3. make the sales_trimmed column on your already unitized df
df["sales_trimmed"] = df.groupby(["store","brand"])["sales"].transform(cap_iqr)

# 4. boxplot & histograms on df["sales"] and df["sales_trimmed"] …
store_id = df["store"].unique()[0]
brand_id = df[df["store"] == store_id]["brand"].unique()[0]
sub = df[(df["store"] == store_id) & (df["brand"] == brand_id)]

plt.figure(figsize=(8, 6))
sns.boxplot(data=sub[["sales", "sales_trimmed"]])
plt.title(f"Store {store_id}, Brand {brand_id} — Sales Before vs After IQR Capping")
plt.xlabel("Original vs Trimmed")
plt.ylabel("Sales")
plt.show()

# 5. 全数据直方图对比 / histograms overall
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df["sales"], bins=50, kde=False)
plt.title("Original Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.histplot(df["sales_trimmed"], bins=50, kde=False)
plt.title("Trimmed Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
