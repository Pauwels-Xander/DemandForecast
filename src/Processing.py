"""Data Ingestion and Preprocessing for OrangeJuiceX25 dataset."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_PATH = "OrangeJuiceX25.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(
        "▶ Loaded file:", path,
        "| rows:", df.shape[0],
        "| unique stores:", df["store"].nunique(),
        "| unique brands:", df["brand"].nunique(),
        "| unique weeks:", df["week"].nunique()
    )
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the 17 regressors as per Proposal (no Box–Cox, no HolidayDummy):
      1. OwnLogPrice
      2. LaggedLogPrice_1, _2
      3. MinCompPrice, LogMinCompPrice
      4. FeatFlag, FeatFlag_L1
      5. DealFlag, DealFlag_L1
      6. CategoryFeatCount, CategoryFeatShare
      7. CategoryDealCount, CategoryDealShare
      8. PromoInteraction
      9. CrossFeatPressure
     10. LagLogSales
     11. MarketShareLog
     12. Price_x_Feat, Price_x_Deal
     13. Holidays
    """
    # 0. 复制并重置索引
    df = df.copy().reset_index(drop=True)

    # 1. 确定所有价格列（price1…price11），并按 store 前向填充
    price_cols = sorted(
        [c for c in df.columns if c.startswith("price")],
        key=lambda x: int(x.replace("price", "")),
    )
    df[price_cols] = (
        df.groupby("store")[price_cols]
          .transform(lambda g: g.fillna(method="ffill"))
    )

    # 2. 特征标记
    df["FeatFlag"] = df["feat"].astype(int)
    df["DealFlag"] = df["deal"].astype(int)

    # 3. 计算 OwnLogPrice 和滞后两期
    prices_arr = df[price_cols].to_numpy()
    brand_idxs = df["brand"].astype(int).to_numpy() - 1  # brand 1→price1索引0
    own_prices = prices_arr[np.arange(len(df)), brand_idxs]
    df["OwnLogPrice"]    = np.log(own_prices + 1e-6)
    df["LagLogPrice_1"]  = df.groupby(["store", "brand"])["OwnLogPrice"].shift(1)
    df["LagLogPrice_2"]  = df.groupby(["store", "brand"])["OwnLogPrice"].shift(2)

    # 4. 计算 MinCompPrice & LogMinCompPrice
    comp_arr = prices_arr.copy()
    comp_arr[np.arange(len(df)), brand_idxs] = np.nan
    min_comp = np.nanmin(comp_arr, axis=1)
    df["MinCompPrice"]     = min_comp
    df["LogMinCompPrice"]  = np.log(min_comp + 1e-6)

    # 5. 滞后 feat/deal 标志
    df["FeatFlag_L1"] = df.groupby(["store", "brand"])["FeatFlag"].shift(1)
    df["DealFlag_L1"] = df.groupby(["store", "brand"])["DealFlag"].shift(1)

    # 6. 类别级 feat/deal 计数与份额
    grp = df.groupby(["store", "week"])
    df["CategoryFeatCount"] = grp["FeatFlag"].transform("sum")
    df["CategoryFeatShare"] = df["CategoryFeatCount"] / 11
    df["CategoryDealCount"] = grp["DealFlag"].transform("sum")
    df["CategoryDealShare"] = df["CategoryDealCount"] / 11

    # 7. 促销交互 & 同品类竞争强度
    df["PromoInteraction"]  = df["FeatFlag"] * df["DealFlag"]
    df["OthersFeatSum"]     = grp["FeatFlag"].transform("sum") - df["FeatFlag"]
    df["CrossFeatPressure"] = df["FeatFlag"] * (df["OthersFeatSum"] > 0).astype(int)

    # 8. 滞后对数销量
    df["LagLogSales"] = (
        df.groupby(["store", "brand"])["sales"]
          .shift(1)
          .pipe(lambda s: np.log(s + 1e-6))
    )

    # 9. MarketShareLog
    df["LogSales"]      = np.log(df["sales"] + 1e-6)
    total_sales_store  = grp["sales"].transform("sum")
    df["MarketShareLog"] = df["LogSales"] - np.log(total_sales_store + 1e-6)

    # 10. Price × Feat, Price × Deal
    df["Price_x_Feat"] = df["OwnLogPrice"] * df["FeatFlag"]
    df["Price_x_Deal"] = df["OwnLogPrice"] * df["DealFlag"]

    # 11. Holidays
    df["Holiday"] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] * 110


    # —— 到此共 17 个 regressor ——
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

    # 11. 按 store–brand–week 排序并重置索引
    df = df.sort_values(["store", "brand", "week"])\
           .reset_index(drop=True)

    # 12. 标准化这 17 个特征
    scaler = StandardScaler()
    df[regressors] = scaler.fit_transform(df[regressors].values)

    # 13. 最终只保留 identifiers + sales + regressors
    keep_cols = ["store", "brand", "week", "sales"] + regressors
    df_final = df[keep_cols].copy()

    return df_final


def main():
    df = load_data()
    processed = preprocess_features(df)
    processed.to_csv("processed_dataBobo.csv", index=False)
    print("Processed Data Sample:\n", processed.head())

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f">>> Loaded `{path}`:",
          f"stores={df['store'].nunique()}",
          f"brands={df['brand'].nunique()}",
          f"weeks={df['week'].nunique()}")
    return df


if __name__ == "__main__":
    main()
