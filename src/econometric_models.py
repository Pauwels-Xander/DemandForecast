import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

DATA_PATH = "OrangeJuiceX25.csv"
WINDOW_SIZE = 52  # number of weeks to train on each iteration

def load_processed() -> pd.DataFrame:
    """Load raw data and apply preprocessing."""
    from Processing import load_data, preprocess_features

    df_raw = load_data(DATA_PATH)
    print(f"After load_data: total rows = {len(df_raw)}")

    df_processed = preprocess_features(df_raw)
    print(f"After preprocess_features: total rows = {len(df_processed)}")

    return df_processed

def naive_rmse_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """
    Compute naïve persistence RMSE for a single store–brand series.
    """
    sub = (
        df[(df["store"] == store) & (df["brand"] == brand)]
        .sort_values("week")
        .reset_index(drop=True)
    )
    preds = sub["sales"].shift(1).dropna()
    actuals = sub["sales"].iloc[1:].reset_index(drop=True)
    return float(np.sqrt(mean_squared_error(actuals, preds)))

def ridge_per_series(df: pd.DataFrame, store: int, brand: int) -> tuple[float, float, list, list]:
    """
    Rolling-window RidgeCV with standardized features.
    Returns (RMSE, MAPE, feats, coef_list).
    """
    sub = (
        df[(df["store"] == store) & (df["brand"] == brand)]
        .sort_values("week")
        .reset_index(drop=True)
    )
    sub["y_next"] = np.log(sub["sales"].shift(-1))
    sub = sub.dropna().reset_index(drop=True)

    if len(sub) < WINDOW_SIZE + 1:
        raise ValueError(f"Not enough data for Ridge store={store},brand={brand}")

    exclude = ["store", "brand", "week", "sales", "y_next"]
    feats = [c for c in sub.columns if c not in exclude]

    preds, actuals = [], []
    coef_list = []
    tscv = TimeSeriesSplit(n_splits=5)
    for end in range(WINDOW_SIZE, len(sub)):
        train = sub.iloc[end - WINDOW_SIZE : end]
        test = sub.iloc[end : end + 1]

        X_train = train[feats].astype(float)
        y_train = train["y_next"].astype(float)
        X_test = test[feats].astype(float)
        y_true = test["sales"].iloc[0]

        # 标准化 X_train 和 X_test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RidgeCV(
            alphas=np.logspace(-3, 3, 10),
            cv=tscv,
            scoring="neg_mean_squared_error",
        ).fit(X_train_scaled, y_train)

        coef_list.append(model.coef_)  # 记录标准化后的系数

        pred_train = model.predict(X_train_scaled)
        resid = y_train - pred_train
        smear = np.mean(np.exp(resid))

        pred_log = model.predict(X_test_scaled)[0]
        pred_sales = smear * np.exp(pred_log)

        preds.append(pred_sales)
        actuals.append(y_true)

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mape = float(np.mean(np.abs((np.array(actuals) - np.array(preds)) / np.array(actuals))))
    return rmse, mape, feats, coef_list

def lasso_per_series(df: pd.DataFrame, store: int, brand: int) -> tuple[float, float, list, list, list]:
    """
    Rolling-window LASSO with standardized features.
    Returns (RMSE, MAPE, feats, selected_feats_list, coef_list).
    """
    sub = (
        df[(df["store"] == store) & (df["brand"] == brand)]
        .sort_values("week")
        .reset_index(drop=True)
    )
    sub["y_next"] = np.log(sub["sales"].shift(-1))
    sub = sub.dropna().reset_index(drop=True)

    if len(sub) < WINDOW_SIZE + 1:
        raise ValueError(f"Not enough data for LASSO store={store},brand={brand}")

    exclude = ["store", "brand", "week", "sales", "y_next"]
    feats = [c for c in sub.columns if c not in exclude]

    preds, actuals = [], []
    selected_feats_list = []
    coef_list = []
    tscv = TimeSeriesSplit(n_splits=5)
    for end in range(WINDOW_SIZE, len(sub)):
        train = sub.iloc[end - WINDOW_SIZE : end]
        test = sub.iloc[end : end + 1]

        X_train = train[feats].astype(float)
        y_train = train["y_next"].astype(float)
        X_test = test[feats].astype(float)
        y_true = test["sales"].iloc[0]

        # 标准化 X_train 和 X_test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LassoCV(cv=tscv, random_state=0, max_iter=10000, tol=1e-5).fit(X_train_scaled, y_train)
        
        selected_feats = [feats[i] for i in range(len(feats)) if model.coef_[i] != 0]
        selected_feats_list.append(selected_feats)
        coef_list.append(model.coef_)

        pred_train = model.predict(X_train_scaled)
        resid = y_train - pred_train
        smear = np.mean(np.exp(resid))

        pred_log = model.predict(X_test_scaled)[0]
        pred_sales = smear * np.exp(pred_log)

        preds.append(pred_sales)
        actuals.append(y_true)

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mape = float(np.mean(np.abs((np.array(actuals) - np.array(preds)) / np.array(actuals))))
    return rmse, mape, feats, selected_feats_list, coef_list

def gets_per_series(df: pd.DataFrame, store: int, brand: int) -> tuple[float, float, list, list]:
    """
    Rolling-window GETS (OLS + t-pruning) with standardized features.
    Returns (RMSE, MAPE, feats, signif_feats_list), with corrected Duan smearing.
    """
    sub = (
        df[(df["store"] == store) & (df["brand"] == brand)]
        .sort_values("week")
        .reset_index(drop=True)
    )
    sub["y_next"] = np.log(sub["sales"].shift(-1))
    sub = sub.dropna().reset_index(drop=True)

    if len(sub) < WINDOW_SIZE + 1:
        raise ValueError(f"Not enough data for GETS store={store},brand={brand}")

    exclude = ["store", "brand", "week", "sales", "y_next"]
    feats = [c for c in sub.columns if c not in exclude]

    preds, actuals = [], []
    signif_feats_list = []
    for end in range(WINDOW_SIZE, len(sub)):
        train = sub.iloc[end - WINDOW_SIZE : end]
        test = sub.iloc[end : end + 1]

        X = train[feats].astype(float)
        y = train["y_next"].astype(float)
        X_test = test[feats].astype(float)
        y_true = test["sales"].iloc[0]

        # 标准化 X 和 X_test
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)

        # 将标准化后的数据转换为 DataFrame，保留列名
        X_scaled = pd.DataFrame(X_scaled, columns=feats, index=X.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feats, index=X_test.index)

        X_const = sm.add_constant(X_scaled)
        model = sm.OLS(y, X_const).fit()

        # t 检验剪枝
        while True:
            # get t-values without the constant
            tvals = model.tvalues.drop(labels="const", errors="ignore")
            # drop any NaNs so we never pick 'nan' as the worst
            tvals = tvals.dropna()
            # if there are no more features, or all remaining are significant, stop
            if tvals.empty or (tvals.abs() >= 1.96).all():
                break

            # pick the feature with the smallest |t-value|
            worst = tvals.abs().idxmin()

            # prune it out
            X_const = X_const.drop(columns=[worst])
            X_test_scaled = X_test_scaled.drop(columns=[worst])

            # re-fit on the reduced set
            model = sm.OLS(y, X_const).fit()

        # 记录显著特征
        signif_feats = [feat for feat, tval in zip(X_const.columns[1:], model.tvalues[1:]) if abs(tval) >= 1.96]
        signif_feats_list.append(signif_feats)

        fitted = model.predict(X_const)
        resid = y - fitted
        smear = np.mean(np.exp(resid))

        pred_log = model.predict(sm.add_constant(X_test_scaled, has_constant="add")).iloc[0]
        pred_sales = smear * np.exp(pred_log)

        preds.append(pred_sales)
        actuals.append(y_true)

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mape = float(np.mean(np.abs((np.array(actuals) - np.array(preds)) / np.array(actuals))))
    return rmse, mape, feats, signif_feats_list

if __name__ == "__main__":
    df = load_processed()
    combos = df[["store", "brand"]].drop_duplicates().values
    print(f"Found {len(combos)} store–brand combinations.\n")

    results = []
    lasso_selected_counts = Counter()
    lasso_coef_values = defaultdict(list)
    gets_signif_counts = Counter()
    ridge_coef_values = defaultdict(list)

    for store, brand in tqdm(combos, desc="Processing combos"):
        try:
            naive = naive_rmse_series(df, store, brand)

            r_rmse, r_mape, feats_r, coef_list_r = ridge_per_series(df, store, brand)
            for coef in coef_list_r:
                for feat, val in zip(feats_r, coef):
                    ridge_coef_values[feat].append(val)

            l_rmse, l_mape, feats_l, selected_feats_list_l, coef_list_l = lasso_per_series(df, store, brand)
            for selected_feats in selected_feats_list_l:
                for feat in selected_feats:
                    lasso_selected_counts[feat] += 1
            for coef in coef_list_l:
                for feat, val in zip(feats_l, coef):
                    if val != 0:
                        lasso_coef_values[feat].append(val)

            g_rmse, g_mape, feats_g, signif_feats_list_g = gets_per_series(df, store, brand)
            for signif_feats in signif_feats_list_g:
                for feat in signif_feats:
                    gets_signif_counts[feat] += 1

            results.append({
                "store": store,
                "brand": brand,
                "naive_rmse": naive,
                "ridge_rmse": r_rmse, "ridge_rel": r_rmse / naive, "ridge_mape": r_mape,
                "lasso_rmse": l_rmse, "lasso_rel": l_rmse / naive, "lasso_mape": l_mape,
                "gets_rmse":  g_rmse, "gets_rel":  g_rmse / naive, "gets_mape":  g_mape,
            })
        except ValueError:
            continue

    res_df = pd.DataFrame(results)
    print("\nAverage metrics across all combos:")
    print(res_df[[
        "ridge_rmse", "ridge_rel", "ridge_mape",
        "lasso_rmse", "lasso_rel", "lasso_mape",
        "gets_rmse", "gets_rel", "gets_mape"
    ]].mean().to_frame("mean_value"))

    # Aggregate total runs for normalization
    total_runs = sum(len(df[(df["store"] == store) & (df["brand"] == brand)]) - WINDOW_SIZE
                    for store, brand in combos if len(df[(df["store"] == store) & (df["brand"] == brand)]) > WINDOW_SIZE)

    # LASSO Summary
    print("\n### LASSO Feature Selection Summary")
    lasso_summary = []
    for feat in feats_l:
        sel_rate = lasso_selected_counts[feat] / total_runs if total_runs > 0 else 0
        mean_coef = np.mean(lasso_coef_values[feat]) if lasso_coef_values[feat] else 0
        std_coef = np.std(lasso_coef_values[feat]) if lasso_coef_values[feat] else 0
        lasso_summary.append({
            "Feature": feat,
            "Selection Rate": sel_rate,
            "Mean Coef": mean_coef,
            "Std Coef": std_coef
        })
        print(f"{feat}: Selection Rate={sel_rate:.2%}, Mean Coef={mean_coef:.3f}, Std Coef={std_coef:.3f}")
    lasso_summary_df = pd.DataFrame(lasso_summary)

    # GETS Summary
    print("\n### GETS Feature Significance Summary")
    gets_summary = []
    for feat in feats_g:
        sig_rate = gets_signif_counts[feat] / total_runs if total_runs > 0 else 0
        gets_summary.append({"Feature": feat, "Significance Rate": sig_rate})
        print(f"{feat}: Significance Rate={sig_rate:.2%}")
    gets_summary_df = pd.DataFrame(gets_summary)

    # Ridge Summary
    print("\n### Ridge Coefficient Summary")
    ridge_summary = []
    for feat in feats_r:
        mean_coef = np.mean(ridge_coef_values[feat])
        std_coef = np.std(ridge_coef_values[feat])
        ridge_summary.append({"Feature": feat, "Mean Coef": mean_coef, "Std Coef": std_coef})
        print(f"{feat}: Mean Coef={mean_coef:.3f}, Std Coef={std_coef:.3f}")
    ridge_summary_df = pd.DataFrame(ridge_summary)

    # Visualizations
    plt.figure(figsize=(12, 6))
    plt.bar(lasso_summary_df["Feature"], lasso_summary_df["Selection Rate"])
    plt.xticks(rotation=45)
    plt.title("LASSO Feature Selection Rate")
    plt.xlabel("Features")
    plt.ylabel("Selection Rate")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.bar(gets_summary_df["Feature"], gets_summary_df["Significance Rate"])
    plt.xticks(rotation=45)
    plt.title("GETS Feature Significance Rate")
    plt.xlabel("Features")
    plt.ylabel("Significance Rate")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    ridge_data = [ridge_coef_values[feat] for feat in feats_r]
    plt.boxplot(ridge_data, labels=feats_r)
    plt.xticks(rotation=45)
    plt.title("Ridge Coefficient Distributions")
    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.tight_layout()
    plt.show()

    # Summary Table of Top Features
    top_lasso = lasso_summary_df.sort_values("Selection Rate", ascending=False).head(5)
    print("\n### Top 5 LASSO Features")
    print(top_lasso.to_string(index=False))
