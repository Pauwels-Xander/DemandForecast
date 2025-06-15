from __future__ import annotations

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # For progress bar

DATA_PATH = "OrangeJuiceX25.csv"
WINDOW_SIZE = 52
LSTM_LOOKBACK = 3  # how many past weeks to feed into the models

# ---------------------------------------------------------------------------
# 1) List of all regressors used
FEATURE_COLS = [
    "store", "brand", "week",
    "OwnLogPrice",
    "LagLogPrice_2",
    "MinCompPrice",
    "CategoryFeatShare",
    "CrossFeatPressure",
    "LagLogSales",
    "DealFlag_L1",
]
def load_processed() -> pd.DataFrame:
    """Load and preprocess the dataset using existing utilities."""
    from ProcessingOutliers import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)
    return df


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

class LSTMDataset(Dataset):
    """Past-LSTM_LOOKBACK-week sequences of FEATURE_COLS (minus store,brand,sales)."""
    def __init__(self, df: pd.DataFrame, L: int = LSTM_LOOKBACK):
        subset = df.copy().reset_index(drop=True)
        self.feature_cols = [c for c in FEATURE_COLS if c not in ("store", "brand", "sales")]
        self.X: list[torch.Tensor] = []
        self.y: list[float] = []

        for t in range(L, len(subset)):
            past_vals = subset.iloc[t-L:t][self.feature_cols].values.astype(np.float32)
            target    = float(subset.iloc[t]["sales"])
            self.X.append(torch.tensor(past_vals, dtype=torch.float32))
            self.y.append(target)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], torch.tensor(self.y[idx], dtype=torch.float32)


class Seq2SeqDataset(Dataset):
    """Past-LSTM_LOOKBACK-week sequences + 1-step-ahead covariates of same FEATURE_COLS."""
    def __init__(self, df: pd.DataFrame, L: int = LSTM_LOOKBACK):
        subset = df.copy().reset_index(drop=True)
        self.feature_cols = [c for c in FEATURE_COLS if c not in ("store", "brand", "sales")]
        self.past:   list[torch.Tensor] = []
        self.future: list[torch.Tensor] = []
        self.y:      list[float]        = []

        for t in range(L, len(subset)):
            past_seq = subset.iloc[t-L:t][self.feature_cols].values.astype(np.float32)
            fut_cov  = subset.iloc[t][self.feature_cols].values.astype(np.float32)
            target   = float(subset.iloc[t]["sales"])

            self.past.append( torch.tensor(past_seq, dtype=torch.float32) )
            self.future.append( torch.tensor(fut_cov, dtype=torch.float32) )
            self.y.append(target)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            self.past[idx],
            self.future[idx],
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LSTMSales(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class Seq2SeqModel(nn.Module):
    def __init__(self, past_dim: int, fut_dim: int, hidden: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(past_dim, hidden, batch_first=True)
        self.decoder = nn.LSTM(fut_dim, hidden, batch_first=True)
        self.fc      = nn.Linear(hidden, 1)

    def forward(self, past: torch.Tensor, fut: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(past)
        out, _   = self.decoder(fut.unsqueeze(1), (h, c))
        return self.fc(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# Rolling evaluation (with NaN‐drop and mask)
# ---------------------------------------------------------------------------

def lstm_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """Rolling‐window LSTM; drops any rows with missing data & filters NaNs in the end."""
    subset = (
        df[(df["store"] == store) & (df["brand"] == brand)]
        .sort_values("week")
        .reset_index(drop=True)
    )
    # 1) drop any row missing a regressor or the target
    subset = subset.dropna(subset=FEATURE_COLS + ["sales"])
    preds, actuals = [], []

    for end in range(WINDOW_SIZE, len(subset)):
        train = subset.iloc[end-WINDOW_SIZE:end].reset_index(drop=True)
        test  = subset.iloc[end : end+1]

        ds     = LSTMDataset(train)
        loader = DataLoader(ds, batch_size=16, shuffle=True)

        model = LSTMSales(input_dim=len(ds.feature_cols))
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit  = nn.MSELoss()

        model.train()
        for _ in range(10):
            for x_b, y_b in loader:
                opt.zero_grad()
                loss = crit(model(x_b), y_b)
                loss.backward()
                opt.step()

        seq = train[ds.feature_cols].iloc[-LSTM_LOOKBACK:].values.astype(np.float32)
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            preds.append(model(seq_t).item())
            actuals.append(float(test["sales"].values[0]))

    if not preds:
        return float("nan")

    # 2) mask out any NaNs before computing RMSE
    y_pred = np.array(preds, dtype=float)
    y_true = np.array(actuals, dtype=float)
    mask   = np.isfinite(y_pred) & np.isfinite(y_true)
    if not mask.any():
        return float("nan")

    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


def seq2seq_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """Rolling‐window Seq2Seq; drops missing rows & filters NaNs before RMSE."""
    subset = (
        df[(df["store"] == store) & (df["brand"] == brand)]
        .sort_values("week")
        .reset_index(drop=True)
    )
    subset = subset.dropna(subset=FEATURE_COLS + ["sales"])
    preds, actuals = [], []

    for end in range(WINDOW_SIZE, len(subset)):
        train = subset.iloc[end-WINDOW_SIZE:end].reset_index(drop=True)
        test  = subset.iloc[end : end+1]

        ds     = Seq2SeqDataset(train)
        loader = DataLoader(ds, batch_size=16, shuffle=True)

        model = Seq2SeqModel(past_dim=len(ds.feature_cols), fut_dim=len(ds.feature_cols))
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit  = nn.MSELoss()

        model.train()
        for _ in range(10):
            for past_b, fut_b, y_b in loader:
                opt.zero_grad()
                loss = crit(model(past_b, fut_b), y_b)
                loss.backward()
                opt.step()

        past_seq = train[ds.feature_cols].iloc[-LSTM_LOOKBACK:].values.astype(np.float32)
        fut_cov  = test[ds.feature_cols].iloc[0].values.astype(np.float32)
        past_t   = torch.tensor(past_seq, dtype=torch.float32).unsqueeze(0)
        fut_t    = torch.tensor(fut_cov, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            preds.append(model(past_t, fut_t).item())
            actuals.append(float(test["sales"].values[0]))

    if not preds:
        return float("nan")

    y_pred = np.array(preds, dtype=float)
    y_true = np.array(actuals, dtype=float)
    mask   = np.isfinite(y_pred) & np.isfinite(y_true)
    if not mask.any():
        return float("nan")

    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


# ---------------------------------------------------------------------------
# Evaluate across all store-brand combinations
# ---------------------------------------------------------------------------

def evaluate_all_models(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    stores = df["store"].unique()
    brands = df["brand"].unique()

    for store_id, brand_id in tqdm(
        [(s, b) for s in stores for b in brands],
        total=len(stores) * len(brands),
        desc="Evaluating combos",
        unit="combo"
    ):
        subset = df[(df["store"] == store_id) & (df["brand"] == brand_id)]
        if len(subset) < WINDOW_SIZE + 1:
            continue

        rl = lstm_per_series(df, store_id, brand_id)
        rs = seq2seq_per_series(df, store_id, brand_id)

        results.append({
            "store":        store_id,
            "brand":        brand_id,
            "rmse_lstm":    rl,
            "rmse_seq2seq": rs,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_processed()
    all_results = evaluate_all_models(df)
    print("Rolling-window RMSE for all store-brand combinations:")
    print(all_results.to_string(index=False))


if __name__ == "__main__":
    main()
