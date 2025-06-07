"""Sequence model prototypes: LSTM and Seq2Seq """

from __future__ import annotations

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # For progress bar

DATA_PATH = "OrangeJuiceX25.csv"
WINDOW_SIZE = 52  # Use exactly 52 weeks of history each time


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_processed() -> pd.DataFrame:
    """Load and preprocess the dataset using existing utilities."""
    from Processing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)
    return df


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

class LSTMDataset(Dataset):
    """Past-3-week sequences of [sales, price_mean] for one store-brand pair, built from a 'train' slice."""

    def __init__(self, df: pd.DataFrame, L: int = 3):
        """
        df: already-filtered DataFrame containing only one (store, brand), sorted by week.
        L: number of lookback weeks (3 by default).
        """
        subset = df.copy()
        price_cols = [c for c in subset.columns if c.startswith("price")]
        subset["price_mean"] = subset[price_cols].mean(axis=1)

        self.X: list[torch.Tensor] = []
        self.y: list[float] = []

        # Build (past L → target) pairs within this slice
        for t in range(L, len(subset)):
            past = subset.iloc[t - L : t][["sales", "price_mean"]].values.astype(np.float32)
            target = float(subset.iloc[t]["sales"])
            self.X.append(torch.tensor(past, dtype=torch.float32))
            self.y.append(target)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], torch.tensor(self.y[idx], dtype=torch.float32)


class Seq2SeqDataset(Dataset):
    """Past-3-week sequences plus one-step-ahead covariates for one store-brand pair."""

    def __init__(self, df: pd.DataFrame, L: int = 3):
        """
        df: already-filtered DataFrame containing only one (store, brand), sorted by week.
        L: number of lookback weeks (3 by default).
        Future covariates: [price_mean, deal, feat, sin_week, cos_week].
        """
        subset = df.copy()
        price_cols = [c for c in subset.columns if c.startswith("price")]
        subset["price_mean"] = subset[price_cols].mean(axis=1)

        self.past: list[torch.Tensor] = []
        self.future: list[torch.Tensor] = []
        self.y: list[float] = []

        for t in range(L, len(subset)):
            past = subset.iloc[t - L : t][["sales", "price_mean"]].values.astype(np.float32)
            fut = subset.iloc[t][["price_mean", "deal", "feat", "sin_week", "cos_week"]].values.astype(np.float32)
            target = float(subset.iloc[t]["sales"])
            self.past.append(torch.tensor(past, dtype=torch.float32))
            self.future.append(torch.tensor(fut, dtype=torch.float32))
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
    def __init__(self, input_dim: int = 2, hidden: int = 32, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class Seq2SeqModel(nn.Module):
    def __init__(self, past_dim: int = 2, fut_dim: int = 5, hidden: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(past_dim, hidden, batch_first=True)
        self.decoder = nn.LSTM(fut_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, past: torch.Tensor, fut: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(past)
        out, _ = self.decoder(fut.unsqueeze(1), (h, c))
        return self.fc(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# Rolling evaluation functions (adjusted for a 52-week fixed window)
# ---------------------------------------------------------------------------

def lstm_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """Rolling-window LSTM forecasting for one store-brand pair using exactly 52 weeks each iteration."""
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week").reset_index(drop=True)

    # Precompute price_mean
    price_cols = [c for c in subset.columns if c.startswith("price")]
    subset["price_mean"] = subset[price_cols].mean(axis=1)

    preds: list[float] = []
    actuals: list[float] = []

    for end in range(WINDOW_SIZE, len(subset)):
        start = end - WINDOW_SIZE
        train = subset.iloc[start:end].reset_index(drop=True)  # exactly 52 rows
        test = subset.iloc[end : end + 1]

        # Build DataLoader on this 52-row training slice
        train_ds = LSTMDataset(train, L=3)
        loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        model = LSTMSales()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()

        # Train for a few epochs on this slice
        model.train()
        for _ in range(10):
            for x_batch, y_batch in loader:
                opt.zero_grad()
                pred = model(x_batch)
                loss = crit(pred, y_batch)
                loss.backward()
                opt.step()

        # Prepare the last-3-week LSTM input from the final part of `train`
        seq = train[["sales", "price_mean"]].iloc[-3:].values.astype(np.float32)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            pred = model(seq_tensor).item()

        preds.append(pred)
        actuals.append(float(test["sales"].values[0]))

    mse = mean_squared_error(actuals, preds) if preds else float("nan")
    return float(np.sqrt(mse)) if preds else float("nan")


def seq2seq_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """Rolling-window Seq2Seq forecasting for one store-brand pair using exactly 52 weeks each iteration."""
    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week").reset_index(drop=True)

    # Precompute price_mean
    price_cols = [c for c in subset.columns if c.startswith("price")]
    subset["price_mean"] = subset[price_cols].mean(axis=1)

    preds: list[float] = []
    actuals: list[float] = []

    for end in range(WINDOW_SIZE, len(subset)):
        start = end - WINDOW_SIZE
        train = subset.iloc[start:end].reset_index(drop=True)  # exactly 52 rows
        test = subset.iloc[end : end + 1]

        # Build DataLoader on this 52-row training slice
        train_ds = Seq2SeqDataset(train, L=3)
        loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        model = Seq2SeqModel()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()

        # Train for a few epochs on this slice
        model.train()
        for _ in range(10):
            for past, fut, target in loader:
                opt.zero_grad()
                pred = model(past, fut)
                loss = crit(pred, target)
                loss.backward()
                opt.step()

        # Prepare the “last 3 weeks” + “future covariates” from `train`
        past_seq = train[["sales", "price_mean"]].iloc[-3:].values.astype(np.float32)
        fut_cov = test[["price_mean", "deal", "feat", "sin_week", "cos_week"]].values.astype(np.float32)[0]

        past_tensor = torch.tensor(past_seq, dtype=torch.float32).unsqueeze(0)
        fut_tensor = torch.tensor(fut_cov, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            pred = model(past_tensor, fut_tensor).item()

        preds.append(pred)
        actuals.append(float(test["sales"].values[0]))

    mse = mean_squared_error(actuals, preds) if preds else float("nan")
    return float(np.sqrt(mse)) if preds else float("nan")


# ---------------------------------------------------------------------------
# Evaluate across all store-brand combinations with a progress bar
# ---------------------------------------------------------------------------

def evaluate_all_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling-window LSTM and Seq2Seq RMSEs for every (store, brand) pair that has
    at least WINDOW_SIZE + 1 observations. Returns a DataFrame with columns:
    [store, brand, rmse_lstm, rmse_seq2seq].
    """
    results = []

    stores = df["store"].unique()
    brands = df["brand"].unique()
    total_combinations = len(stores) * len(brands)

    # Use tqdm to show progress over all combinations
    for store_id, brand_id in tqdm(
        [(s, b) for s in stores for b in brands],
        total=total_combinations,
        desc="Evaluating store-brand combos",
        unit="combo",
    ):
        subset = df[(df["store"] == store_id) & (df["brand"] == brand_id)].copy()
        if len(subset) < WINDOW_SIZE + 1:
            # Skip series with insufficient data
            continue

        rmse_lstm = lstm_per_series(df, store_id, brand_id)
        rmse_seq = seq2seq_per_series(df, store_id, brand_id)

        results.append({
            "store": store_id,
            "brand": brand_id,
            "rmse_lstm": rmse_lstm,
            "rmse_seq2seq": rmse_seq,
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
