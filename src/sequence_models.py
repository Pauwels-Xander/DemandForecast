"""Sequence model prototypes: LSTM and Seq2Seq for demand forecasting."""

from __future__ import annotations

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error

DATA_PATH = "data/OrangeJuiceX25.csv"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_processed() -> pd.DataFrame:
    """Load and preprocess the dataset using existing utilities."""
    from src.data_ingestion_preprocessing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)
    return df


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

class LSTMDataset(Dataset):
    """Past sales and price mean sequences for one store-brand pair."""

    def __init__(self, df: pd.DataFrame, store: int, brand: int, L: int = 3):
        subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
        subset = subset.sort_values("week")
        price_cols = [c for c in df.columns if c.startswith("price")]
        subset["price_mean"] = subset[price_cols].mean(axis=1)
        self.X: list[torch.Tensor] = []
        self.y: list[float] = []
        for t in range(L, len(subset)):
            past = subset.iloc[t - L : t][["sales", "price_mean"]].values.astype(
                np.float32
            )
            target = float(subset.iloc[t]["sales"])
            self.X.append(torch.tensor(past))
            self.y.append(target)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.y)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.X[idx], torch.tensor(self.y[idx], dtype=torch.float32)


class Seq2SeqDataset(Dataset):
    """Past sequence with future covariates for one-step forecasting."""

    def __init__(self, df: pd.DataFrame, store: int, brand: int, L: int = 3):
        subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
        subset = subset.sort_values("week")
        price_cols = [c for c in df.columns if c.startswith("price")]
        subset["price_mean"] = subset[price_cols].mean(axis=1)
        self.past: list[torch.Tensor] = []
        self.future: list[torch.Tensor] = []
        self.y: list[float] = []
        for t in range(L, len(subset)):
            past = subset.iloc[t - L : t][["sales", "price_mean"]].values.astype(
                np.float32
            )
            fut = subset.iloc[t][
                ["price_mean", "deal", "feat", "sin_week", "cos_week"]
            ].values.astype(np.float32)
            target = float(subset.iloc[t]["sales"])
            self.past.append(torch.tensor(past))
            self.future.append(torch.tensor(fut))
            self.y.append(target)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.y)

    def __getitem__(self, idx: int):  # type: ignore[override]
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
# Rolling evaluation functions
# ---------------------------------------------------------------------------

def lstm_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """Rolling-window LSTM forecasting for a single series."""

    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week")
    price_cols = [c for c in subset.columns if c.startswith("price")]
    subset["price_mean"] = subset[price_cols].mean(axis=1)

    preds: list[float] = []
    actuals: list[float] = []

    for end in range(60, len(subset)):
        train = subset.iloc[:end]
        test = subset.iloc[end : end + 1]

        train_ds = LSTMDataset(train, store, brand, L=3)
        loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        model = LSTMSales()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()

        model.train()
        for _ in range(10):
            for x_batch, y_batch in loader:
                opt.zero_grad()
                pred = model(x_batch)
                loss = crit(pred, y_batch)
                loss.backward()
                opt.step()

        seq = train[["sales", "price_mean"]].iloc[-3:].values.astype(np.float32)
        seq_tensor = torch.tensor(seq).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            pred = model(seq_tensor).item()
        preds.append(pred)
        actuals.append(float(test["sales"].values[0]))

    rmse = mean_squared_error(actuals, preds, squared=False)
    return rmse


def seq2seq_per_series(df: pd.DataFrame, store: int, brand: int) -> float:
    """Seq2Seq model with future covariates for one series."""

    subset = df[(df["store"] == store) & (df["brand"] == brand)].copy()
    subset = subset.sort_values("week")
    price_cols = [c for c in subset.columns if c.startswith("price")]
    subset["price_mean"] = subset[price_cols].mean(axis=1)

    preds: list[float] = []
    actuals: list[float] = []

    for end in range(60, len(subset)):
        train = subset.iloc[:end]
        test = subset.iloc[end : end + 1]

        train_ds = Seq2SeqDataset(train, store, brand, L=3)
        loader = DataLoader(train_ds, batch_size=16, shuffle=True)

        model = Seq2SeqModel()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()

        model.train()
        for _ in range(10):
            for past, fut, target in loader:
                opt.zero_grad()
                pred = model(past, fut)
                loss = crit(pred, target)
                loss.backward()
                opt.step()

        past_seq = train[["sales", "price_mean"]].iloc[-3:].values.astype(np.float32)
        fut_cov = test[["price_mean", "deal", "feat", "sin_week", "cos_week"]].values.astype(
            np.float32
        )[0]
        past_tensor = torch.tensor(past_seq).unsqueeze(0)
        fut_tensor = torch.tensor(fut_cov).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            pred = model(past_tensor, fut_tensor).item()
        preds.append(pred)
        actuals.append(float(test["sales"].values[0]))

    rmse = mean_squared_error(actuals, preds, squared=False)
    return rmse


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_processed()
    store, brand = 1, 1
    print("LSTM per-series RMSE:")
    rmse_lstm = lstm_per_series(df, store, brand)
    print(rmse_lstm)

    print("\nSeq2Seq per-series RMSE:")
    rmse_seq = seq2seq_per_series(df, store, brand)
    print(rmse_seq)


if __name__ == "__main__":
    main()
