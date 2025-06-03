"""Lightweight Transformer model for demand forecasting.

This script implements a small sequence-to-one transformer designed for the
OrangeJuiceX25 dataset.  It loads the preprocessed data, builds sequences of
length ``L`` for each store-brand pair, and trains a modest transformer model to
predict next-week sales.  The architecture is intentionally small because the
dataset is limited.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA_PATH = "data/OrangeJuiceX25.csv"


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

@dataclass
class SequenceSample:
    past: np.ndarray  # shape (L, num_features)
    store: int
    brand: int
    target: float


class OJSequenceDataset(Dataset):
    """Create sliding windows of length ``L`` for all store-brand series."""

    def __init__(self, df: pd.DataFrame, L: int):
        self.samples: List[SequenceSample] = []
        price_cols = [c for c in df.columns if c.startswith("price")]
        lag_cols = [f"sales_lag{lag}" for lag in [1, 2, 3]]
        feat_cols = price_cols + ["deal", "feat", "sin_week", "cos_week"] + lag_cols

        for (store, brand), grp in df.groupby(["store", "brand"]):
            grp = grp.sort_values("week").reset_index(drop=True)
            for t in range(L, len(grp) - 1):
                past = grp.iloc[t - L : t][feat_cols].values.astype(np.float32)
                target = float(grp.iloc[t]["sales"])
                self.samples.append(SequenceSample(past, int(store), int(brand), target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        past = torch.from_numpy(sample.past)
        return past, torch.tensor(sample.store), torch.tensor(sample.brand), torch.tensor(sample.target)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class SmallTransformer(nn.Module):
    """A tiny Transformer encoder predicting next-week sales."""

    def __init__(self, num_features: int, num_stores: int, num_brands: int, d_model: int = 32, nhead: int = 2, num_layers: int = 2):
        super().__init__()
        self.store_emb = nn.Embedding(num_stores, d_model)
        self.brand_emb = nn.Embedding(num_brands, d_model)
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, past: torch.Tensor, store: torch.Tensor, brand: torch.Tensor) -> torch.Tensor:
        # past: (batch, L, num_features)
        x = self.input_proj(past)
        # Add store/brand embeddings to every time step
        store_vec = self.store_emb(store)
        brand_vec = self.brand_emb(brand)
        context = store_vec + brand_vec  # (batch, d_model)
        context = context.unsqueeze(1)
        x = x + context
        # Positional encoding (simple sin/cos)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).float()
        pe = torch.zeros(seq_len, x.size(2), device=x.device)
        pe[:, 0::2] = torch.sin(pos[:, None] / 10.0)
        pe[:, 1::2] = torch.cos(pos[:, None] / 10.0)
        x = x + pe.unsqueeze(0)
        out = self.encoder(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# Training routine (rolling windows)
# ---------------------------------------------------------------------------

def train_transformer(df: pd.DataFrame, L: int = 16, epochs: int = 20) -> None:
    df = df.sort_values(["store", "brand", "week"])
    dataset = OJSequenceDataset(df, L)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_features = dataset[0][0].shape[1]
    model = SmallTransformer(num_features, df["store"].nunique(), df["brand"].nunique())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for past, store, brand, target in loader:
            optimizer.zero_grad()
            pred = model(past, store, brand)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch} loss: {loss.item():.4f}")

    print("Training completed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_processed() -> pd.DataFrame:
    from src.data_ingestion_preprocessing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)
    return df


def main() -> None:
    df = load_processed()
    train_transformer(df, L=16, epochs=5)


if __name__ == "__main__":
    main()
