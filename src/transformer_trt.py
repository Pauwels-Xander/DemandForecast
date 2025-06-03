"""Lightweight Transformer model for demand forecasting, now with test‐time RMSE.

We:
  1. Preprocess entire DataFrame as before (sales_lag, price, sin_week, cos_week, store_id, brand_id, etc.).
  2. Choose a cutoff = (max_week − 10). All windows ending at t ≤ cutoff are used for training.
  3. Build training dataset and train Transformer.
  4. Build a separate “test” dataset of windows whose last index t is in (cutoff+1 … max_week−1).
  5. Run model on test windows, collect (pred, actual), compute RMSE.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb  # (if you still need it elsewhere)
from sklearn.metrics import mean_squared_error

DATA_PATH = "data\OrangeJuiceX25.csv"

# ---------------------------------------------------------------------------
# Data utilities (almost identical to before, but accepts an explicit cutoff)
# ---------------------------------------------------------------------------

@dataclass
class SequenceSample:
    past: np.ndarray       # shape (L, num_features)
    store_id: int          # 0-based
    brand_id: int          # 0-based
    target: float


class OJSequenceDataset(Dataset):
    """
    Sliding‐window dataset for training or testing.

    - `df` must already have price_*, deal, feat, sin_week, cos_week, sales_lag1/2/3,
      store_id, brand_id.
    - `L` is window length.
    - `cutoff_week` determines which windows go into this split:
         * If mode=="train", we include only windows whose last index (t) ≤ cutoff_week.
         * If mode=="test", we include only windows whose last index (t) > cutoff_week.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        L: int,
        cutoff_week: int,
        mode: str = "train"  # either "train" or "test"
    ):
        assert mode in ("train", "test")
        self.samples: List[SequenceSample] = []
        price_cols = [c for c in df.columns if c.startswith("price")]
        lag_cols   = [f"sales_lag{lag}" for lag in [1, 2, 3]]
        feat_cols  = price_cols + ["deal", "feat", "sin_week", "cos_week"] + lag_cols

        # Group by (store, brand), sort by week
        for (_, _), grp in df.groupby(["store", "brand"]):
            grp = grp.sort_values("week").reset_index(drop=True)
            sid = int(grp["store_id"].iloc[0])
            bid = int(grp["brand_id"].iloc[0])

            # For each t in [L .. len(grp)-2], last index = t
            # If mode=="train": only keep t ≤ cutoff_week
            # If mode=="test": only keep t > cutoff_week
            for t in range(L, len(grp) - 1):
                week_t = int(grp.iloc[t]["week"])
                if mode == "train" and week_t > cutoff_week:
                    continue
                if mode == "test" and week_t <= cutoff_week:
                    continue

                past = grp.iloc[t - L : t][feat_cols].values.astype(np.float32)
                if np.isnan(past).any():
                    continue  # skip windows containing NaNs

                target = float(grp.iloc[t]["sales"])
                self.samples.append(SequenceSample(past, sid, bid, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        past_tensor = torch.from_numpy(sample.past)                          # float32 (L, num_features)
        store_tensor = torch.tensor(sample.store_id, dtype=torch.long)       # ( )
        brand_tensor = torch.tensor(sample.brand_id, dtype=torch.long)       # ( )
        target_tensor = torch.tensor(sample.target, dtype=torch.float32)     # ( )
        return past_tensor, store_tensor, brand_tensor, target_tensor


# ---------------------------------------------------------------------------
# Model definition (unchanged)
# ---------------------------------------------------------------------------

class SmallTransformer(nn.Module):
    """Tiny Transformer encoder to predict next‐week sales."""

    def __init__(
        self,
        num_features: int,
        num_stores: int,
        num_brands: int,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 2
    ):
        super().__init__()
        self.store_emb = nn.Embedding(num_stores, d_model)
        self.brand_emb = nn.Embedding(num_brands, d_model)

        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(
        self,
        past: torch.Tensor,     # (batch, L, num_features)
        store_id: torch.Tensor,  # (batch,)
        brand_id: torch.Tensor   # (batch,)
    ) -> torch.Tensor:
        x = self.input_proj(past)  # → (batch, L, d_model)

        store_vec = self.store_emb(store_id)  # (batch, d_model)
        brand_vec = self.brand_emb(brand_id)  # (batch, d_model)
        context = store_vec + brand_vec       # (batch, d_model)
        context = context.unsqueeze(1)        # (batch, 1, d_model)
        x = x + context                       # broadcast to (batch, L, d_model)

        # Simple sinusoidal positional encoding
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).float()       # (L,)
        pe = torch.zeros(seq_len, x.size(2), device=x.device)      # (L, d_model)
        pe[:, 0::2] = torch.sin(pos.unsqueeze(1) / 10.0)
        pe[:, 1::2] = torch.cos(pos.unsqueeze(1) / 10.0)
        x = x + pe.unsqueeze(0)  # (batch, L, d_model)

        out = self.encoder(x)    # (batch, L, d_model)
        return self.fc(out[:, -1, :]).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# Training + Testing routine
# ---------------------------------------------------------------------------

def train_and_evaluate_transformer(
    df: pd.DataFrame,
    L: int = 16,
    epochs: int = 20,
    holdout_weeks: int = 10
) -> None:
    """
    1) Split data into `train` windows (t ≤ max_week - holdout_weeks)
       and `test` windows (t >  max_week - holdout_weeks).
    2) Build and train on the “train” dataset.
    3) After training, run on the “test” dataset and compute RMSE.
    """

    # 1a. Compute cutoff_week = max_week - holdout_weeks
    max_week = int(df["week"].max())
    cutoff_week = max_week - holdout_weeks

    # 1b. Build train/test datasets
    train_dataset = OJSequenceDataset(df, L=L, cutoff_week=cutoff_week, mode="train")
    test_dataset  = OJSequenceDataset(df, L=L, cutoff_week=cutoff_week, mode="test")

    print(f"Number of training windows: {len(train_dataset)}")
    print(f"Number of  testing windows:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)

    # 2. Instantiate model
    # Infer num_features from one sample
    sample_past, _, _, _ = train_dataset[0]
    num_features = sample_past.shape[1]
    num_stores = int(df["store_id"].nunique())
    num_brands = int(df["brand_id"].nunique())

    model = SmallTransformer(
        num_features=num_features,
        num_stores=num_stores,
        num_brands=num_brands,
        d_model=32,
        nhead=2,
        num_layers=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # 2a. Train
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for past, store_id, brand_id, target in train_loader:
            past = past.to(device)            # (batch, L, num_features)
            store_id = store_id.to(device)    # (batch,)
            brand_id = brand_id.to(device)    # (batch,)
            target = target.to(device)        # (batch,)

            optimizer.zero_grad()
            pred = model(past, store_id, brand_id)  # (batch,)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * past.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs}, train loss: {epoch_loss:.4f}")

    print("Training completed.\n")

    # 3. Evaluate on test set
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for past, store_id, brand_id, target in test_loader:
            past = past.to(device)
            store_id = store_id.to(device)
            brand_id = brand_id.to(device)
            out = model(past, store_id, brand_id)  # (batch,)
            all_preds.append(out.cpu().numpy())
            all_targets.append(target.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    rmse_test = np.sqrt(mean_squared_error(all_targets, all_preds))
    print(f"Test RMSE (out‐of‐sample): {rmse_test:.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_processed() -> pd.DataFrame:
    """
    Load and preprocess the OrangeJuiceX25 data. Ensure store_id/brand_id are present.
    """
    from Processing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)

    # If preprocess_features didn’t add store_id/brand_id, add now:
    if "store_id" not in df.columns:
        df["store_id"] = df["store"].astype("category").cat.codes
    if "brand_id" not in df.columns:
        df["brand_id"] = df["brand"].astype("category").cat.codes

    return df


def main() -> None:
    df = load_processed()
    # Here we hold out the last 10 weeks of each series for testing:
    train_and_evaluate_transformer(df, L=16, epochs=5, holdout_weeks=10)


if __name__ == "__main__":
    main()
