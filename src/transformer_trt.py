from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = "data\OrangeJuiceX25.csv"


def load_processed() -> pd.DataFrame:
    """
    Load and preprocess the OrangeJuiceX25 data using the user’s existing pipeline.
    This ensures price and lagged‐sales columns are already standardized.
    """
    from Processing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)

    # Ensure store_id and brand_id codes exist (preprocess_features already does this).
    if "store_id" not in df.columns:
        df["store_id"] = df["store"].astype("category").cat.codes
    if "brand_id" not in df.columns:
        df["brand_id"] = df["brand"].astype("category").cat.codes

    return df


@dataclass
class SequenceSample:
    past: np.ndarray       # shape (L, num_features); already scaled
    store_id: int          # 0-based
    brand_id: int          # 0-based
    target: float


class OJSequenceDataset(Dataset):
    """
    Sliding‐window dataset for training or testing.

    - df must already have price_*, deal, feat, sin_week, cos_week, sales_lag1/2/3, store_id, brand_id.
    - L is window length.
    - cutoff_week determines which windows go into this split:
         * If mode == "train", include only windows whose last index (t) <= cutoff_week.
         * If mode == "test", include only windows whose last index (t) > cutoff_week.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        L: int,
        cutoff_week: int,
        mode: str = "train"
    ):
        assert mode in ("train", "test")
        self.samples: List[SequenceSample] = []
        price_cols = [c for c in df.columns if c.startswith("price")]
        lag_cols = [f"sales_lag{lag}" for lag in [1, 2, 3]]
        feat_cols = price_cols + ["deal", "feat", "sin_week", "cos_week"] + lag_cols

        # Group by (store, brand), sort by week
        for (_, _), grp in df.groupby(["store", "brand"]):
            grp = grp.sort_values("week").reset_index(drop=True)
            sid = int(grp["store_id"].iloc[0])
            bid = int(grp["brand_id"].iloc[0])

            # For each t in [L .. len(grp)-2], last index = t
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
        past_tensor = torch.from_numpy(sample.past)                          # (L, num_features)
        store_tensor = torch.tensor(sample.store_id, dtype=torch.long)       # ( )
        brand_tensor = torch.tensor(sample.brand_id, dtype=torch.long)       # ( )
        target_tensor = torch.tensor(sample.target, dtype=torch.float32)     # ( )
        return past_tensor, store_tensor, brand_tensor, target_tensor


class SmallTransformer(nn.Module):
    """Tiny Transformer encoder to predict next‐week sales, with learnable positional embeddings."""

    def __init__(
        self,
        num_features: int,
        num_stores: int,
        num_brands: int,
        L: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.store_emb = nn.Embedding(num_stores, d_model)
        self.brand_emb = nn.Embedding(num_brands, d_model)

        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=d_model * 4
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Embedding(L, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(
        self,
        past: torch.Tensor,       # (batch, L, num_features)
        store_id: torch.Tensor,   # (batch,)
        brand_id: torch.Tensor    # (batch,)
    ) -> torch.Tensor:
        batch_size, seq_len, _ = past.size()
        x = self.input_proj(past)  # → (batch, L, d_model)

        store_vec = self.store_emb(store_id)  # (batch, d_model)
        brand_vec = self.brand_emb(brand_id)  # (batch, d_model)
        context = (store_vec + brand_vec).unsqueeze(1)  # (batch, 1, d_model)
        x = x + context

        # Add learnable positional embeddings
        positions = torch.arange(seq_len, device=past.device).unsqueeze(0).repeat(batch_size, 1)  # (batch, L)
        x = x + self.pos_emb(positions)  # (batch, L, d_model)

        x = self.encoder(x)             # (batch, L, d_model)
        x = x[:, -1, :]                 # (batch, d_model)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)   # (batch,)


def train_and_evaluate_transformer(
    df: pd.DataFrame,
    L: int = 16,
    epochs: int = 20,
    holdout_weeks: int = 10,
    batch_size: int = 32
) -> None:
    """
    1) Split data into train windows (week <= max_week - holdout_weeks)
       and test windows (week >  max_week - holdout_weeks).
    2) Build and train on the train dataset.
    3) After training, run on the test dataset and compute RMSE.
    """

    # 1a. Compute cutoff_week = max_week - holdout_weeks
    max_week = int(df["week"].max())
    cutoff_week = max_week - holdout_weeks

    # 1b. Build train/test datasets (features already standardized upstream)
    train_dataset = OJSequenceDataset(df, L=L, cutoff_week=cutoff_week, mode="train")
    test_dataset  = OJSequenceDataset(df, L=L, cutoff_week=cutoff_week, mode="test")

    print(f"Number of training windows: {len(train_dataset)}")
    print(f"Number of  testing windows:  {len(test_dataset)}")

    # 1c. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader  = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 2. Instantiate model
    sample_past, _, _, _ = train_dataset[0]
    num_features = sample_past.shape[1]
    num_stores = int(df["store_id"].nunique())
    num_brands = int(df["brand_id"].nunique())

    model = SmallTransformer(
        num_features=num_features,
        num_stores=num_stores,
        num_brands=num_brands,
        L=L,
        d_model=64,
        nhead=4,
        num_layers=3,
        dropout=0.1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Set up optimizer, scheduler, and mixed‐precision
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "emb" not in n], "weight_decay": 1e-5},
            {"params": [p for n, p in model.named_parameters() if "emb" in n], "weight_decay": 0.0},
        ],
        lr=1e-3
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler_amp = torch.cuda.amp.GradScaler()
    criterion = nn.MSELoss()

    # 3a. Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for past, store_id, brand_id, target in train_loader:
            past, store_id, brand_id, target = [
                t.to(device, non_blocking=True) for t in (past, store_id, brand_id, target)
            ]
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                pred = model(past, store_id, brand_id)  # (batch,)
                loss = criterion(pred, target)          # MSE

            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            running_loss += loss.item() * past.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs}, train loss: {epoch_loss:.4f}")

    print("Training completed.\n")

    # 4. Evaluate on test set
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for past, store_id, brand_id, target in test_loader:
            past, store_id, brand_id = [
                t.to(device, non_blocking=True) for t in (past, store_id, brand_id)
            ]
            out = model(past, store_id, brand_id)  # (batch,)
            all_preds.append(out.cpu().numpy())
            all_targets.append(target.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    rmse_test = np.sqrt(mean_squared_error(all_targets, all_preds))
    print(f"Test RMSE (out‐of‐sample): {rmse_test:.2f}")


def main() -> None:
    df = load_processed()
    # Hold out the last 10 weeks of each series for testing
    train_and_evaluate_transformer(df, L=16, epochs=20, holdout_weeks=10, batch_size=32)


if __name__ == "__main__":
    main()
