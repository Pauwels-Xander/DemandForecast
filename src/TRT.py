from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import LambdaLR

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = "OrangeJuiceX25.csv"


def load_processed() -> pd.DataFrame:
    """
    Load and preprocess the OrangeJuiceX25 data using the user’s existing pipeline.
    This ensures price and lagged-sales columns are already standardized.
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
    Sliding-window dataset for training or testing.

    - df must already have price_*, deal, feat, sin_week, cos_week, sales_lag1/2/3, store_id, brand_id.
    - L is window length.
    - cutoff_week determines which windows go into this split:
         * If mode == "train", include only windows whose last index (t) <= cutoff_week.
         * If mode == "test",  include only windows whose last index (t) >  cutoff_week.
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


class SinusoidalPositionalEncoding(nn.Module):
    """
    Generates the classic sinusoidal positional encoding:
      PE[pos, 2i]   = sin(pos / (10000^(2i/d_model)))
      PE[pos, 2i+1] = cos(pos / (10000^(2i/d_model)))
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model))
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, seq_len, d_model)
        returns x + pe[:, :seq_len, :]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SmallTransformerV2(nn.Module):
    """
    Enhanced Transformer model with:
      - 1D convolution front-end
      - Sinusoidal positional encoding
      - Static MLP for store/brand context
      - LayerNorm before the encoder
      - Dropout + Xavier initialization
    """

    def __init__(
        self,
        num_features: int,
        num_stores: int,
        num_brands: int,
        L: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.L = L

        # 1) Static embeddings → small MLP
        self.store_emb = nn.Embedding(num_stores, d_model)
        self.brand_emb = nn.Embedding(num_brands, d_model)
        self.static_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # 2) 1D convolutional “front end”
        #    Input: (batch, L, num_features) → transpose → (batch, num_features, L)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=d_model, kernel_size=3, padding=1)
        self.conv_relu = nn.ReLU()

        # 3) Project conv output → Transformer dimension (already d_model)
        #    After conv: (batch, d_model, L) → transpose → (batch, L, d_model)

        # 4) Sinusoidal positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=L)

        # 5) LayerNorm before feeding into Transformer
        self.pre_norm = nn.LayerNorm(d_model)

        # 6) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=d_model * 4,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 7) Final MLP that merges sequence output + static MLP output
        #    - Sequence output: take the last token’s vector (batch, d_model)
        #    - Static MLP: (batch, d_model)
        #    → Concatenate → (batch, 2*d_model) → MLP → scalar
        self.combiner = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Xavier init for all Linear layers and embeddings
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        past: torch.Tensor,       # (batch, L, num_features)
        store_id: torch.Tensor,   # (batch,)
        brand_id: torch.Tensor    # (batch,)
    ) -> torch.Tensor:
        batch_size = past.size(0)

        # 1) Static context
        store_vec = self.store_emb(store_id)  # (batch, d_model)
        brand_vec = self.brand_emb(brand_id)  # (batch, d_model)
        static_context = store_vec + brand_vec  # (batch, d_model)
        static_context = self.static_mlp(static_context)  # (batch, d_model)

        # 2) Convolution over time
        #    past: (batch, L, num_features) → (batch, num_features, L)
        x = past.transpose(1, 2)
        x = self.conv1(x)            # (batch, d_model, L)
        x = self.conv_relu(x)        # (batch, d_model, L)
        x = x.transpose(1, 2)        # → (batch, L, d_model)

        # 3) Positional encoding + LayerNorm
        x = self.pos_enc(x)          # (batch, L, d_model)
        x = self.pre_norm(x)         # (batch, L, d_model)

        # 4) Transformer Encoder
        x = self.encoder(x)          # (batch, L, d_model)

        # 5) Take last token's representation
        seq_out = x[:, -1, :]        # (batch, d_model)

        # 6) Concatenate with static context
        combined = torch.cat([seq_out, static_context], dim=-1)  # (batch, 2*d_model)

        # 7) Final MLP → scalar
        out = self.combiner(combined).squeeze(-1)  # (batch,)
        return out


def train_and_evaluate_transformer_v2(
    df: pd.DataFrame,
    L: int = 16,
    epochs: int = 50,
    holdout_weeks: int = 10,
    batch_size: int = 64,
    early_stopping_patience: int = 3,
    warmup_epochs: int = 5
) -> None:
    """
    1) Split data into:
       - train windows (week <= max_week - holdout_weeks)
       - test  windows (week >  max_week - holdout_weeks).
    2) Further split 'train' into (train ⇢ 90%) and (val ⇢ 10%) by index order (time-aware).
    3) Build and train on the training subset, with early stopping on validation RMSE.
    4) After early stopping, evaluate the best model on the held-out test set and report RMSE.
    """

    # 1a. Compute cutoff_week
    max_week = int(df["week"].max())
    cutoff_week = max_week - holdout_weeks

    # 1b. Build full train/test datasets
    full_train_ds = OJSequenceDataset(df, L=L, cutoff_week=cutoff_week, mode="train")
    test_ds       = OJSequenceDataset(df, L=L, cutoff_week=cutoff_week, mode="test")

    print(f"→ Total training windows before split: {len(full_train_ds)}")
    print(f"→ Total testing windows: {len(test_ds)}")

    # 2a. Time-aware 90/10 split:
    #      Keep first 90% of full_train_ds for training, last 10% for validation
    total_windows = len(full_train_ds)
    split_idx = int(total_windows * 0.9)

    train_ds, val_ds = random_split(
        full_train_ds,
        lengths=[split_idx, total_windows - split_idx],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   • Training subset size:   {len(train_ds)}")
    print(f"   • Validation subset size: {len(val_ds)}")
    print()

    # 2b. DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 3a. Instantiate model
    sample_past, _, _, _ = full_train_ds[0]
    num_features = sample_past.shape[1]
    num_stores = int(df["store_id"].nunique())
    num_brands = int(df["brand_id"].nunique())

    model = SmallTransformerV2(
        num_features=num_features,
        num_stores=num_stores,
        num_brands=num_brands,
        L=L,
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3b. Optimizer + scheduler with warmup
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-5
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-6)

    # Linear warmup for `warmup_epochs`, then cosine decay for (epochs - warmup_epochs)
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    criterion = nn.MSELoss()

    # 3c. Early stopping variables
    best_val_rmse = float("inf")
    best_state_dict = None
    patience_counter = 0

    # 4) Training loop with validation each epoch
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for past, store_id, brand_id, target in train_loader:
            past, store_id, brand_id, target = [
                t.to(device, non_blocking=True) for t in (past, store_id, brand_id, target)
            ]
            optimizer.zero_grad()

            # Forward & backward
            pred = model(past, store_id, brand_id)           # (batch,)
            loss = criterion(pred, target)                   # MSE
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * past.size(0)

        scheduler.step()
        train_rmse = math.sqrt(running_loss / len(train_ds))
        print(f"Epoch {epoch+1}/{epochs} → Train RMSE: {train_rmse:.4f}", end="; ")

        # 4a. Validate
        model.eval()
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for past, store_id, brand_id, target in val_loader:
                past, store_id, brand_id = [
                    t.to(device, non_blocking=True) for t in (past, store_id, brand_id)
                ]
                out = model(past, store_id, brand_id)  # (batch,)
                val_preds.append(out.cpu().numpy())
                val_trues.append(target.numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_trues = np.concatenate(val_trues, axis=0)
        val_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
        print(f"Validation RMSE: {val_rmse:.4f}")

        # 4b. Early stopping check
        if val_rmse < best_val_rmse - 1e-5:
            best_val_rmse = val_rmse
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"→ Early stopping at epoch {epoch+1}. Best val RMSE = {best_val_rmse:.4f}")
                break

    # 5) Load best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 6) Final evaluation on test set
    model.eval()
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for past, store_id, brand_id, target in test_loader:
            past, store_id, brand_id = [
                t.to(device, non_blocking=True) for t in (past, store_id, brand_id)
            ]
            out = model(past, store_id, brand_id)  # (batch,)
            test_preds.append(out.cpu().numpy())
            test_trues.append(target.numpy())

    test_preds = np.concatenate(test_preds, axis=0)
    test_trues = np.concatenate(test_trues, axis=0)
    test_rmse = np.sqrt(mean_squared_error(test_trues, test_preds))
    print(f"\n→ Final Test RMSE (out-of-sample): {test_rmse:.2f}")


def main() -> None:
    df = load_processed()
    # Hold out the last 10 weeks of each series for testing
    train_and_evaluate_transformer_v2(
        df,
        L=16,
        epochs=50,
        holdout_weeks=10,
        batch_size=64,
        early_stopping_patience=3,
        warmup_epochs=5
    )


if __name__ == "__main__":
    main()
