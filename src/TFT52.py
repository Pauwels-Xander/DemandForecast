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
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm  # for the progress bar

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = "OrangeJuiceX25.csv"
WINDOW_SIZE = 52     # must match the L you use for the Transformer
EPOCHS = 20          # number of epochs per retrain
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 3
WARMUP_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_processed() -> pd.DataFrame:
    """
    Load and preprocess the OrangeJuiceX25 data using the user’s existing pipeline.
    Assumes “Processing.load_data” and “Processing.preprocess_features” exist.
    """
    from Processing import load_data, preprocess_features

    df = load_data(DATA_PATH)
    df = preprocess_features(df)

    # If store_id / brand_id columns are missing, create them from the raw “store” / “brand”:
    if "store_id" not in df.columns:
        df["store_id"] = df["store"].astype("category").cat.codes
    if "brand_id" not in df.columns:
        df["brand_id"] = df["brand"].astype("category").cat.codes

    return df


@dataclass
class SequenceSample:
    past: np.ndarray       # shape = (L, num_features)
    store_id: int          # 0-based
    brand_id: int          # 0-based
    target: float          # “sales” at the target week


class OJSequenceDataset(Dataset):
    """
    Sliding‐window dataset builder. Supports three modes:

      • mode="train":   include all windows whose target‐week_idx ≤ cutoff_week_idx.
      • mode="test_one": include all windows whose target‐week_idx == cutoff_week_idx+1.
      • mode="test_all": include all windows whose target‐week_idx > cutoff_week_idx.

    Each sample uses exactly WINDOW_SIZE = L past weeks of features to predict “sales” at week_idx_t.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        L: int,
        cutoff_week_idx: int,
        mode: str = "test_one"
    ):
        assert mode in ("train", "test_one", "test_all")
        self.samples: List[SequenceSample] = []

        # Gather all columns that start with "price", plus static features and lag columns.
        price_cols = [c for c in df.columns if c.startswith("price")]
        lag_cols = [f"sales_lag{lag}" for lag in (1, 2, 3)]
        feat_cols = price_cols + ["deal", "feat", "sin_week", "cos_week"] + lag_cols

        # Group by (store, brand); sort each group by “week_idx”
        for (_, _), grp in df.groupby(["store", "brand"]):
            grp = grp.sort_values("week_idx").reset_index(drop=True)
            sid = int(grp["store_id"].iloc[0])
            bid = int(grp["brand_id"].iloc[0])

            # Consider every index t = L .. len(grp)-1 (so that t = len(grp)-1 is the final week_idx).
            for t in range(L, len(grp)):
                week_idx_t = int(grp.iloc[t]["week_idx"])

                # TRAIN mode: keep if week_idx_t ≤ cutoff_week_idx
                if mode == "train" and week_idx_t > cutoff_week_idx:
                    continue

                # TEST_ONE mode: keep only windows where week_idx_t == cutoff_week_idx + 1
                if mode == "test_one" and week_idx_t != (cutoff_week_idx + 1):
                    continue

                # TEST_ALL mode: keep every window with week_idx_t > cutoff_week_idx
                if mode == "test_all" and week_idx_t <= cutoff_week_idx:
                    continue

                # Build the “past” feature array: weeks [t-L .. t-1]
                past = grp.iloc[t - L : t][feat_cols].values.astype(np.float32)
                if np.isnan(past).any():
                    # Skip windows containing any NaN in their 52-week block
                    continue

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
    Standard sinusoidal positional encoding (Vaswani et al., “Attention is All You Need”).
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model))
        )                                                   # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)       # even dims
        pe[:, 1::2] = torch.cos(position * div_term)       # odd dims
        self.register_buffer("pe", pe.unsqueeze(0))        # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (batch, seq_len, d_model)
        Returns:
          x + positional encoding (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]  # broadcast over batch


class SmallTransformerV2(nn.Module):
    """
    The same Transformer architecture you posted:
      - Convolutional “front end” to embed features
      - Sinusoidal positional encoding
      - TransformerEncoder
      - Static store/brand embedding MLP
      - Final MLP that merges the last‐token encoding + static MLP → scalar
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

        # 1) Static embeddings for store + brand → small MLP
        self.store_emb = nn.Embedding(num_stores, d_model)
        self.brand_emb = nn.Embedding(num_brands, d_model)
        self.static_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # 2) 1D Convolutional “front end”
        #    Input: (batch, L, num_features) → transpose → (batch, num_features, L)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=d_model, kernel_size=3, padding=1)
        self.conv_relu = nn.ReLU()

        # 3) Positional encoding + pre‐LayerNorm
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=L)
        self.pre_norm = nn.LayerNorm(d_model)

        # 4) TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=d_model * 4,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5) Final MLP that merges “last‐token” output + static context
        #    → Linear(2*d_model → d_model) → ReLU → Dropout → Linear(d_model → 1)
        self.combiner = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for all Linear, Conv1d, and Embedding layers
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

        # 1) Static context embedding
        store_vec = self.store_emb(store_id)  # (batch, d_model)
        brand_vec = self.brand_emb(brand_id)  # (batch, d_model)
        static_context = store_vec + brand_vec  # (batch, d_model)
        static_context = self.static_mlp(static_context)  # (batch, d_model)

        # 2) Convolutional front‐end over time
        #    past: (batch, L, num_features) → (batch, num_features, L)
        x = past.transpose(1, 2)
        x = self.conv1(x)            # (batch, d_model, L)
        x = self.conv_relu(x)        # (batch, d_model, L)
        x = x.transpose(1, 2)        # (batch, L, d_model)

        # 3) Positional encoding + LayerNorm
        x = self.pos_enc(x)          # (batch, L, d_model)
        x = self.pre_norm(x)         # (batch, L, d_model)

        # 4) Transformer encoder (batch_first=True)
        x = self.encoder(x)          # (batch, L, d_model)

        # 5) “Take” the last token’s vector (i.e. time‐step L-1)
        seq_out = x[:, -1, :]        # (batch, d_model)

        # 6) Concatenate with static context (batch, 2*d_model)
        combined = torch.cat([seq_out, static_context], dim=-1)  # (batch, 2*d_model)

        # 7) Final MLP → scalar output
        out = self.combiner(combined).squeeze(-1)  # (batch,)
        return out


def train_one_transformer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    df: pd.DataFrame,
    L: int = WINDOW_SIZE,
    epochs: int = EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    warmup_epochs: int = WARMUP_EPOCHS
) -> SmallTransformerV2:
    """
    Train a single Transformer from scratch on train_loader, validating on val_loader.
    Returns the best‐checkpointed model (state dict loaded).
    """
    # 1) Instantiate a fresh model
    sample_past, _, _, _ = next(iter(train_loader))
    num_features = sample_past.shape[2]
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
    ).to(DEVICE)

    # 2) Optimizer + scheduler
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

    def lr_lambda(curr_epoch: int) -> float:
        if curr_epoch < warmup_epochs:
            return float(curr_epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(curr_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()

    # 3) Early stopping bookkeeping
    best_val_rmse = float("inf")
    best_state_dict = None
    patience_counter = 0

    # 4) Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for past, store_id, brand_id, target in train_loader:
            past = past.to(DEVICE, non_blocking=True)
            store_id = store_id.to(DEVICE, non_blocking=True)
            brand_id = brand_id.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            pred = model(past, store_id, brand_id)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * past.size(0)

        scheduler.step()
        train_rmse = math.sqrt(running_loss / len(train_loader.dataset))

        # 4a) Validate
        model.eval()
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for past, store_id, brand_id, target in val_loader:
                past = past.to(DEVICE, non_blocking=True)
                store_id = store_id.to(DEVICE, non_blocking=True)
                brand_id = brand_id.to(DEVICE, non_blocking=True)

                out = model(past, store_id, brand_id)
                val_preds.append(out.cpu().numpy())
                val_trues.append(target.numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_trues = np.concatenate(val_trues, axis=0)
        val_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))

        # 4b) Early stopping check
        if val_rmse < best_val_rmse - 1e-5:
            best_val_rmse = val_rmse
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # 5) Load best weights (if any) and return
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model


def transformer_sliding_window_evaluation(df: pd.DataFrame, L: int = WINDOW_SIZE) -> float:
    """
    1) Let max_week_idx = df['week_idx'].max(). We will loop t_idx = L .. max_week_idx-1.
    2) For each t_idx:
         - Build train_ds_t = windows with target_week_idx ≤ t_idx-1  (mode="train", cutoff_week_idx=t_idx-1)
         - Split train_ds_t 80/20 → (train_sub, val_sub), build DataLoaders
         - Train a fresh Transformer on (train_sub, val_sub)
         - Build test_one_ds_t = windows with target_week_idx == t_idx   (mode="test_one", cutoff_week_idx=t_idx-1)
         - Let the trained Transformer predict on test_one_ds_t; record the (pred, true)
    3) Concatenate all (pred, true) pairs across t_idx, compute RMSE.
    """

    max_week_idx = int(df["week_idx"].max())
    all_preds = []
    all_trues = []

    for t_idx in tqdm(range(L, max_week_idx), desc="Sliding‐window steps"):
        cutoff_idx = t_idx - 1

        # 1) Build “train all” for this step:
        train_all_ds = OJSequenceDataset(df, L=L, cutoff_week_idx=cutoff_idx, mode="train")
        if len(train_all_ds) == 0:
            continue

        # 2) Time‐aware 80/20 split of train_all_ds → train_sub / val_sub
        total_windows = len(train_all_ds)
        split_idx = int(total_windows * 0.8)
        train_sub, val_sub = torch.utils.data.random_split(
            train_all_ds,
            lengths=[split_idx, total_windows - split_idx],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_sub,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )
        val_loader = DataLoader(
            val_sub,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )

        # 3) Train a fresh Transformer on this data:
        model = train_one_transformer(train_loader, val_loader, df, L=L)

        # 4) Build the “test_one” set for t_idx:
        test_one_ds = OJSequenceDataset(df, L=L, cutoff_week_idx=cutoff_idx, mode="test_one")
        if len(test_one_ds) == 0:
            continue

        test_one_loader = DataLoader(
            test_one_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )

        # 5) Run inference on test_one_loader and collect (pred, true)
        model.eval()
        with torch.no_grad():
            for past, store_id, brand_id, target in test_one_loader:
                past     = past.to(DEVICE, non_blocking=True)
                store_id = store_id.to(DEVICE, non_blocking=True)
                brand_id = brand_id.to(DEVICE, non_blocking=True)

                out = model(past, store_id, brand_id)  # (batch,)
                all_preds.append(out.cpu().numpy())
                all_trues.append(target.numpy())

    if not all_preds:
        raise ValueError("No sliding‐window predictions were made. Check data or L.")

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    overall_rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
    return overall_rmse


def main() -> None:
    print("==== DEVICE CHECK ====")
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("DEVICE                      =", DEVICE)
    if torch.cuda.is_available():
        print("  GPU count                =", torch.cuda.device_count())
        print("  Current GPU index        =", torch.cuda.current_device())
        print("  Current GPU name         =", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("==== END DEVICE CHECK ====\n")

    # 1) Load and preprocess
    df = load_processed()

    # 2) Remap raw week IDs to 1..102 globally
    unique_global_weeks = sorted(df["week"].unique())  # e.g. [43, 44, …, 144]
    remap = {wk: i + 1 for i, wk in enumerate(unique_global_weeks)}
    df["week_idx"] = df["week"].map(remap)

    # 3) Now run sliding‐window over week_idx
    print("Running full sliding‐window Transformer evaluation...\n")
    rmse = transformer_sliding_window_evaluation(df, L=WINDOW_SIZE)
    print(f"\n→ Overall sliding‐window RMSE (Transformer): {rmse:.2f}")


if __name__ == "__main__":
    main()
