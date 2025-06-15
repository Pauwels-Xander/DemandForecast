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
WINDOW_SIZE = 52
EPOCHS = 20
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 3
WARMUP_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------
# 1) List of all regressors you want to use
FEATURE_COLS = [
    "store", "brand", "week",
    "OwnLogPrice",
    "LagLogPrice_2",
    "MinCompPrice",
    "CategoryFeatShare",
    "CrossFeatPressure",
    "LagLogSales",
    "DealFlag_L1"
]

# --------------------------------------------------------------------------

def load_processed() -> pd.DataFrame:
    from ProcessingOutliers import load_data, preprocess_features

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
    Sliding‐window dataset builder using exactly WINDOW_SIZE past weeks of ALL your FEATURE_COLS
    (minus store,brand,sales) to predict “sales” at the target week.
    Supports modes: train / test_one / test_all.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        L: int,
        cutoff_week_idx: int,
        mode: str = "test_one"
    ):
        assert mode in ("train", "test_one", "test_all")

        # Build list of feature columns (drop grouping + target)
        feature_cols = [c for c in FEATURE_COLS if c not in ("store", "brand", "sales")]

        # Drop any rows missing a feature or the sales target
        df_clean = df.dropna(subset=FEATURE_COLS + ["sales"])

        self.samples: List[SequenceSample] = []

        # Group by (store, brand); sort each group by “week_idx”
        for (_, _), grp in df_clean.groupby(["store", "brand"]):
            grp = grp.sort_values("week_idx").reset_index(drop=True)
            sid = int(grp["store_id"].iloc[0])
            bid = int(grp["brand_id"].iloc[0])

            # Build windows t = L .. len(grp)-1
            for t in range(L, len(grp)):
                week_idx_t = int(grp.iloc[t]["week_idx"])

                # Apply mode filter
                if mode == "train" and week_idx_t > cutoff_week_idx:
                    continue
                if mode == "test_one" and week_idx_t != (cutoff_week_idx + 1):
                    continue
                if mode == "test_all" and week_idx_t <= cutoff_week_idx:
                    continue

                # Extract past L×features
                past = grp.iloc[t - L : t][feature_cols].values.astype(np.float32)
                # Skip any window with missing
                if np.isnan(past).any():
                    continue

                target = float(grp.iloc[t]["sales"])
                # Skip if target NaN (unlikely after dropna, but safe)
                if np.isnan(target):
                    continue

                self.samples.append(SequenceSample(past, sid, bid, target))

        self.feature_cols = feature_cols

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample       = self.samples[idx]
        past_tensor  = torch.from_numpy(sample.past)                          # (L, num_features)
        store_tensor = torch.tensor(sample.store_id, dtype=torch.long)       # ( )
        brand_tensor = torch.tensor(sample.brand_id, dtype=torch.long)       # ( )
        target_tensor= torch.tensor(sample.target, dtype=torch.float32)      # ( )
        return past_tensor, store_tensor, brand_tensor, target_tensor

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class SmallTransformerV2(nn.Module):
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

        # Static embeddings
        self.store_emb = nn.Embedding(num_stores, d_model)
        self.brand_emb = nn.Embedding(num_brands, d_model)
        self.static_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Conv front‐end
        self.conv1 = nn.Conv1d(num_features, d_model, kernel_size=3, padding=1)
        self.conv_relu = nn.ReLU()

        # Positional + pre‐norm
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=L)
        self.pre_norm = nn.LayerNorm(d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model,  # model dimension
                                                    nhead,  # number of heads
                                                    d_model * 4,  # dim_feedforward
                                                    dropout = dropout,  # dropout rate
                                                    activation = "gelu",  # activation fn
                                                    batch_first = True)  # batch‐first inputs

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Combiner MLP
        self.combiner = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        past: torch.Tensor,       # (batch, L, num_features)
        store_id: torch.Tensor,   # (batch,)
        brand_id: torch.Tensor    # (batch,)
    ) -> torch.Tensor:
        # Static context
        s = self.store_emb(store_id)
        b = self.brand_emb(brand_id)
        static_ctx = self.static_mlp(s + b)

        # Conv over time
        x = past.transpose(1, 2)
        x = self.conv_relu(self.conv1(x))
        x = x.transpose(1, 2)

        # Positional + norm + encode
        x = self.pre_norm(self.pos_enc(x))
        x = self.encoder(x)

        # Take last token + combine
        seq_out = x[:, -1, :]
        combined = torch.cat([seq_out, static_ctx], dim=-1)
        return self.combiner(combined).squeeze(-1)

def train_one_transformer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    df: pd.DataFrame,
    L: int = WINDOW_SIZE,
    epochs: int = EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    warmup_epochs: int = WARMUP_EPOCHS
) -> SmallTransformerV2:
    sample_past, _, _, _ = next(iter(train_loader))
    num_features = sample_past.shape[2]
    num_stores = int(df["store_id"].nunique())
    num_brands = int(df["brand_id"].nunique())

    model = SmallTransformerV2(
        num_features=num_features,
        num_stores=num_stores,
        num_brands=num_brands,
        L=L
    ).to(DEVICE)

    no_decay = ["bias", "LayerNorm.weight"]
    opt_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-5
        },
        {
            "params": [p for n, p in model.named_parameters() if  any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = torch.optim.AdamW(opt_groups, lr=1e-3, betas=(0.9,0.999), eps=1e-6)
    scheduler = LambdaLR(optimizer, lambda e:
        (e+1)/warmup_epochs if e < warmup_epochs
        else 0.5*(1+math.cos(math.pi*(e-warmup_epochs)/(epochs-warmup_epochs)))
    )
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        run_loss = 0.0
        for past, s_id, b_id, tgt in train_loader:
            past, s_id, b_id, tgt = [t.to(DEVICE) for t in (past, s_id, b_id, tgt)]
            optimizer.zero_grad()
            out = model(past, s_id, b_id)
            loss = criterion(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run_loss += loss.item() * past.size(0)
        scheduler.step()
        train_rmse = math.sqrt(run_loss / len(train_loader.dataset))

        # validate
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for past, s_id, b_id, tgt in val_loader:
                past, s_id, b_id = past.to(DEVICE), s_id.to(DEVICE), b_id.to(DEVICE)
                out = model(past, s_id, b_id)
                vp.append(out.cpu().numpy())
                vt.append(tgt.numpy())
        vp = np.concatenate(vp); vt = np.concatenate(vt)
        val_rmse = np.sqrt(mean_squared_error(vt, vp))

        if val_rmse < best_rmse - 1e-5:
            best_rmse = val_rmse
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def transformer_sliding_window_evaluation(df: pd.DataFrame, L: int = WINDOW_SIZE) -> float:
    max_idx = int(df["week_idx"].max())
    all_preds, all_trues = [], []

    for t_idx in tqdm(range(L, max_idx), desc="Sliding‐window steps"):
        cutoff = t_idx - 1

        # build train / val
        train_ds = OJSequenceDataset(df, L, cutoff, mode="train")
        if len(train_ds)==0: continue
        split = int(len(train_ds)*0.8)
        t_sub, v_sub = torch.utils.data.random_split(train_ds, [split, len(train_ds)-split], generator=torch.Generator().manual_seed(42))
        t_loader = DataLoader(t_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        v_loader = DataLoader(v_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # train
        model = train_one_transformer(t_loader, v_loader, df, L=L)

        # test step
        test_ds = OJSequenceDataset(df, L, cutoff, mode="test_one")
        if len(test_ds)==0: continue
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model.eval()
        with torch.no_grad():
            for past, s_id, b_id, tgt in test_loader:
                past, s_id, b_id = past.to(DEVICE), s_id.to(DEVICE), b_id.to(DEVICE)
                out = model(past, s_id, b_id)
                all_preds.append(out.cpu().numpy())
                all_trues.append(tgt.numpy())

    if not all_preds:
        raise ValueError("No sliding‐window predictions were made. Check data or L.")

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    return float(np.sqrt(mean_squared_error(all_trues, all_preds)))

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

    # 2) Remap raw week IDs to 1..N globally
    unique_weeks = sorted(df["week"].unique())
    remap = {wk: i+1 for i, wk in enumerate(unique_weeks)}
    df["week_idx"] = df["week"].map(remap)

    # 3) Run sliding‐window Transformer evaluation
    print("Running full sliding‐window Transformer evaluation...\n")
    rmse = transformer_sliding_window_evaluation(df, L=WINDOW_SIZE)
    print(f"\n→ Overall sliding‐window RMSE (Transformer): {rmse:.2f}")

if __name__ == "__main__":
    main()
