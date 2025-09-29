import os
import io
import math
import time
import uuid
import json
import joblib
import pickle
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# optional libs
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cascade_trader_app")
logger.setLevel(logging.INFO)

# Streamlit UI top-level
st.set_page_config(page_title="Cascade Trader — Scope→Aim→Shoot", layout="wide")
st.title("Cascade Trader — Scope → Aim → Shoot (L1/L2/L3)")

# --- Basic symbol/time inputs ---
symbol = st.text_input("Symbol (Yahoo)", value="GC=F")
start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=90))
end_date = st.date_input("End date", value=datetime.today())
interval = st.selectbox("Interval", ["1d", "1h", "15m", "5m", "1m"], index=0)

# --- Training & model settings ---
st.sidebar.header("Training & Model")
num_boost = int(st.sidebar.number_input("XGBoost rounds (if used)", min_value=1, value=200))
early_stop = int(st.sidebar.number_input("Early stopping rounds", min_value=1, value=20))
test_size = float(st.sidebar.number_input("Validation fraction", min_value=0.01, max_value=0.5, value=0.2))
seq_len = int(st.sidebar.number_input("L1 sequence length", min_value=8, max_value=256, value=64, step=8))
device_choice = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

# --- Confirm thresholds used in quick sims ---
st.sidebar.header("Quick sim thresholds")
p_fast = st.sidebar.number_input("Confirm threshold (fast)", 0.0, 1.0, 0.60)
p_slow = st.sidebar.number_input("Confirm threshold (slow)", 0.0, 1.0, 0.55)
p_deep = st.sidebar.number_input("Confirm threshold (deep)", 0.0, 1.0, 0.45)

# --- Level gating UI (explicit exclusive ranges) ---
st.sidebar.header("Level gating (exclusive ranges) — signal scale 0..10")
st.sidebar.markdown("Define explicit buy/sell ranges per level. Ranges should be exclusive (L3 inside L2 inside L1).")
# L1
lvl1_buy_min = st.sidebar.number_input("L1 buy min", 0.0, 10.0, 5.5, step=0.1)
lvl1_buy_max = st.sidebar.number_input("L1 buy max", 0.0, 10.0, 9.9, step=0.1)
lvl1_sell_min = st.sidebar.number_input("L1 sell min", 0.0, 10.0, 0.0, step=0.1)
lvl1_sell_max = st.sidebar.number_input("L1 sell max", 0.0, 10.0, 4.5, step=0.1)
# L2
lvl2_buy_min = st.sidebar.number_input("L2 buy min", 0.0, 10.0, 6.0, step=0.1)
lvl2_buy_max = st.sidebar.number_input("L2 buy max", 0.0, 10.0, 9.9, step=0.1)
lvl2_sell_min = st.sidebar.number_input("L2 sell min", 0.0, 10.0, 0.0, step=0.1)
lvl2_sell_max = st.sidebar.number_input("L2 sell max", 0.0, 10.0, 4.0, step=0.1)
# L3
lvl3_buy_min = st.sidebar.number_input("L3 buy min", 0.0, 10.0, 6.5, step=0.1)
lvl3_buy_max = st.sidebar.number_input("L3 buy max", 0.0, 10.0, 9.9, step=0.1)
lvl3_sell_min = st.sidebar.number_input("L3 sell min", 0.0, 10.0, 0.0, step=0.1)
lvl3_sell_max = st.sidebar.number_input("L3 sell max", 0.0, 10.0, 3.5, step=0.1)

# --- Breadth & sweep controls ---
st.sidebar.header("Breadth & Sweep")
run_breadth = st.sidebar.checkbox("Enable breadth backtest (on full run)", value=True)
run_sweep = st.sidebar.checkbox("Enable grid sweep (on full run)", value=False)

# One-button pipeline: fetch → candidates → train → export
st.sidebar.header("Run pipeline")
run_full_pipeline_btn = st.sidebar.button("Run full pipeline (fetch → train → export)")

# Chunk 2/8: features, sequences, and dataset classes

def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    """Compute a compact set of engineered features from OHLCV."""
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(0.0, index=df.index)

    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    f['logret1'] = np.log1p(ret1.replace(-1, -0.999999))
    tr = (h - l).clip(lower=0)
    f['tr'] = tr.fillna(0.0)
    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'vol_z_{w}'] = (v.rolling(w).mean() - v.rolling(max(1,w*3)).mean()).fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)
    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def to_sequences(features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Build sequences ending at each index t: [t-seq_len+1, ..., t]
    Returns shape [N, seq_len, F]
    """
    Nrows, F = features.shape
    X = np.zeros((len(indices), seq_len, F), dtype=features.dtype)
    for i, t in enumerate(indices):
        t = int(t)
        t0 = t - seq_len + 1
        if t0 < 0:
            pad_count = -t0
            pad = np.repeat(features[[0]], pad_count, axis=0)
            seq = np.vstack([pad, features[0:t+1]])
        else:
            seq = features[t0:t+1]
        if seq.shape[0] < seq_len:
            pad_needed = seq_len - seq.shape[0]
            pad = np.repeat(seq[[0]], pad_needed, axis=0)
            seq = np.vstack([pad, seq])
        X[i] = seq[-seq_len:]
    return X

# Dataset wrappers for torch
if torch is not None:
    from torch.utils.data import Dataset, DataLoader

    class SequenceDataset(Dataset):
        def __init__(self, X_seq: np.ndarray, y: np.ndarray):
            self.X = X_seq.astype(np.float32)  # [N, T, F]
            # model expects [B, F, T], we'll transpose in __getitem__
            self.y = y.astype(np.float32).reshape(-1, 1)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            x = self.X[idx].transpose(1,0)  # [F, T]
            y = self.y[idx]
            return x, y

    class TabDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32).reshape(-1, 1)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
else:
    # Provide no-op placeholders to avoid NameError in non-train flows
    SequenceDataset = None
    TabDataset = None
# Chunk 3/8: candidate generation, simulation and summarization

def ensure_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return df
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()

def _true_range(high, low, close):
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr

def generate_candidates_and_labels(
    bars: pd.DataFrame,
    lookback: int = 64,
    k_tp: float = 3.0,
    k_sl: float = 1.0,
    atr_window: int = 14,
    max_bars: int = 60,
    direction: str = "long"
) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    for col in ("high","low","close"):
        if col not in bars.columns:
            raise KeyError(f"Missing column {col}")
    bars["tr"] = _true_range(bars["high"], bars["low"], bars["close"])
    bars["atr"] = bars["tr"].rolling(atr_window, min_periods=1).mean()
    records = []
    n = len(bars)
    for i in range(lookback, n):
        t = bars.index[i]
        entry_px = float(bars["close"].iat[i])
        atr_val = float(bars["atr"].iat[i])
        if atr_val <= 0 or math.isnan(atr_val):
            continue
        sl_px = entry_px - k_sl*atr_val if direction=="long" else entry_px + k_sl*atr_val
        tp_px = entry_px + k_tp*atr_val if direction=="long" else entry_px - k_tp*atr_val
        end_i = min(i+max_bars, n-1)
        label, hit_i, hit_px = 0, end_i, float(bars["close"].iat[end_i])
        for j in range(i+1, end_i+1):
            hi, lo = float(bars["high"].iat[j]), float(bars["low"].iat[j])
            if direction=="long":
                if hi >= tp_px:
                    label, hit_i, hit_px = 1, j, tp_px; break
                if lo <= sl_px:
                    label, hit_i, hit_px = 0, j, sl_px; break
            else:
                if lo <= tp_px:
                    label, hit_i, hit_px = 1, j, tp_px; break
                if hi >= sl_px:
                    label, hit_i, hit_px = 0, j, sl_px; break
        end_t = bars.index[hit_i]
        ret_val = (hit_px-entry_px)/entry_px if direction=="long" else (entry_px-hit_px)/entry_px
        dur_min = (end_t - t).total_seconds()/60.0
        records.append(dict(candidate_time=t,
                            entry_price=entry_px,
                            atr=float(atr_val),
                            sl_price=float(sl_px),
                            tp_price=float(tp_px),
                            end_time=end_t,
                            label=int(label),
                            duration=float(dur_min),
                            realized_return=float(ret_val),
                            direction=direction))
    return pd.DataFrame(records)

def simulate_limits(df: pd.DataFrame,
                    bars: pd.DataFrame,
                    label_col: str = "pred_label",
                    symbol: str = "GC=F",
                    sl: float = 0.02,
                    tp: float = 0.04,
                    max_holding: int = 60) -> pd.DataFrame:
    if df is None or df.empty or bars is None or bars.empty:
        return pd.DataFrame()
    trades = []
    bars = bars.copy(); bars.index = pd.to_datetime(bars.index)
    for _, row in df.iterrows():
        lbl = row.get(label_col, 0)
        if lbl == 0 or pd.isna(lbl):
            continue
        entry_t = pd.to_datetime(row.get("candidate_time", row.name))
        if entry_t not in bars.index:
            continue
        entry_px = float(bars.loc[entry_t, "close"])
        direction = 1 if lbl > 0 else -1
        sl_px = entry_px * (1 - sl) if direction > 0 else entry_px * (1 + sl)
        tp_px = entry_px * (1 + tp) if direction > 0 else entry_px * (1 - tp)
        exit_t, exit_px, pnl = None, None, None
        segment = bars.loc[entry_t:].head(max_holding)
        if segment.empty:
            continue
        for t, b in segment.iterrows():
            lo, hi = float(b["low"]), float(b["high"])
            if direction > 0:
                if lo <= sl_px:
                    exit_t, exit_px, pnl = t, sl_px, -sl; break
                if hi >= tp_px:
                    exit_t, exit_px, pnl = t, tp_px, tp; break
            else:
                if hi >= sl_px:
                    exit_t, exit_px, pnl = t, sl_px, -sl; break
                if lo <= tp_px:
                    exit_t, exit_px, pnl = t, tp_px, tp; break
        if exit_t is None:
            last_bar = segment.iloc[-1]
            exit_t = last_bar.name
            exit_px = float(last_bar["close"])
            pnl = (exit_px - entry_px)/entry_px * direction
        trades.append(dict(symbol=symbol,
                           entry_time=entry_t,
                           entry_price=entry_px,
                           direction=direction,
                           exit_time=exit_t,
                           exit_price=exit_px,
                           pnl=float(pnl)))
    return pd.DataFrame(trades)

def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    total_trades = len(trades)
    win_rate = float((trades["pnl"] > 0).mean())
    avg_pnl = float(trades["pnl"].mean())
    median_pnl = float(trades["pnl"].median())
    total_pnl = float(trades["pnl"].sum())
    max_dd = float(trades["pnl"].cumsum().min())
    start_time = trades["entry_time"].min()
    end_time = trades["exit_time"].max()
    return pd.DataFrame([{
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "median_pnl": median_pnl,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "start_time": start_time,
        "end_time": end_time
    }])
# Chunk 4/8: model building blocks and temperature scaler

if torch is None:
    # We'll still allow non-training flows, but training button should error early.
    logger.warning("Torch not available — training disabled. Install torch to enable cascade training.")

else:
    class ConvBlock(nn.Module):
        def __init__(self, c_in, c_out, k, d, pdrop):
            super().__init__()
            pad = (k - 1) * d // 2
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
            self.bn = nn.BatchNorm1d(c_out)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(pdrop)
            self.res = (c_in == c_out)
        def forward(self, x):
            out = self.conv(x); out = self.bn(out); out = self.act(out); out = self.drop(out)
            if self.res: out = out + x
            return out

    class Level1ScopeCNN(nn.Module):
        def __init__(self, in_features=12, channels=(32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1):
            super().__init__()
            chs = [in_features] + list(channels)
            blocks = []
            for i in range(len(channels)):
                k = kernel_sizes[min(i, len(kernel_sizes)-1)]
                d = dilations[min(i, len(dilations)-1)]
                blocks.append(ConvBlock(chs[i], chs[i+1], k, d, dropout))
            self.blocks = nn.Sequential(*blocks)
            self.project = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
            self.head = nn.Linear(chs[-1], 1)
        @property
        def embedding_dim(self): return int(self.blocks[-1].conv.out_channels)
        def forward(self, x):
            z = self.blocks(x)
            z = self.project(z)
            z_pool = z.mean(dim=-1)
            logit = self.head(z_pool)
            return logit, z_pool

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden, out_dim=1, dropout=0.1):
            super().__init__()
            layers = []
            last = in_dim
            for h in hidden:
                layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
                last = h
            layers += [nn.Linear(last, out_dim)]
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    class Level3ShootMLP(nn.Module):
        def __init__(self, in_dim, hidden=(128,64), dropout=0.1, use_regression_head=True):
            super().__init__()
            self.backbone = MLP(in_dim, list(hidden), out_dim=128, dropout=dropout)
            self.cls_head = nn.Linear(128, 1)
            self.reg_head = nn.Linear(128, 1) if use_regression_head else None
        def forward(self, x):
            h = self.backbone(x)
            logit = self.cls_head(h)
            ret = self.reg_head(h) if self.reg_head is not None else None
            return logit, ret

    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_temp = nn.Parameter(torch.zeros(1))
        def forward(self, logits):
            T = torch.exp(self.log_temp)
            return logits / T
        def fit(self, logits: np.ndarray, y: np.ndarray, max_iter=200, lr=1e-2):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            logits_t = torch.tensor(logits.reshape(-1,1), dtype=torch.float32, device=device)
            y_t = torch.tensor(y.reshape(-1,1), dtype=torch.float32, device=device)
            opt = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
            bce = nn.BCEWithLogitsLoss()
            def closure():
                opt.zero_grad()
                scaled = self.forward(logits_t)
                loss = bce(scaled, y_t)
                loss.backward()
                return loss
            try:
                opt.step(closure)
            except Exception as e:
                logger.warning("Temp scaler LBFGS failed: %s", e)
        def transform(self, logits: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                device = next(self.parameters()).device
                logits_t = torch.tensor(logits.reshape(-1,1), dtype=torch.float32, device=device)
                scaled = self.forward(logits_t).cpu().numpy()
            return scaled.reshape(-1)

# Chunk 5/8: training helpers and CascadeTrader (fit + predict)

def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def train_torch_classifier(model: nn.Module,
                           train_ds,
                           val_ds,
                           lr: float = 1e-3,
                           epochs: int = 10,
                           patience: int = 3,
                           pos_weight: float = 1.0,
                           device: str = "auto"):
    dev = _device(device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pos_weight_t = torch.tensor([pos_weight], device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=getattr(train_ds, "batch_size", 128), shuffle=True)
    va_loader = torch.utils.data.DataLoader(val_ds, batch_size=getattr(val_ds, "batch_size", 1024), shuffle=False)
    best_loss = float("inf"); best_state = None; no_imp = 0
    history = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        running_loss = 0.0; n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            logit = out[0] if isinstance(out, tuple) else out
            loss = bce(logit, yb)
            loss.backward()
            opt.step()
            running_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        train_loss = running_loss / max(1, n)
        # val
        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                logit = out[0] if isinstance(out, tuple) else out
                loss = bce(logit, yb)
                vloss += float(loss.item()) * xb.size(0)
                vn += xb.size(0)
        val_loss = vloss / max(1, vn)
        history["train"].append(train_loss); history["val"].append(val_loss)
        if val_loss + 1e-8 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_loss, "history": history}

class CascadeTrader:
    def __init__(self, seq_len: int = 64, feat_windows=(5,10,20), device: str = "auto"):
        if torch is None:
            raise RuntimeError("Torch required for CascadeTrader")
        self.seq_len = seq_len
        self.feat_windows = feat_windows
        self.device = _device(device)
        self.scaler_seq = StandardScaler()
        self.scaler_tab = StandardScaler()
        self.l1 = None
        self.l1_temp = TemperatureScaler()
        self.l2_backend = None
        self.l2_model = None
        self.l3 = None
        self.l3_temp = TemperatureScaler()
        self.tab_feature_names = []
        self._fitted = False
        self.metadata = {}

    def fit(self, df: pd.DataFrame, events: pd.DataFrame, l2_use_xgb: bool = True, epochs_l1: int = 10, epochs_l23: int = 10):
        """
        Fit the cascade on df using events (columns 't' indices and 'y' labels).
        Training does:
          - compute features
          - build sequences and tabular
          - train L1 (CNN)
          - build embeddings + tabular matrix
          - train L2 (xgb/MLP) and L3 (MLP) in parallel
        """
        t0 = time.time()
        df = ensure_unique_index(df)
        eng = compute_engineered_features(df, windows=self.feat_windows)
        seq_cols = ['open','high','low','close','volume']
        # pick micro columns present
        micro_cols = [c for c in ['ret1','tr','vol_5','mom_5','chanpos_10'] if c in eng.columns or c in df.columns]
        # build feature frames
        feat_seq_df = pd.concat([df[seq_cols].astype(float), eng[[c for c in micro_cols if c in eng.columns]]], axis=1, sort=False).fillna(0.0)
        feat_tab_df = eng.fillna(0.0)
        self.tab_feature_names = list(feat_tab_df.columns)
        # prepare event indices and labels
        idx = events['t'].astype(int).values
        y = events['y'].astype(int).values
        # time-based or stratified split: here we do stratified for simplicity
        tr_idx, va_idx = train_test_split(np.arange(len(idx)), test_size=test_size, random_state=42, stratify=y)
        idx_tr, idx_va = idx[tr_idx], idx[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        # fit scalers on training positions
        X_seq_all = feat_seq_df.values
        self.scaler_seq.fit(X_seq_all[idx_tr])
        X_seq_all_scaled = self.scaler_seq.transform(X_seq_all)
        X_tab_all = feat_tab_df.values
        self.scaler_tab.fit(X_tab_all[idx_tr])
        X_tab_all_scaled = self.scaler_tab.transform(X_tab_all)
        # build sequences
        Xseq_tr = to_sequences(X_seq_all_scaled, idx_tr, seq_len=self.seq_len)
        Xseq_va = to_sequences(X_seq_all_scaled, idx_va, seq_len=self.seq_len)
        # datasets
        ds_l1_tr = SequenceDataset(Xseq_tr, y_tr)
        ds_l1_va = SequenceDataset(Xseq_va, y_va)
        ds_l1_tr.batch_size = 128
        ds_l1_va.batch_size = 256
        # init L1 with correct in_features
        in_features = Xseq_tr.shape[2]
        self.l1 = Level1ScopeCNN(in_features=in_features, channels=(32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1)
        self.l1, l1_hist = train_torch_classifier(self.l1, ds_l1_tr, ds_l1_va, lr=1e-3, epochs=epochs_l1, patience=3, pos_weight=1.0, device=str(self.device))
        # infer L1 embeddings for full event set
        all_idx_seq = to_sequences(X_seq_all_scaled, idx, seq_len=self.seq_len)
        l1_logits, l1_emb = self._l1_infer_logits_emb(all_idx_seq)
        # split embedding arrays to train/val
        l1_emb_tr, l1_emb_va = l1_emb[tr_idx], l1_emb[va_idx]
        Xtab_tr = X_tab_all_scaled[idx_tr]; Xtab_va = X_tab_all_scaled[idx_va]
        X_l2_tr = np.hstack([l1_emb_tr, Xtab_tr]); X_l2_va = np.hstack([l1_emb_va, Xtab_va])
        # Train L2 (xgb if available and requested) and L3 in parallel
        results = {}
        def do_l2_xgb():
            if xgb is None:
                raise RuntimeError("xgboost not installed")
            clf = xgb.XGBClassifier(n_estimators=num_boost, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss")
            clf.fit(X_l2_tr, y_tr, eval_set=[(X_l2_va, y_va)], verbose=False)
            return ("xgb", clf)
        def do_l2_mlp():
            in_dim = X_l2_tr.shape[1]
            m = MLP(in_dim, [128,64], out_dim=1, dropout=0.1)
            ds2_tr = TabDataset(X_l2_tr, y_tr)
            ds2_va = TabDataset(X_l2_va, y_va)
            ds2_tr.batch_size = 256; ds2_va.batch_size = 512
            m, hist = train_torch_classifier(m, ds2_tr, ds2_va, lr=1e-3, epochs=epochs_l23, patience=3, pos_weight=1.0, device=str(self.device))
            return ("mlp", m)
        def do_l3():
            in_dim = X_l2_tr.shape[1]
            m3 = Level3ShootMLP(in_dim, hidden=(128,64), dropout=0.1, use_regression_head=True)
            ds3_tr = TabDataset(X_l2_tr, y_tr)
            ds3_va = TabDataset(X_l2_va, y_va)
            ds3_tr.batch_size = 256; ds3_va.batch_size = 512
            m3, hist3 = train_torch_classifier(m3, ds3_tr, ds3_va, lr=1e-3, epochs=epochs_l23, patience=3, pos_weight=1.0, device=str(self.device))
            return ("l3", (m3, hist3))
        futures = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            if (xgb is not None) and l2_use_xgb:
                futures[ex.submit(do_l2_xgb)] = "l2"
            else:
                futures[ex.submit(do_l2_mlp)] = "l2"
            futures[ex.submit(do_l3)] = "l3"
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    results[label] = fut.result()
                except Exception as e:
                    logger.exception("Parallel train error for %s: %s", label, e)
                    results[label] = None
        # attach L2/L3
        if results.get("l2") is not None:
            self.l2_backend, self.l2_model = results["l2"]
        if results.get("l3") is not None:
            # results["l3"] == ("l3", (m3, hist3))
            self.l3 = results["l3"][1][0] if isinstance(results["l3"][1], tuple) else results["l3"][1]
        # calibrate temperature scalers (best-effort)
        try:
            self.l1_temp.fit(l1_logits.reshape(-1,1), y)
        except Exception as e:
            logger.warning("L1 temperature scaling failed: %s", e)
        try:
            l3_val_logits = self._l3_infer_logits(X_l2_va)
            self.l3_temp.fit(l3_val_logits.reshape(-1,1), y_va)
        except Exception as e:
            logger.warning("L3 temperature scaling failed: %s", e)
        self.metadata = {"l1_hist": l1_hist, "fit_time_sec": round(time.time() - t0, 2), "l2_backend": self.l2_backend}
        self._fitted = True
        logger.info("CascadeTrader.fit finished in %.2fs", time.time() - t0)
        return self

    def _l1_infer_logits_emb(self, Xseq: np.ndarray):
        self.l1.eval()
        logits = []; embeds = []
        batch = 256
        with torch.no_grad():
            for i in range(0, len(Xseq), batch):
                sub = Xseq[i:i+batch]
                xb = torch.tensor(sub.transpose(0,2,1), dtype=torch.float32, device=self.device)  # [B, F, T]
                logit, emb = self.l1(xb)
                logits.append(logit.detach().cpu().numpy())
                embeds.append(emb.detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1,1), np.concatenate(embeds, axis=0)

    def _l3_infer_logits(self, X: np.ndarray):
        self.l3.eval()
        logits = []
        batch = 2048
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=self.device)
                logit, _ = self.l3(xb)
                logits.append(logit.detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1,1)

    def _mlp_predict_proba(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        probs = []
        batch = 4096
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=self.device)
                logit = model(xb)
                p = torch.sigmoid(logit).cpu().numpy().reshape(-1)
                probs.append(p)
        return np.concatenate(probs, axis=0)

    def predict_batch(self, df: pd.DataFrame, t_indices: np.ndarray) -> pd.DataFrame:
        assert self._fitted, "CascadeTrader not fitted"
        eng = compute_engineered_features(df, windows=self.feat_windows)
        seq_cols = ['open','high','low','close','volume']; micro_cols=['ret1','tr','vol_5','mom_5','chanpos_10']
        use_cols = [c for c in seq_cols + micro_cols if c in list(df.columns) + list(eng.columns)]
        feat_seq_df = pd.concat([df[seq_cols].astype(float), eng[[c for c in micro_cols if c in eng.columns]]], axis=1)[use_cols].fillna(0.0)
        feat_tab_df = eng[self.tab_feature_names].fillna(0.0)
        X_seq_all_scaled = self.scaler_seq.transform(feat_seq_df.values)
        X_tab_all_scaled = self.scaler_tab.transform(feat_tab_df.values)
        Xseq = to_sequences(X_seq_all_scaled, t_indices, seq_len=self.seq_len)
        l1_logits, l1_emb = self._l1_infer_logits_emb(Xseq)
        l1_logits_scaled = self.l1_temp.transform(l1_logits.reshape(-1,1)).reshape(-1)
        p1 = 1.0/(1.0+np.exp(-l1_logits_scaled))
        go2 = p1 >= 0.30
        X_l2 = np.hstack([l1_emb, X_tab_all_scaled[t_indices]])
        if self.l2_backend == "xgb":
            try:
                p2 = self.l2_model.predict_proba(X_l2)[:,1]
            except Exception:
                p2 = np.zeros(len(X_l2))
        else:
            p2 = self._mlp_predict_proba(self.l2_model, X_l2)
        go3 = (p2 >= 0.55) & go2
        p3 = np.zeros_like(p1); rhat = np.zeros_like(p1)
        if go3.any() and self.l3 is not None:
            X_l3 = X_l2[go3]
            l3_logits = self._l3_infer_logits(X_l3)
            l3_logits_scaled = self.l3_temp.transform(l3_logits.reshape(-1,1)).reshape(-1)
            p3_vals = 1.0/(1.0+np.exp(-l3_logits_scaled))
            p3[go3] = p3_vals
            rhat[go3] = p3_vals - 0.5
        trade = (p3 >= 0.65) & go3
        size = np.clip(rhat, 0.0, None) * trade.astype(float)
        return pd.DataFrame({"t": t_indices, "p1": p1, "p2": p2, "p3": p3, "go2": go2.astype(int), "go3": go3.astype(int), "trade": trade.astype(int), "size": size})

# Chunk 6/8: export_model_and_metadata (user-provided) + breadth and sweep helpers

def export_model_and_metadata(model_wrapper, feature_list: List[str], metrics: Dict[str,Any], model_basename: str, save_fi: bool = True):
    """
    Save model artifacts and a portable .pt bundle for each model_wrapper.
    Returns dict of saved paths.
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    paths = {}
    try:
        # native xgb model file if booster exists
        if hasattr(model_wrapper, "booster") and getattr(model_wrapper, "booster") is not None:
            model_file = f"{model_basename}_{ts}.model"
            try:
                logger.info(f"Exporting native model file to {model_file}")
                model_wrapper.save_model(model_file)
                paths["model"] = model_file
                logger.info(f"Successfully saved native model file to {model_file}")
            except Exception as e:
                error_msg = f"Failed to save native booster file: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                paths["model_error"] = error_msg

        # metadata
        meta_file = f"{model_basename}_{ts}.json"
        try:
            logger.info(f"Exporting metadata to {meta_file}")
            with open(meta_file, "w") as f:
                json.dump({"features": feature_list, "metrics": metrics, "saved_at": ts}, f, default=str, indent=2)
            paths["meta"] = meta_file
            logger.info(f"Successfully saved metadata to {meta_file}")
        except Exception as e:
            error_msg = f"Failed to save metadata file: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            paths["meta_error"] = error_msg

        # feature importance
        if save_fi and hasattr(model_wrapper, "feature_importance"):
            try:
                logger.info("Generating feature importance data")
                fi_df = model_wrapper.feature_importance()
                fi_path = f"{model_basename}_{ts}_fi.csv"
                fi_df.to_csv(fi_path, index=False)
                paths["feature_importance"] = fi_path
                logger.info(f"Successfully saved feature importance to {fi_path}")
            except Exception as e:
                error_msg = f"Could not save feature importance: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                paths["fi_error"] = error_msg

        # Save portable .pt as a bundle (joblib or torch.save)
        pt_path = f"{model_basename}_{ts}.pt"
        try:
            logger.info(f"Preparing to save .pt bundle to {pt_path}")
            payload = {"model_wrapper": model_wrapper, "features": feature_list, "metrics": metrics, "export_log": []}
            payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Starting export of .pt bundle")
            if 'torch' in globals() and torch is not None:
                payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Using torch.save for .pt export")
                torch.save(payload, pt_path)
            else:
                payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] torch not available, using joblib for .pt export")
                joblib.dump(payload, pt_path)
            payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Successfully saved .pt bundle")
            paths["pt"] = pt_path
            logger.info(f"Successfully saved .pt bundle to {pt_path}")
        except Exception as e:
            error_msg = f"Failed to save .pt payload: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            paths["pt_error"] = error_msg
            # fallback
            fallback = f"{model_basename}_{ts}_fallback.pt"
            try:
                logger.info(f"Attempting joblib fallback to {fallback}")
                fallback_payload = {
                    "model_wrapper": model_wrapper,
                    "features": feature_list,
                    "metrics": metrics,
                    "export_error": error_msg,
                    "export_log": [
                        f"[{datetime.utcnow().isoformat()}] Primary .pt export failed",
                        f"[{datetime.utcnow().isoformat()}] Error: {str(e)}",
                        f"[{datetime.utcnow().isoformat()}] Attempting fallback with joblib"
                    ]
                }
                joblib.dump(fallback_payload, fallback)
                paths["pt_fallback"] = fallback
                logger.info(f"Successfully saved fallback .pt bundle to {fallback}")
            except Exception as fallback_e:
                fallback_error = f"Fallback also failed: {str(fallback_e)}\n{traceback.format_exc()}"
                logger.error(fallback_error)
                paths["pt_fallback_error"] = fallback_error
    except Exception as exc:
        error_msg = f"Failed to export model: {str(exc)}\n{traceback.format_exc()}"
        logger.exception(error_msg)
        paths["export_error"] = error_msg

    if "pt" in paths:
        try:
            logger.info(f"Verifying .pt bundle at {paths['pt']}")
            pt_file_size = os.path.getsize(paths["pt"])
            if pt_file_size > 0:
                logger.info(f"Verified .pt bundle exists with size {pt_file_size} bytes")
                paths["pt_verification"] = f"File size: {pt_file_size} bytes"
            else:
                logger.warning(f".pt bundle has zero size")
                paths["pt_verification"] = "Warning: Zero file size"
        except Exception as e:
            verification_error = f"PT verification failed: {str(e)}"
            logger.error(verification_error)
            paths["pt_verification_error"] = verification_error
    return paths

def run_breadth_levels(preds: pd.DataFrame, cands: pd.DataFrame, bars: pd.DataFrame) -> Dict[str,Any]:
    """Apply exclusive level ranges to preds -> cands and simulate trades per level."""
    out = {"detailed": {}, "summary": []}
    df = cands.copy().reset_index(drop=True)
    # Map preds.t -> df rows (assume aligned as used earlier)
    preds_indexed = preds.set_index('t')
    df['signal'] = (preds_indexed.reindex(df.index)['p3'].fillna(0.0) * 10).values
    def run_level(name, buy_min, buy_max, sl_pct, rr):
        sel = df[(df['signal'] >= buy_min) & (df['signal'] <= buy_max)].copy()
        tp_pct = rr * sl_pct
        if sel.empty:
            out['detailed'][name] = pd.DataFrame()
            return
        sel['pred_label'] = (sel['signal'] >= buy_min).astype(int)  # simple mapping
        trades = simulate_limits(sel, bars, label_col='pred_label', sl=sl_pct, tp=tp_pct, max_holding=60)
        out['detailed'][name] = trades
        s = summarize_trades(trades)
        if not s.empty:
            row = s.iloc[0].to_dict(); row['mode'] = name; out['summary'].append(row)
    # apply L1/L2/L3 with sl/rr defaults from UI if present
    # default sl & rr values if the UI didn't include them (safe fallback)
    l1_sl = float((globals().get("lvl1_sl_min",0.02) + globals().get("lvl1_sl_max",0.04))/2.0)
    l1_rr = float((globals().get("lvl1_rr_min",2.0) + globals().get("lvl1_rr_max",2.5))/2.0)
    l2_sl = float((globals().get("lvl2_sl_min",0.01) + globals().get("lvl2_sl_max",0.03))/2.0)
    l2_rr = float((globals().get("lvl2_rr_min",2.0) + globals().get("lvl2_rr_max",3.5))/2.0)
    l3_sl = float((globals().get("lvl3_sl_min",0.005) + globals().get("lvl3_sl_max",0.02))/2.0)
    l3_rr = float((globals().get("lvl3_rr_min",3.0) + globals().get("lvl3_rr_max",5.0))/2.0)
    run_level("L1", lvl1_buy_min, lvl1_buy_max, l1_sl, l1_rr)
    run_level("L2", lvl2_buy_min, lvl2_buy_max, l2_sl, l2_rr)
    run_level("L3", lvl3_buy_min, lvl3_buy_max, l3_sl, l3_rr)
    return out

def run_grid_sweep(clean: pd.DataFrame, bars: pd.DataFrame, rr_vals: List[float], sl_ranges: List[Tuple[float,float]], mpt_list: List[float]) -> Dict[str,Any]:
    results = {}
    if clean is None or clean.empty:
        return results
    for rr in rr_vals:
        for sl_min, sl_max in sl_ranges:
            sl_pct = (sl_min + sl_max) / 2.0
            tp_pct = rr * sl_pct
            for mpt in mpt_list:
                key = f"rr{rr:.2f}_sl{sl_min:.3f}-{sl_max:.3f}_mpt{mpt:.2f}"
                try:
                    df = clean.copy()
                    if "pred_prob" not in df.columns:
                        df["pred_prob"] = 0.0
                    df["pred_label"] = (df["pred_prob"] >= mpt).astype(int)
                    trades = simulate_limits(df, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=60)
                    results[key] = {"trades_count": len(trades), "trades": trades}
                except Exception as exc:
                    logger.exception("Sweep config %s failed: %s", key, exc)
                    results[key] = {"error": str(exc)}
    return results

# Chunk 7/8: Streamlit actions including one-button run_full_pipeline

# session state containers
if "bars" not in st.session_state: st.session_state.bars = pd.DataFrame()
if "cands" not in st.session_state: st.session_state.cands = pd.DataFrame()
if "trader" not in st.session_state: st.session_state.trader = None
if "export_paths" not in st.session_state: st.session_state.export_paths = {}

def fetch_and_prepare():
    st.info(f"Fetching {symbol} {interval} from YahooQuery …")
    bars = pd.DataFrame()
    if YahooTicker is None:
        st.error("yahooquery not installed — cannot fetch price data.")
        return pd.DataFrame()
    try:
        tq = YahooTicker(symbol)
        raw = tq.history(start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
        if raw is None:
            st.error("No data returned from yahooquery.")
            return pd.DataFrame()
        if isinstance(raw, dict): raw = pd.DataFrame(raw)
        if isinstance(raw.index, pd.MultiIndex): raw = raw.reset_index(level=0, drop=True)
        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index()
        raw.columns = [c.lower() for c in raw.columns]
        if "close" not in raw.columns and "adjclose" in raw.columns:
            raw["close"] = raw["adjclose"]
        bars = raw[~raw.index.duplicated(keep="first")]
        st.success(f"Fetched {len(bars)} bars.")
    except Exception as e:
        logger.exception("Fetch failed: %s", e)
        st.error(f"Fetch failed: {e}")
    return bars

def run_full_pipeline():
    # 1) fetch
    bars = fetch_and_prepare()
    if bars is None or bars.empty:
        return {"error": "No bars"}
    st.session_state.bars = bars
    bars = ensure_unique_index(bars)
    # 2) compute rvol and candidates
    try:
        bars["rvol"] = (bars["volume"] / bars["volume"].rolling(20, min_periods=1).mean()).fillna(1.0)
    except Exception:
        bars["rvol"] = 1.0
    cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
    if cands is None or cands.empty:
        st.warning("No candidates generated.")
        st.session_state.cands = pd.DataFrame()
        return {"error": "No candidates"}
    # add placeholder pred_prob column for sim if needed
    cands["pred_prob"] = 0.0
    st.session_state.cands = cands
    st.success(f"Generated {len(cands)} candidates.")
    # 3) map candidate_time to integer positions for events
    bar_idx_map = {t: i for i, t in enumerate(bars.index)}
    cand_idx = []
    for t in cands["candidate_time"]:
        t0 = pd.Timestamp(t)
        if t0 in bar_idx_map:
            cand_idx.append(bar_idx_map[t0])
        else:
            locs = bars.index[bars.index <= t0]
            cand_idx.append(int(bar_idx_map[locs[-1]] if len(locs) else 0))
    events = pd.DataFrame({"t": np.array(cand_idx, dtype=int), "y": cands["label"].astype(int).values})
    # 4) train cascade
    if torch is None:
        st.error("Torch not installed — cannot train cascade.")
        return {"error": "no_torch"}
    st.info("Training cascade (L1/L2/L3). This may take a while.")
    trader = CascadeTrader(seq_len=seq_len, feat_windows=(5,10,20), device=device_choice)
    try:
        trader.fit(bars, events, l2_use_xgb=(xgb is not None), epochs_l1=8, epochs_l23=8)
        st.success("Cascade trained.")
        st.session_state.trader = trader
    except Exception as e:
        logger.exception("Cascade training failed: %s", e)
        st.error(f"Cascade training failed: {e}")
        return {"error": "train_failed", "exception": str(e)}
    # 5) predict on candidate indices and breadth/backtest & optional sweep
    t_indices = np.array(cand_idx, dtype=int)
    preds = trader.predict_batch(bars, t_indices)
    st.write("Predictions head:")
    st.dataframe(preds.head(10))
    # breadth
    if run_breadth:
        st.info("Running breadth backtest")
        res = run_breadth_levels(preds, cands, bars)
        st.session_state.last_breadth = res
        if res["summary"]:
            st.subheader("Breadth summary")
            st.dataframe(pd.DataFrame(res["summary"]))
        else:
            st.warning("Breadth returned no summary rows.")
    # sweep
    if run_sweep:
        st.info("Running grid sweep (light)")
        rr_vals = [2.0, 2.5, 3.0]  # small set for speed
        sl_ranges = [(0.02, 0.04), (0.01, 0.03), (0.005, 0.02)]
        mpt_list = [p_fast, p_slow, p_deep]
        sweep_results = run_grid_sweep(cands, bars, rr_vals, sl_ranges, mpt_list)
        st.session_state.last_sweep = sweep_results
        st.success("Sweep completed.")
    # 6) export artifacts & models (per-level)
    st.info("Exporting artifacts and .pt bundles")
    out_dir = f"artifacts_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    os.makedirs(out_dir, exist_ok=True)
    # Save L1/L3 Torch state_dicts + scalers + meta
    try:
        trader_path = os.path.join(out_dir, "cascade_trader")
        os.makedirs(trader_path, exist_ok=True)
        # L1
        torch.save(trader.l1.state_dict(), os.path.join(trader_path, "l1_state.pt"))
        torch.save(trader.l3.state_dict(), os.path.join(trader_path, "l3_state.pt"))
        # If L2 is xgb, save model; if mlp, save state_dict
        if trader.l2_backend == "xgb":
            # xgboost wrapper saving
            try:
                trader.l2_model.save_model(os.path.join(trader_path, "l2_xgb.json"))
            except Exception:
                joblib.dump(trader.l2_model, os.path.join(trader_path, "l2_xgb.joblib"))
        else:
            torch.save(trader.l2_model.state_dict(), os.path.join(trader_path, "l2_mlp_state.pt"))
        # scalers and metadata
        with open(os.path.join(trader_path, "scaler_seq.pkl"), "wb") as f:
            pickle.dump(trader.scaler_seq, f)
        with open(os.path.join(trader_path, "scaler_tab.pkl"), "wb") as f:
            pickle.dump(trader.scaler_tab, f)
        with open(os.path.join(trader_path, "metadata.json"), "w") as f:
            json.dump(trader.metadata, f, default=str, indent=2)
    except Exception as e:
        logger.exception("Saving artifacts failed: %s", e)
        st.error(f"Saving artifacts failed: {e}")
    # use export_model_and_metadata to save a .pt bundle for L2 (as an example) and L3 wrapper
    export_paths = {}
    try:
        # create a wrapper-like object for export (minimal)
        l2_wrapper = type("L2W", (), {})()
        if trader.l2_backend == "xgb":
            l2_wrapper.booster = trader.l2_model
        else:
            l2_wrapper.booster = None
            # provide feature_importance stub if desired
            def fi_stub(): return pd.DataFrame([{"feature":"dummy","gain":1.0}])
            l2_wrapper.feature_importance = fi_stub
        paths_l2 = export_model_and_metadata(l2_wrapper, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time_sec")}, os.path.join(out_dir, "l2_model"), save_fi=True)
        export_paths["l2"] = paths_l2
        # export L3 as .pt bundle
        l3_wrapper = type("L3W", (), {})()
        def l3_fi(): return pd.DataFrame([{"feature":"l3_emb","gain":1.0}])
        l3_wrapper.feature_importance = l3_fi
        paths_l3 = export_model_and_metadata(l3_wrapper, trader.tab_feature_names, {"fit_time": trader.metadata.get("fit_time_sec")}, os.path.join(out_dir, "l3_model"), save_fi=True)
        export_paths["l3"] = paths_l3
        st.success(f"Exported models to {out_dir}")
        st.write(export_paths)
        st.session_state.export_paths = export_paths
    except Exception as e:
        logger.exception("Export_model_and_metadata failed: %s", e)
        st.error(f"Export failed: {e}")
    return {"bars": bars, "cands": cands, "trader": trader, "export": export_paths}

# If user clicked the full pipeline button:
if run_full_pipeline_btn:
    with st.spinner("Running full pipeline (fetch→cands→train→export) … this may take several minutes"):
        result = run_full_pipeline()
        if result is None:
            st.error("Pipeline failed with no result.")
        elif "error" in result:
            st.error(f"Pipeline error: {result.get('error')} - {result.get('exception','')}")
        else:
            st.success("Full pipeline completed.")
            st.write("Artifacts saved:", st.session_state.export_paths)
            # show final trades if breadth run was executed
            if run_breadth and "last_breadth" in st.session_state:
                br = st.session_state.last_breadth
                st.subheader("Breadth detailed results")
                for lvl, df in br["detailed"].items():
                    st.write(lvl, len(df))
                    if not df.empty:
                        st.dataframe(df.head(20))

# Chunk 8/8: final notes / help text

st.sidebar.markdown("### Notes\n- Level thresholds adjust the breadth backtest behaviors.\n- Breadth returns per-level trade lists and summaries.\n- Save/Log model after a successful training run.\n\nIf you encounter pickling/export issues for L2 wrappers, consider defining module-level wrapper classes (L2Wrapper/L3Wrapper) instead of dynamic types.")
```