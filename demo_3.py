# app_cfd_vs_futures.py
# Streamlit dashboard: concurrent fetch of Yahoo (futures) and Finage (CFD).
# Labels are generated from Yahoo; training performed on CFD limit-order outcomes.
# Interval for both = '1d'.

import os
import time
import json
import logging
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import requests
import aiohttp
import asyncio

# optional imports
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
except Exception:
    torch = None
    nn = None

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cfd_futures_dashboard")

st.set_page_config(page_title="CFD vs Futures — Label/Train", layout="wide")
st.title("CFD ↔ Futures: Label on Yahoo (futures), Train on Finage (CFD) — Interval = 1d")

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Data sources")
symbol_futures = st.sidebar.text_input("Futures symbol (Yahoo)", value="GC=F")
symbol_cfd = st.sidebar.text_input("CFD symbol (Finage)", value="GBPUSD")
default_days = 180
start_date = st.sidebar.date_input("Start date", value=(datetime.utcnow() - timedelta(days=default_days)).date())
end_date = st.sidebar.date_input("End date", value=datetime.utcnow().date())
interval = "1d"  # fixed to 1d per request

st.sidebar.header("Finage / CFD API")
finage_base = st.sidebar.text_input("Finage base URL (full endpoint root)", value=os.getenv("FINAGE_BASE_URL", "https://api.finage.co.uk"))
finage_api_key = st.sidebar.text_input("Finage API key", value=os.getenv("FINAGE_API_KEY", ""))

st.sidebar.header("Model / Training")
use_xgb_if_present = st.sidebar.checkbox("Prefer XGBoost if installed", value=True)
test_size = st.sidebar.slider("Validation fraction", 0.05, 0.4, 0.2)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42)
train_button = st.sidebar.button("Fetch → Label → Train (run)")

# -----------------------
# Utility functions
# -----------------------

def safe_fetch_yahoo(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV from yahoo via yahooquery. Returns dataframe or empty df."""
    if YahooTicker is None:
        logger.warning("yahooquery not installed.")
        return pd.DataFrame()
    try:
        t = YahooTicker(symbol)
        raw = t.history(start=start, end=end, interval=interval)
        if raw is None:
            return pd.DataFrame()
        if isinstance(raw, dict):
            raw = pd.DataFrame(raw)
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index(level=0, drop=True)
        raw.index = pd.to_datetime(raw.index)
        raw.columns = [c.lower() for c in raw.columns]
        if "close" not in raw.columns and "adjclose" in raw.columns:
            raw["close"] = raw["adjclose"]
        df = raw[~raw.index.duplicated(keep="first")].sort_index()
        return df
    except Exception as e:
        logger.exception("Yahoo fetch failed: %s", e)
        return pd.DataFrame()


# -----------------------
# New CFD fetch (Finage Forex Aggregates)
# -----------------------
async def fetch_cfd_finage(symbol: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """
    Fetch CFD (forex) OHLCV data from Finage via async HTTP.
    Interval fixed at 1 day.
    """
    url = f"https://api.finage.co.uk/agg/forex/{symbol}/1/day/{start}/{end}?apikey={api_key}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("Finage fetch failed with status %s", resp.status)
                    return pd.DataFrame()
                data = await resp.json()
                results = data.get("results", [])
                if not results:
                    return pd.DataFrame()
                df = pd.DataFrame(results)
                df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
                df = df.set_index("timestamp")
                df = df.sort_index()
                df = df[~df.index.duplicated(keep="first")]
                return df
        except Exception as e:
            logger.exception("Async Finage CFD fetch failed: %s", e)
            return pd.DataFrame()

async def fetch_data_concurrent(symbol_fut: str, symbol_cfd: str, start: str, end: str, api_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fut_task = asyncio.to_thread(safe_fetch_yahoo, symbol_fut, start, end, "1d")
    cfd_task = fetch_cfd_finage(symbol_cfd, start, end, api_key)
    futures, cfd = await asyncio.gather(fut_task, cfd_task)
    return futures, cfd


# -----------------------
# Existing feature / label / training helpers
# -----------------------
def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    for w in windows:
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
    return f.replace([np.inf,-np.inf],0.0).fillna(0.0)

def generate_candidates_and_labels_from_futures(bars: pd.DataFrame, lookback=64, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60):
    if bars is None or bars.empty: return pd.DataFrame()
    bars = bars.copy(); bars.index = pd.to_datetime(bars.index)
    bars['tr'] = (bars['high'] - bars['low']).abs().fillna(0.0)
    bars['atr'] = bars['tr'].rolling(atr_window, min_periods=1).mean()
    recs = []
    for i in range(lookback, len(bars)):
        entry_t = bars.index[i]
        px = float(bars['close'].iat[i])
        atr = float(bars['atr'].iat[i])
        if atr <= 0 or np.isnan(atr): continue
        sl_px = px - k_sl*atr
        tp_px = px + k_tp*atr
        end_i = min(i+max_bars, len(bars)-1)
        label = 0
        hit_i = end_i
        for j in range(i+1, end_i+1):
            hi = float(bars['high'].iat[j]); lo = float(bars['low'].iat[j])
            if hi >= tp_px:
                label = 1; hit_i = j; break
            if lo <= sl_px:
                label = 0; hit_i = j; break
        recs.append({"candidate_time": entry_t, "label": int(label), "entry_px": px, "hit_idx": hit_i})
    return pd.DataFrame(recs)

def align_labels_to_cfd(candidates: pd.DataFrame, cfd_bars: pd.DataFrame) -> pd.DataFrame:
    if candidates is None or candidates.empty or cfd_bars is None or cfd_bars.empty:
        return pd.DataFrame(), {}
    cfd_index_map = {t:i for i,t in enumerate(cfd_bars.index)}
    t_indices = []
    for t in pd.to_datetime(candidates['candidate_time']):
        if t in cfd_index_map:
            t_indices.append(cfd_index_map[t])
        else:
            locs = cfd_bars.index[cfd_bars.index <= t]
            t_indices.append(cfd_index_map[locs[-1]] if len(locs) else 0)
    events = pd.DataFrame({"t": np.array(t_indices, dtype=int), "y": candidates['label'].astype(int).values})
    return events, {"map": cfd_index_map}
def train_execution_model_on_cfd(cfd_bars: pd.DataFrame, events: pd.DataFrame, test_size=0.2, seed=42, prefer_xgb=True):
    if cfd_bars is None or cfd_bars.empty or events is None or events.empty:
        raise ValueError("No CFD bars or events for training.")
    eng = compute_engineered_features(cfd_bars, windows=(5,10,20))
    idx = events['t'].astype(int).values
    X = eng.values[idx]
    y = events['y'].astype(int).values
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y if len(np.unique(y))>1 else None)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    model = None; metrics = {}
    if prefer_xgb and xgb is not None:
        try:
            clf = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss")
            clf.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
            model = clf
            yhat = clf.predict_proba(X_va_s)[:,1]
            metrics['auc'] = float(roc_auc_score(y_va, yhat)) if len(np.unique(y_va))>1 else None
        except Exception as e:
            logger.exception("XGBoost training failed, falling back to LogisticRegression: %s", e)
    if model is None:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr_s, y_tr)
        model = clf
        try:
            yhat = clf.predict_proba(X_va_s)[:,1]
            metrics['auc'] = float(roc_auc_score(y_va, yhat)) if len(np.unique(y_va))>1 else None
        except Exception:
            metrics['auc'] = None
    ypred = (model.predict_proba(X_va_s)[:,1] >= 0.5).astype(int)
    metrics['accuracy'] = float(accuracy_score(y_va, ypred))
    return {"model": model, "scaler": scaler, "metrics": metrics, "X_va": X_va_s, "y_va": y_va}

# -----------------------
# UI: run pipeline
# -----------------------
status = st.empty()
col1, col2 = st.columns(2)
with col1:
    st.subheader("Futures (label source)")
    st.markdown("- Using Yahoo (futures) for labeling.")
    st.write("Symbol (Yahoo):", symbol_futures)
with col2:
    st.subheader("CFD (training / limit order outcomes)")
    st.markdown("- Using Finage CFD price series for training execution outcomes.")
    st.write("Symbol (Finage):", symbol_cfd)

def run_pipeline():
    status.info("Starting concurrent fetch of futures (Yahoo) and CFD (Finage)...")
    start_iso = pd.Timestamp(start_date).isoformat()
    end_iso = pd.Timestamp(end_date).isoformat()

    fut_df, cfd_df = asyncio.run(fetch_data_concurrent(symbol_futures, symbol_cfd, start_iso, end_iso, finage_api_key))

    if fut_df is None or fut_df.empty:
        status.error("Futures (Yahoo) fetch failed or returned no data.")
        return
    if cfd_df is None or cfd_df.empty:
        status.error("CFD (Finage) fetch failed or returned no data.")
        return

    status.success(f"Fetched futures ({len(fut_df)}) and CFD ({len(cfd_df)}) rows.")

    status.info("Generating candidate labels from futures...")
    candidates = generate_candidates_and_labels_from_futures(fut_df, lookback=32, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=30)
    if candidates.empty:
        status.warning("No candidates generated from futures.")
        return
    status.success(f"Generated {len(candidates)} candidates from futures.")

    events, mapping = align_labels_to_cfd(candidates, cfd_df)
    if events.empty:
        status.error("Failed to align labels to CFD bars (no matching timestamps).")
        return
    status.info(f"Aligned candidates -> CFD indices (training events={len(events)}).")

    status.info("Training execution model on CFD data...")
    try:
        train_res = train_execution_model_on_cfd(cfd_df, events, test_size=test_size, seed=int(random_state), prefer_xgb=(use_xgb_if_present and xgb is not None))
    except Exception as e:
        logger.exception("Training failed: %s", e)
        st.error(f"Training failed: {e}")
        return

    status.success("Training completed.")
    st.subheader("Training metrics")
    st.json(train_res["metrics"])

    # Save artifact
    artifact_dir = f"cfd_futures_artifacts_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    os.makedirs(artifact_dir, exist_ok=True)
    try:
        import joblib
        joblib.dump({
            "model": train_res["model"],
            "scaler": train_res["scaler"],
            "metrics": train_res["metrics"],
            "symbol_cfd": symbol_cfd,
            "symbol_futures": symbol_futures,
            "created_at": datetime.utcnow().isoformat()
        }, os.path.join(artifact_dir, "execution_model.joblib"))
        st.success(f"Saved artifact to {artifact_dir}")
    except Exception as e:
        logger.exception("Failed to save artifact: %s", e)
        st.warning(f"Failed to save artifact: {e}")

if train_button:
    try:
        run_pipeline()
    except Exception as e:
        logger.exception("Unhandled error in pipeline: %s", e)
        st.error(f"Unhandled error: {e}")