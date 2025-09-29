# Chunk 1/8: imports + Streamlit UI + module-level wrappers + robust exporter
import os
import io
import math
import time
import uuid
import json
import joblib
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
import concurrent.futures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass

# ML / metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Optional libs (graceful degradation)
try:
    from yahooquery import Ticker as YahooTicker
except Exception:
    YahooTicker = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("entry_triangulation_app")
logger.setLevel(logging.INFO)

st.set_page_config(page_title="Entry-Range Triangulation", layout="wide")
st.title("Entry-Range Triangulation Dashboard — Multi-level (3) Models")

# Module-level small wrapper classes for export pickling
class L2Wrapper:
    """Wrapper for L2 model export (xgboost or MLP)."""
    def __init__(self):
        self.booster = None  # xgboost.Booster or None
        self.model = None    # torch model fallback
        self.feature_names = []

    def predict_proba(self, X: pd.DataFrame):
        if self.booster is not None:
            d = xgb.DMatrix(X[self.feature_names].fillna(0.0))
            return self.booster.predict(d)
        if self.model is not None:
            self.model.eval()
            with torch.no_grad():
                Xt = torch.tensor(X.values.astype(np.float32))
                logits = self.model(Xt).numpy().reshape(-1)
                return 1.0 / (1.0 + np.exp(-logits))
        return np.zeros(len(X))

    def feature_importance(self) -> pd.DataFrame:
        if self.booster is not None:
            imp = self.booster.get_score(importance_type="gain")
            return pd.DataFrame([(k, float(imp.get(k, 0.0))) for k in self.feature_names],
                                columns=["feature", "gain"]).sort_values("gain", ascending=False)
        return pd.DataFrame([{"feature": "none", "gain": 0.0}])

class L3Wrapper:
    """Wrapper for L3 model export (torch)."""
    def __init__(self):
        self.model = None
        self.input_features = []

    def predict_proba(self, X: pd.DataFrame):
        if self.model is None:
            return np.zeros(len(X))
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X.values.astype(np.float32))
            logits, _ = self.model(Xt)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            return probs

    def feature_importance(self) -> pd.DataFrame:
        # For neural networks we fallback to empty / placeholder
        return pd.DataFrame([{"feature": f, "gain": 0.0} for f in self.input_features])

# Use the robust exporter you provided (kept verbatim with logging)
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
                model_wrapper.save_model(model_file) if hasattr(model_wrapper, "save_model") else model_wrapper.booster.save_model(model_file)
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

            # Add initial log entry
            payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Starting export of .pt bundle")

            # Check torch availability
            if 'torch' in globals() and torch is not None:
                payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Using torch.save for .pt export")
                torch.save(payload, pt_path)
            else:
                payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] torch not available, using joblib for .pt export")
                joblib.dump(payload, pt_path)

            # Add success log entry
            payload["export_log"].append(f"[{datetime.utcnow().isoformat()}] Successfully saved .pt bundle")

            paths["pt"] = pt_path
            logger.info(f"Successfully saved .pt bundle to {pt_path}")
        except Exception as e:
            error_msg = f"Failed to save .pt payload: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            paths["pt_error"] = error_msg

            # Try fallback with more detailed error logging
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

    # Final validation of PT file
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

# Chunk 2/8: fetching + rvol + health gauge + index helpers

def fetch_price(symbol: str, start: Optional[str] = None, end: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
    """YahooQuery OHLCV fetcher returning a tidy DataFrame."""
    logger.info("fetch_price: symbol=%s start=%s end=%s interval=%s", symbol, start, end, interval)
    if YahooTicker is None:
        logger.error("yahooquery missing.")
        return pd.DataFrame()
    try:
        tq = YahooTicker(symbol)
        raw = tq.history(start=start, end=end, interval=interval)
        if raw is None:
            logger.warning("fetch_price returned no data")
            return pd.DataFrame()
        if isinstance(raw, dict):
            raw = pd.DataFrame(raw)
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index(level=0, drop=True)
        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index()
        raw.columns = [c.lower() for c in raw.columns]
        if "close" not in raw.columns and "adjclose" in raw.columns:
            raw["close"] = raw["adjclose"]
        raw = raw[~raw.index.duplicated(keep="first")]
        logger.info("fetch_price: returned %d rows", len(raw))
        return raw
    except Exception as exc:
        logger.exception("fetch_price failed: %s", exc)
        return pd.DataFrame()

def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype=float)
    if "volume" not in df.columns:
        # fallback constant
        return pd.Series(1.0, index=df.index)
    rolling = df["volume"].rolling(window=lookback, min_periods=1).mean()
    r = (df["volume"] / rolling.replace(0, np.nan)).fillna(1.0)
    return r

def calculate_health_gauge(cot_df: pd.DataFrame, daily_bars: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    if daily_bars is None or daily_bars.empty:
        return pd.DataFrame()
    db = daily_bars.copy()
    db["rvol"] = compute_rvol(db)
    db["health_gauge"] = (db["rvol"] >= threshold).astype(float)
    return db[["health_gauge"]]

def ensure_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    df = df.copy()
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# Chunk 3/8: candidate generation & labeling (fixed tp_px expression)

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def generate_candidates_and_labels(bars: pd.DataFrame,
                                   lookback: int = 64,
                                   k_tp: float = 3.0,
                                   k_sl: float = 1.0,
                                   atr_window: int = 14,
                                   max_bars: int = 60,
                                   direction: str = "long") -> pd.DataFrame:
    """
    ATR-based TP/SL labeling. Ensures tp_px expression is correct.
    """
    if bars is None or bars.empty:
        logger.warning("generate_candidates_and_labels: empty bars")
        return pd.DataFrame()
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    bars = ensure_unique_index(bars)
    for col in ("high", "low", "close"):
        if col not in bars.columns:
            raise KeyError(f"Missing column {col}")
    bars["tr"] = _true_range(bars["high"], bars["low"], bars["close"])
    bars["atr"] = bars["tr"].rolling(window=atr_window, min_periods=1).mean()

    recs = []
    n = len(bars)
    for i in range(lookback, n):
        t = bars.index[i]
        entry_px = float(bars["close"].iat[i])
        atr = float(bars["atr"].iat[i])
        if atr <= 0 or math.isnan(atr):
            continue

        if direction == "long":
            sl_px = entry_px - k_sl * atr
            tp_px = entry_px + k_tp * atr
        else:
            sl_px = entry_px + k_sl * atr
            tp_px = entry_px - k_tp * atr

        end_idx = min(i + max_bars, n - 1)
        label = 0
        hit_idx = end_idx
        hit_px = float(bars["close"].iat[end_idx])

        for j in range(i + 1, end_idx + 1):
            hi = float(bars["high"].iat[j]); lo = float(bars["low"].iat[j])
            if direction == "long":
                if hi >= tp_px:
                    label, hit_idx, hit_px = 1, j, tp_px
                    break
                if lo <= sl_px:
                    label, hit_idx, hit_px = 0, j, sl_px
                    break
            else:
                if lo <= tp_px:
                    label, hit_idx, hit_px = 1, j, tp_px
                    break
                if hi >= sl_px:
                    label, hit_idx, hit_px = 0, j, sl_px
                    break

        end_t = bars.index[hit_idx]
        realized_return = (hit_px - entry_px) / entry_px if direction == "long" else (entry_px - hit_px) / entry_px
        dur_min = (end_t - t).total_seconds() / 60.0

        recs.append({
            "candidate_time": t,
            "entry_price": float(entry_px),
            "atr": float(atr),
            "sl_price": float(sl_px),
            "tp_price": float(tp_px),
            "end_time": end_t,
            "label": int(label),
            "duration": float(dur_min),
            "realized_return": float(realized_return),
            "direction": direction
        })
    df_recs = pd.DataFrame(recs)
    logger.info("generate_candidates_and_labels: generated %d candidates", len(df_recs))
    return df_recs

# Chunk 4/8: simulate_limits + summaries

def simulate_limits(df: pd.DataFrame,
                    bars: pd.DataFrame,
                    label_col: str = "pred_label",
                    symbol: str = "GC=F",
                    sl: float = 0.02,
                    tp: float = 0.04,
                    max_holding: int = 60) -> pd.DataFrame:
    if df is None or df.empty or bars is None or bars.empty:
        logger.warning("simulate_limits: empty inputs")
        return pd.DataFrame()
    trades = []
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
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
                    exit_t, exit_px, pnl = t, sl_px, -sl
                    break
                if hi >= tp_px:
                    exit_t, exit_px, pnl = t, tp_px, tp
                    break
            else:
                if hi >= sl_px:
                    exit_t, exit_px, pnl = t, sl_px, -sl
                    break
                if lo <= tp_px:
                    exit_t, exit_px, pnl = t, tp_px, tp
                    break
        if exit_t is None:
            last_bar = segment.iloc[-1]
            exit_t = last_bar.name
            exit_px = float(last_bar["close"])
            pnl = (exit_px - entry_px) / entry_px * direction
        trades.append({
            "symbol": symbol,
            "entry_time": entry_t,
            "entry_price": entry_px,
            "direction": direction,
            "exit_time": exit_t,
            "exit_price": exit_px,
            "pnl": float(pnl)
        })
    df_trades = pd.DataFrame(trades)
    logger.info("simulate_limits: %d trades generated", len(df_trades))
    return df_trades

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

def combine_summaries(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for mode, trades_df in results.items():
        if trades_df is None or trades_df.empty:
            continue
        s = summarize_trades(trades_df)
        if not s.empty:
            s["mode"] = mode
            rows.append(s)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

# Chunk 5/8: BoosterWrapper + train_xgb_confirm (expanded metrics) + predict_confirm_prob

class BoosterWrapper:
    def __init__(self, booster: xgb.Booster, feature_names: List[str]):
        self.booster = booster
        self.feature_names = feature_names
        self.best_iteration = getattr(booster, "best_iteration", None)

    def _dmatrix(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        Xp = X.reindex(columns=self.feature_names).fillna(0.0)
        return xgb.DMatrix(Xp, feature_names=self.feature_names)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        d = self._dmatrix(X)
        if self.best_iteration is not None:
            raw = self.booster.predict(d, iteration_range=(0, int(self.best_iteration) + 1))
        else:
            raw = self.booster.predict(d)
        raw = np.asarray(raw, dtype=float)
        return pd.Series(raw, index=X.index, name="confirm_proba")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def save_model(self, path: str):
        try:
            self.booster.save_model(path)
        except Exception:
            joblib.dump({"booster": self.booster, "feature_names": self.feature_names}, path)

    def feature_importance(self) -> pd.DataFrame:
        try:
            importance = self.booster.get_score(importance_type="gain")
            df = pd.DataFrame([(k, importance.get(k, 0.0)) for k in self.feature_names],
                              columns=["feature", "importance"]).sort_values("importance", ascending=False)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame([(f, 0.0) for f in self.feature_names], columns=["feature", "importance"])

def train_xgb_confirm(clean: pd.DataFrame,
                      feature_cols: List[str],
                      label_col: str = "label",
                      num_boost_round: int = 200,
                      early_stopping_rounds: int = 20,
                      test_size: float = 0.2,
                      random_state: int = 42,
                      verbose: bool = False) -> (BoosterWrapper, Dict[str,Any]):
    if xgb is None:
        raise RuntimeError("xgboost not installed")

    missing = [c for c in feature_cols if c not in clean.columns]
    if missing:
        raise KeyError(f"Missing feature columns in training data: {missing}")

    X_all = clean[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_all = pd.to_numeric(clean[label_col], errors="coerce").astype(int)

    mask = X_all.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1) & y_all.notnull()
    X_all = X_all.loc[mask].reset_index(drop=True)
    y_all = y_all.loc[mask].reset_index(drop=True)

    if len(X_all) < 10 or y_all.nunique() < 2:
        raise ValueError("Not enough data or not both classes present for training")

    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "verbosity": 0,
        "scale_pos_weight": float((y_train == 0).sum()) / max(1, float((y_train == 1).sum()))
    }

    t0 = time.time()
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dval, "validation")],
                    early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose)
    fit_time = time.time() - t0

    wrapper = BoosterWrapper(bst, feature_cols)

    # Metrics
    y_proba_val = wrapper.predict_proba(X_val)
    y_pred_val = (y_proba_val >= 0.5).astype(int)

    acc = float(accuracy_score(y_val, y_pred_val))
    f1 = float(f1_score(y_val, y_pred_val, zero_division=0))
    try:
        roc = float(roc_auc_score(y_val, y_proba_val))
    except Exception:
        roc = float("nan")

    cm = confusion_matrix(y_val, y_pred_val).tolist()
    creport = classification_report(y_val, y_pred_val, zero_division=0, output_dict=False)

    metrics = {
        "fit_time_sec": round(fit_time, 4),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc,
        "confusion_matrix": cm,
        "classification_report": creport
    }

    logger.info("train_xgb_confirm: metrics=%s", metrics)
    return wrapper, metrics

def predict_confirm_prob(model, df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    missing = [c for c in feature_cols if c not in df.columns]
    for m in missing:
        df[m] = 0.0
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(df[feature_cols])
        elif hasattr(model, "predict"):
            preds = model.predict(df[feature_cols])
            return pd.Series(preds, index=df.index, name="confirm_proba")
        else:
            return pd.Series(np.zeros(len(df)), index=df.index, name="confirm_proba")
    except Exception as exc:
        logger.exception("predict_confirm_prob failed: %s", exc)
        return pd.Series(np.zeros(len(df)), index=df.index, name="confirm_proba")

# Chunk 6/8: run_breadth_backtest + run_grid_sweep (with optional per-level concurrent training)

def run_breadth_backtest(clean: pd.DataFrame,
                         bars: pd.DataFrame,
                         levels_config: Dict[str, Dict[str, Any]],
                         feature_cols: List[str],
                         model_train_kwargs: Dict[str,Any],
                         train_per_level: bool = False,
                         concurrent_workers: int = 3) -> Dict[str, Any]:
    """
    For each level in levels_config produce trades. Optionally retrain a confirm model per level.
    Returns standardized dict with summaries, per-level trades, metrics and export paths.
    """
    out = {"summary": [], "detailed_trades": {}, "metrics": {}, "export_paths": {}, "diagnostics": []}
    if clean is None or clean.empty:
        out["diagnostics"].append("No candidate set provided.")
        return out

    # Prepare thread pool if requested
    executor = None
    if train_per_level:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers)
        futures = {}

    for lvl_name, cfg in levels_config.items():
        try:
            buy_th = float(cfg.get("buy_th"))
            sell_th = float(cfg.get("sell_th"))
            rr_min = float(cfg.get("rr_min"))
            rr_max = float(cfg.get("rr_max"))
            sl_min = float(cfg.get("sl_min"))
            sl_max = float(cfg.get("sl_max"))

            df = clean.copy()

            if "signal" not in df.columns:
                if "pred_prob" in df.columns:
                    df["signal"] = (df["pred_prob"] * 10).round().astype(int)
                else:
                    df["signal"] = 0

            df["pred_label"] = 0
            df.loc[df["signal"] > buy_th, "pred_label"] = 1
            df.loc[df["signal"] < sell_th, "pred_label"] = -1

            sl_pct = float((sl_min + sl_max) / 2.0)
            rr = float((rr_min + rr_max) / 2.0)
            tp_pct = rr * sl_pct

            # Optionally retrain per-level on filtered df where label in {0,1}
            if train_per_level:
                # spawn training in thread using filtered positives/negatives
                df_train = df[df["label"].isin([0,1])].reset_index(drop=True)
                if df_train.shape[0] >= 20:
                    future = executor.submit(train_xgb_confirm, df_train, feature_cols,
                                             model_train_kwargs.get("label_col", "label"),
                                             int(model_train_kwargs.get("num_boost_round", 200)),
                                             int(model_train_kwargs.get("early_stopping_rounds", 20)),
                                             float(model_train_kwargs.get("test_size", 0.2)),
                                             int(model_train_kwargs.get("random_state", 42)),
                                             False)
                    futures[lvl_name] = future
                else:
                    out["diagnostics"].append(f"{lvl_name}: not enough rows to train per-level model ({len(df_train)})")
                    futures[lvl_name] = None
            else:
                # Use existing pred_label and no retrain
                trades = simulate_limits(df, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=int(model_train_kwargs.get("max_bars", 60)))
                out["detailed_trades"][lvl_name] = trades
                if not trades.empty:
                    s = summarize_trades(trades)
                    out["summary"].append(s.iloc[0].to_dict())
                out["diagnostics"].append(f"{lvl_name}: simulated {len(trades)} trades, sl={sl_pct:.4f}, tp={tp_pct:.4f}")

        except Exception as exc:
            logger.exception("Breadth level %s failed: %s", lvl_name, exc)
            out["diagnostics"].append(f"{lvl_name} error: {exc}")

    # Wait for training futures and if present, simulate and export
    if train_per_level and futures:
        for lvl_name, fut in futures.items():
            try:
                if fut is None:
                    continue
                model_wrap, metrics = fut.result(timeout=600)
                out["metrics"][lvl_name] = metrics
                # Build a wrapper for export
                wrapper = L2Wrapper()
                wrapper.booster = model_wrap.booster if hasattr(model_wrap, "booster") else None
                wrapper.feature_names = model_wrap.feature_names if hasattr(model_wrap, "feature_names") else feature_cols
                # simulate using model's predicted probs
                df_local = clean.copy()
                df_local["pred_prob"] = predict_confirm_prob(wrapper, df_local, feature_cols)
                df_local["pred_label"] = (df_local["pred_prob"] >= model_train_kwargs.get("mpt", 0.6)).astype(int)
                sl_pct = float((levels_config[lvl_name]["sl_min"] + levels_config[lvl_name]["sl_max"]) / 2.0)
                rr = float((levels_config[lvl_name]["rr_min"] + levels_config[lvl_name]["rr_max"]) / 2.0)
                tp_pct = rr * sl_pct
                trades = simulate_limits(df_local, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=int(model_train_kwargs.get("max_bars", 60)))
                out["detailed_trades"][lvl_name] = trades
                if not trades.empty:
                    s = summarize_trades(trades)
                    out["summary"].append(s.iloc[0].to_dict())

                # Export model
                model_basename = f"confirm_{lvl_name}_{uuid.uuid4().hex[:8]}"
                paths = export_model_and_metadata(model_wrap, feature_cols, metrics, model_basename, save_fi=True)
                out["export_paths"][lvl_name] = paths
                out["diagnostics"].append(f"{lvl_name}: trained & exported, trades={len(trades)}")
            except Exception as exc:
                logger.exception("Per-level training failed for %s: %s", lvl_name, exc)
                out["diagnostics"].append(f"{lvl_name} training error: {exc}")

    if executor:
        executor.shutdown(wait=False)
    return out

def run_grid_sweep(clean: pd.DataFrame,
                   bars: pd.DataFrame,
                   rr_vals: List[float],
                   sl_ranges: List[Tuple[float,float]],
                   mpt_list: List[float],
                   feature_cols: List[str],
                   model_train_kwargs: Dict[str,Any]) -> Dict[str, Any]:
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
                        if "signal" in df.columns:
                            df["pred_prob"] = (df["signal"] / 10.0).clip(0.0, 1.0)
                        else:
                            df["pred_prob"] = 0.0
                    df["pred_label"] = (df["pred_prob"] >= mpt).astype(int)
                    trades = simulate_limits(df, bars, label_col="pred_label", sl=sl_pct, tp=tp_pct, max_holding=int(model_train_kwargs.get("max_bars", 60)))
                    results[key] = {"trades_count": len(trades), "overlay": trades}
                except Exception as exc:
                    logger.exception("Sweep config %s failed: %s", key, exc)
                    results[key] = {"error": str(exc)}
    return results

# Chunk 7/8: Supabase logger + helpers + pick_top_runs_by_metrics

class SupabaseLogger:
    def __init__(self):
        if create_client is None:
            raise RuntimeError("supabase client not installed")
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE env vars missing")
        self.client = create_client(url, key)
        self.runs_tbl = "entry_runs"
        self.trades_tbl = "entry_trades"

    def log_run(self, metrics: Dict, metadata: Dict, trades: Optional[List[Dict]] = None) -> str:
        run_id = metadata.get("run_id") or str(uuid.uuid4())
        metadata["run_id"] = run_id
        payload = {**metadata, "metrics": metrics}
        resp = self.client.table(self.runs_tbl).insert(payload).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Failed to insert run: {resp.error}")
        if trades:
            for t in trades:
                t["run_id"] = run_id
            tr_resp = self.client.table(self.trades_tbl).insert(trades).execute()
            if getattr(tr_resp, "error", None):
                raise RuntimeError(f"Failed to insert trades: {tr_resp.error}")
        return run_id

    def fetch_runs(self, symbol: Optional[str] = None, limit: int = 50):
        q = self.client.table(self.runs_tbl)
        if symbol:
            q = q.eq("symbol", symbol)
        resp = q.order("start_date", desc=True).limit(limit).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Failed to fetch runs: {resp.error}")
        return getattr(resp, "data", [])

    def fetch_trades(self, run_id: str):
        resp = self.client.table(self.trades_tbl).select("*").eq("run_id", run_id).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Failed to fetch trades: {resp.error}")
        return getattr(resp, "data", [])

def pick_top_runs_by_metrics(runs: List[Dict], top_n: int = 3):
    scored = []
    for r in runs:
        metrics = r.get("metrics") if isinstance(r, dict) else r
        total_pnl = metrics.get("total_pnl", 0.0) if metrics else 0.0
        win_rate = metrics.get("win_rate", 0.0) if metrics else 0.0
        score = float(total_pnl) + float(win_rate) * 0.01
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_n]]

# Chunk 8/8: main pipeline + handlers (fetch → features → label → train → backtest → export)

# --- UI controls (kept / extended)
st.sidebar.header("HealthGauge")
buy_threshold_global = st.sidebar.number_input("Global buy threshold (signal scale 0-10)", 0.0, 10.0, 5.5)
sell_threshold_global = st.sidebar.number_input("Global sell threshold (signal scale 0-10)", 0.0, 10.0, 4.5)
force_run = st.sidebar.checkbox("Force run even if gating fails", value=False)

st.sidebar.header("Training / Export")
num_boost = int(st.sidebar.number_input("XGBoost rounds", min_value=1, value=200))
early_stop = int(st.sidebar.number_input("Early stopping rounds", min_value=1, value=20))
test_size = float(st.sidebar.number_input("Validation fraction", min_value=0.01, max_value=0.5, value=0.2))
train_per_level = st.sidebar.checkbox("Train per-level confirm models (concurrent)", value=False)
concurrent_workers = int(st.sidebar.number_input("Concurrent workers (if per-level)", 1, 8, 3))

st.sidebar.header("Confirm thresholds")
p_fast = st.sidebar.number_input("Confirm threshold (fast)", 0.0, 1.0, 0.60)
p_slow = st.sidebar.number_input("Confirm threshold (slow)", 0.0, 1.0, 0.55)
p_deep = st.sidebar.number_input("Confirm threshold (deep)", 0.0, 1.0, 0.45)

st.sidebar.header("Levels (explicit min/max scope)")
# Level 1
lvl1_buy_min = st.sidebar.number_input("L1 buy min", 0.0, 10.0, 5.5)
lvl1_buy_max = st.sidebar.number_input("L1 buy max", 0.0, 10.0, 6.0)
lvl1_sell_min = st.sidebar.number_input("L1 sell min", 0.0, 10.0, 4.5)
lvl1_sell_max = st.sidebar.number_input("L1 sell max", 0.0, 10.0, 5.0)
lvl1_rr_min = st.sidebar.number_input("L1 RR min", 0.1, 10.0, 1.0)
lvl1_rr_max = st.sidebar.number_input("L1 RR max", 0.1, 10.0, 2.5)
lvl1_sl_min = st.sidebar.number_input("L1 SL min (pct)", 0.001, 0.5, 0.02)
lvl1_sl_max = st.sidebar.number_input("L1 SL max (pct)", 0.001, 0.5, 0.04)

# Level 2
lvl2_buy_min = st.sidebar.number_input("L2 buy min", 0.0, 10.0, 6.0)
lvl2_buy_max = st.sidebar.number_input("L2 buy max", 0.0, 10.0, 6.5)
lvl2_sell_min = st.sidebar.number_input("L2 sell min", 0.0, 10.0, 4.0)
lvl2_sell_max = st.sidebar.number_input("L2 sell max", 0.0, 10.0, 4.5)
lvl2_rr_min = st.sidebar.number_input("L2 RR min", 0.1, 10.0, 2.0)
lvl2_rr_max = st.sidebar.number_input("L2 RR max", 0.1, 10.0, 3.5)
lvl2_sl_min = st.sidebar.number_input("L2 SL min (pct)", 0.001, 0.5, 0.01)
lvl2_sl_max = st.sidebar.number_input("L2 SL max (pct)", 0.001, 0.5, 0.03)

# Level 3
lvl3_buy_min = st.sidebar.number_input("L3 buy min", 0.0, 10.0, 6.5)
lvl3_buy_max = st.sidebar.number_input("L3 buy max", 0.0, 10.0, 7.5)
lvl3_sell_min = st.sidebar.number_input("L3 sell min", 0.0, 10.0, 3.5)
lvl3_sell_max = st.sidebar.number_input("L3 sell max", 0.0, 10.0, 4.0)
lvl3_rr_min = st.sidebar.number_input("L3 RR min", 0.1, 10.0, 3.0)
lvl3_rr_max = st.sidebar.number_input("L3 RR max", 0.1, 10.0, 5.0)
lvl3_sl_min = st.sidebar.number_input("L3 SL min (pct)", 0.001, 0.5, 0.005)
lvl3_sl_max = st.sidebar.number_input("L3 SL max (pct)", 0.001, 0.5, 0.02)

# Buttons
run_full = st.button("Run full pipeline (fetch → features → candidates → train → backtest → export)")
run_breadth = st.button("Run breadth backtest (3 levels only)")
run_sweep = st.button("Run grid sweep (RR x SL x MPT)")

# defaults
feat_cols_default = ["atr", "rvol", "duration", "hg"]

def _assemble_levels_config():
    return {
        "L1": {"buy_th": (lvl1_buy_min + lvl1_buy_max)/2.0, "sell_th": (lvl1_sell_min + lvl1_sell_max)/2.0,
               "rr_min": lvl1_rr_min, "rr_max": lvl1_rr_max, "sl_min": lvl1_sl_min, "sl_max": lvl1_sl_max},
        "L2": {"buy_th": (lvl2_buy_min + lvl2_buy_max)/2.0, "sell_th": (lvl2_sell_min + lvl2_sell_max)/2.0,
               "rr_min": lvl2_rr_min, "rr_max": lvl2_rr_max, "sl_min": lvl2_sl_min, "sl_max": lvl2_sl_max},
        "L3": {"buy_th": (lvl3_buy_min + lvl3_buy_max)/2.0, "sell_th": (lvl3_sell_min + lvl3_sell_max)/2.0,
               "rr_min": lvl3_rr_min, "rr_max": lvl3_rr_max, "sl_min": lvl3_sl_min, "sl_max": lvl3_sl_max},
    }

def run_main_pipeline_and_export():
    st.info("Starting full pipeline...")
    try:
        bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
        if bars is None or bars.empty:
            st.error("No price data returned.")
            return
        bars = ensure_unique_index(bars)
        # compute rvol and attach
        bars["rvol"] = compute_rvol(bars, lookback=20)
        daily = bars.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        health = calculate_health_gauge(None, daily)
        latest_health = float(health["health_gauge"].iloc[-1]) if not health.empty else 0.0
        st.metric("Latest HealthGauge", f"{latest_health:.3f}")

        if not (latest_health >= buy_threshold_global or latest_health <= sell_threshold_global or force_run):
            st.warning("Health gating prevented run. Use 'Force run' to override.")
            return

        # generate candidates
        cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
        if cands is None or cands.empty:
            st.error("No candidates generated.")
            return

        # ensure rvol mapped into candidates
        if "rvol" not in cands.columns:
            # map via candidate_time to bar rvol
            cands["rvol"] = cands["candidate_time"].map(lambda t: bars["rvol"].get(pd.to_datetime(t), np.nan))
        cands["rvol"] = cands["rvol"].fillna(1.0)

        # health gauge
        if not health.empty:
            hg_map = health["health_gauge"].reindex(pd.to_datetime(health.index)).ffill().to_dict()
            cands["hg"] = cands["candidate_time"].dt.normalize().map(lambda t: hg_map.get(pd.Timestamp(t).normalize(), 0.0))
        else:
            cands["hg"] = 0.0

        feat_cols = ["atr", "rvol", "duration", "hg"]
        # filter labels
        clean = cands.dropna(subset=["label"]).query("label in [0,1]").reset_index(drop=True)
        if clean.empty:
            st.error("No labelled candidates for training.")
            return

        # Train main confirm model (single model)
        st.info("Training confirm-stage XGBoost model (main pipeline)...")
        model_wrap, metrics = train_xgb_confirm(clean, feat_cols, label_col="label",
                                                num_boost_round=num_boost, early_stopping_rounds=early_stop,
                                                test_size=test_size, random_state=42, verbose=False)
        st.subheader("Training metrics")
        st.json(metrics)

        # predict and simulate
        clean["pred_prob"] = predict_confirm_prob(model_wrap, clean, feat_cols)
        clean["pred_label"] = (clean["pred_prob"] >= p_fast).astype(int)
        trades = simulate_limits(clean, bars, label_col="pred_label", max_holding=60)
        st.write("Simulated trades:", len(trades))
        if not trades.empty:
            st.dataframe(trades.head())
            s = summarize_trades(trades)
            st.dataframe(s)

        # Export main model
        model_basename = f"confirm_main_{symbol.replace('=','_')}"
        paths_main = export_model_and_metadata(model_wrap, feat_cols, metrics, model_basename, save_fi=True)
        st.success(f"Exported main model: {paths_main}")

        # Optionally run breadth per-level with per-level training & export
        levels_cfg = _assemble_levels_config()
        if train_per_level or st.checkbox("Also run breadth backtest + per-level export now", value=False):
            st.info("Running breadth backtest with per-level training & export...")
            res = run_breadth_backtest(clean=clean, bars=bars, levels_config=levels_cfg,
                                      feature_cols=feat_cols,
                                      model_train_kwargs={"num_boost_round": num_boost, "early_stopping_rounds": early_stop, "test_size": test_size, "max_bars": 60, "mpt": p_fast},
                                      train_per_level=train_per_level,
                                      concurrent_workers=concurrent_workers)
            st.subheader("Breadth diagnostics")
            st.write(res.get("diagnostics", []))
            if res.get("summary"):
                st.subheader("Breadth summary")
                st.dataframe(pd.DataFrame(res["summary"]))
            if res.get("export_paths"):
                st.subheader("Per-level export paths")
                st.json(res["export_paths"])

        st.success("Full pipeline complete.")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        st.error(f"Pipeline failed: {exc}")

if run_full:
    run_main_pipeline_and_export()

# Breadth-only button (uses same candidate generation)
if run_breadth:
    try:
        bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
        bars = ensure_unique_index(bars)
        bars["rvol"] = compute_rvol(bars, 20)
        cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
        if cands is None or cands.empty:
            st.error("No candidates available for breadth run.")
        else:
            # ensure rvol/hg
            cands["rvol"] = cands["candidate_time"].map(lambda t: bars["rvol"].get(pd.to_datetime(t), np.nan)).fillna(1.0)
            levels_cfg = _assemble_levels_config()
            res = run_breadth_backtest(clean=cands, bars=bars, levels_config=levels_cfg,
                                      feature_cols=feat_cols_default, model_train_kwargs={"max_bars": 60, "mpt": p_fast},
                                      train_per_level=train_per_level, concurrent_workers=concurrent_workers)
            st.subheader("Breadth diagnostics")
            st.write(res.get("diagnostics", []))
            if res.get("summary"):
                st.subheader("Breadth summary")
                st.dataframe(pd.DataFrame(res["summary"]))
    except Exception as exc:
        logger.exception("Breadth failed: %s", exc)
        st.error(f"Breadth failed: {exc}")

# Sweep button
if run_sweep:
    try:
        bars = fetch_price(symbol, start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
        bars = ensure_unique_index(bars)
        bars["rvol"] = compute_rvol(bars, 20)
        cands = generate_candidates_and_labels(bars, k_tp=2.0, k_sl=1.0, atr_window=14, max_bars=60)
        if cands is None or cands.empty:
            st.error("No candidates for sweep.")
        else:
            cands["rvol"] = cands["candidate_time"].map(lambda t: bars["rvol"].get(pd.to_datetime(t), np.nan)).fillna(1.0)
            rr_vals = sorted(list(set([lvl1_rr_min, lvl1_rr_max, lvl2_rr_min, lvl2_rr_max, lvl3_rr_min, lvl3_rr_max])))
            sl_ranges = [(lvl1_sl_min, lvl1_sl_max), (lvl2_sl_min, lvl2_sl_max), (lvl3_sl_min, lvl3_sl_max)]
            mpt_list = [p_fast, p_slow, p_deep]
            sweep_results = run_grid_sweep(clean=cands, bars=bars, rr_vals=rr_vals, sl_ranges=sl_ranges, mpt_list=mpt_list,
                                          feature_cols=feat_cols_default, model_train_kwargs={"max_bars": 60})
            summary_rows = []
            for k, v in sweep_results.items():
                if isinstance(v, dict) and "overlay" in v and not v["overlay"].empty:
                    s = summarize_trades(v["overlay"])
                    if not s.empty:
                        r = s.iloc[0].to_dict()
                        r.update({"config": k, "trades_count": v.get("trades_count", 0)})
                        summary_rows.append(r)
            if summary_rows:
                sdf = pd.DataFrame(summary_rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)
                st.subheader("Sweep summary (top configs)")
                st.dataframe(sdf.head(20))
                st.download_button("Download sweep summary CSV", df_to_csv_bytes(sdf), "sweep_summary.csv", "text/csv")
            else:
                st.warning("Sweep returned no runs with trades.")
    except Exception as exc:
        logger.exception("Sweep failed: %s", exc)
        st.error(f"Sweep failed: {exc}")

st.sidebar.markdown("### Notes\n- Full pipeline fetches data, computes features, generates candidates, trains confirm model, backtests, and exports artifacts.\n- Enable 'Train per-level' to train and export one confirm model per level concurrently.\n- Exported .pt metadata contains full training metrics (accuracy, f1, roc_auc, confusion matrix, fit_time).")