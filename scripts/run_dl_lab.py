"""
Deep Learning Laboratory — run_dl_lab.py
=========================================
Trains MLP, CNN1D, LSTM, and PatchTST across all 48 assets with a
hyperparameter grid sweep.  Results are saved incrementally to a JSON file.

What this script teaches you
-----------------------------
Optimizer comparison  : SGD vs Adam on every asset.
  → SGD: vanilla gradient descent with optional momentum.
    Slow, noisy, but generalises well when tuned correctly.
  → Adam: adaptive per-parameter learning rates (Kingma & Ba, 2015).
    Converges ~10× faster on financial time series.
    You will see this empirically in the loss curves.

Activation comparison : ReLU vs Tanh
  → ReLU: max(0,x).  Sparse, fast, solves vanishing gradient.
    Dead neuron problem: if a neuron always outputs 0, it never recovers.
  → Tanh: saturates at ±1. Smooth gradient everywhere but squashes
    large activations → slower gradient flow in deep nets.

Regularisation comparison : Dropout 0.0 vs 0.3
  → 0.0: no dropout. Model memorises training data perfectly (overfitting).
    You see training loss → near 0, validation loss blows up.
  → 0.3: each forward pass randomly drops 30% of neurons. Forces the net
    to learn redundant representations → better generalisation.

Architecture comparison : MLP, CNN1D, LSTM, PatchTST
  → MLP:      ignores temporal order.  Worst on trending assets (BTC, NVDA).
  → CNN:      captures short-range patterns (momentum, reversals).
  → LSTM:     captures long-range dependencies via gated memory.
  → PatchTST: Attention over patches → global context.  Typically best.

Usage
-----
  # Full run (all 48 assets, all configs) — takes hours, run in tmux/nohup
  tmux new -s lab
  python scripts/run_dl_lab.py 2>&1 | tee data/output/lab_run.log

  # Subset for quick sanity check
  python scripts/run_dl_lab.py --assets NVDA BTC-USD EURUSD=X

  # Skip already-completed combos (crash-safe resume)
  python scripts/run_dl_lab.py --resume

  # Just analyse existing results
  python scripts/run_dl_lab.py --analyse data/output/lab_results_TIMESTAMP.json
"""

import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# tqdm — progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
LAB_DIR   = ROOT / "data" / "lab_models"
OUT_DIR   = ROOT / "data" / "output"
LAB_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.lab.models import MLPBaseline, CNN1DModel, LSTMModel
from src.forecast.patchtst_forecast import PatchTST, build_features, make_sequences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Universe ─────────────────────────────────────────────────────────────────
UNIVERSE = [
    # Indices
    ("^GSPC",    "S&P 500",        "equity"),
    ("^IXIC",    "NASDAQ",         "equity"),
    ("^DJI",     "Dow Jones",      "equity"),
    # US Stocks
    ("AAPL",     "Apple",          "equity"),
    ("MSFT",     "Microsoft",      "equity"),
    ("NVDA",     "NVIDIA",         "equity"),
    ("TSLA",     "Tesla",          "equity"),
    ("AMZN",     "Amazon",         "equity"),
    ("META",     "Meta",           "equity"),
    ("GOOGL",    "Alphabet",       "equity"),
    ("JPM",      "JPMorgan",       "equity"),
    # Semiconductors / GPU supply chain
    ("CRWV",     "CoreWeave",      "equity"),
    ("MU",       "Micron",         "equity"),
    ("SMCI",     "Super Micro",    "equity"),
    ("ARM",      "Arm Holdings",   "equity"),
    ("TSM",      "TSMC",           "equity"),
    ("VRT",      "Vertiv",         "equity"),
    ("MRVL",     "Marvell",        "equity"),
    ("NBIS",     "Nebius",         "equity"),
    ("IREN",     "IREN",           "equity"),
    # Crypto
    ("BTC-USD",  "Bitcoin",        "crypto"),
    ("ETH-USD",  "Ethereum",       "crypto"),
    ("SOL-USD",  "Solana",         "crypto"),
    ("BNB-USD",  "BNB",            "crypto"),
    ("XRP-USD",  "XRP",            "crypto"),
    # Forex Majors
    ("EURUSD=X", "EUR/USD",        "forex"),
    ("GBPUSD=X", "GBP/USD",        "forex"),
    ("USDJPY=X", "USD/JPY",        "forex"),
    ("USDCHF=X", "USD/CHF",        "forex"),
    ("AUDUSD=X", "AUD/USD",        "forex"),
    ("NZDUSD=X", "NZD/USD",        "forex"),
    ("USDCAD=X", "USD/CAD",        "forex"),
    # Forex Cross
    ("EURGBP=X", "EUR/GBP",        "forex"),
    ("EURJPY=X", "EUR/JPY",        "forex"),
    ("GBPJPY=X", "GBP/JPY",        "forex"),
    ("AUDNZD=X", "AUD/NZD",        "forex"),
    # African/Emerging
    ("USDNGN=X", "USD/NGN",        "forex"),
    ("EURNGN=X", "EUR/NGN",        "forex"),
    ("USDZAR=X", "USD/ZAR",        "forex"),
    ("USDKES=X", "USD/KES",        "forex"),
    ("USDGHS=X", "USD/GHS",        "forex"),
    # Commodities
    ("GC=F",     "Gold",           "commodity"),
    ("SI=F",     "Silver",         "commodity"),
    ("CL=F",     "Crude Oil",      "commodity"),
    ("NG=F",     "Natural Gas",    "commodity"),
    ("HG=F",     "Copper",         "commodity"),
    ("ZW=F",     "Wheat",          "commodity"),
    ("ZC=F",     "Corn",           "commodity"),
]

# ── Hyperparameter Grid ───────────────────────────────────────────────────────
#
# Theory note: why these choices?
#
# Optimizers:
#   SGD   — stochastic gradient descent, lr=0.01, momentum=0.9 (Nesterov variant)
#           Simple, interpretable.  Requires careful LR tuning.  Can generalise
#           better than Adam if LR schedule is hand-tuned (many CV papers use SGD).
#   Adam  — adaptive lr per parameter.  Fast convergence on noisy financial data.
#           Default choice for deep learning; almost always outperforms SGD on
#           new problems.
#
# Activations:
#   ReLU  — max(0,x). Default choice since AlexNet (2012).  Sparse gradients.
#   Tanh  — maps ℝ → (-1,1).  Historically used in RNNs.  Saturates at extremes.
#
# Dropout:
#   0.0   → no regularisation → deliberately force overfitting for learning purposes
#   0.3   → moderate regularisation → observe gap reduction vs 0.0

GRID = {
    "optimizer":   ["sgd", "adam"],
    "activation":  ["relu", "tanh"],
    "dropout":     [0.0, 0.3],
}

# PatchTST has its own fixed config — it doesn't participate in the MLP/LSTM sweep
# but we train one per asset for the architecture comparison.
PATCHTST_CFG = {
    "patch_length": 16, "stride": 8, "lookback": 60,
    "d_model": 64, "n_heads": 4, "n_layers": 2,
    "dropout": 0.2, "prediction_horizon": 5,
    "batch_size": 32, "epochs": 40, "patience": 8,
    "lr": 1e-4, "weight_decay": 1e-4, "huber_delta": 0.001,
    "use_revin": True,
}

LAB_EPOCHS   = 40     # keep fast enough to compare; increase for final results
LAB_PATIENCE = 8
LAB_BATCH    = 32
LOOKBACK     = 60
HORIZON      = 5
N_FEATURES   = 10     # must match build_features() in patchtst_forecast.py


# ── Helper: optimizer factory ────────────────────────────────────────────────
def make_optimizer(name: str, params, lr: float = 1e-3):
    """
    Theory:
      SGD:  θ_{t+1} = θ_t - lr * ∇L(θ_t)
            With momentum: v_{t+1} = γv_t + ∇L; θ = θ - lr*v
      Adam: m_t = β1*m_{t-1} + (1-β1)*∇L      ← 1st moment (mean)
            v_t = β2*v_{t-1} + (1-β2)*∇L²     ← 2nd moment (variance)
            θ -= lr * m̂_t / (√v̂_t + ε)       ← bias-corrected update
    """
    if name == "sgd":
        return torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def make_activation(name: str):
    """Return activation class (not instance) for passing to model constructors."""
    return {"relu": nn.ReLU, "tanh": nn.Tanh}[name]


# ── Data preparation ─────────────────────────────────────────────────────────
def prepare_data(df: "pd.DataFrame"):
    """
    Build scaled (X_train, y_train, X_val, y_val, X_test, y_test) tensors
    from a raw OHLCV dataframe using the same build_features() as PatchTST.

    Split: 70% train | 15% val | 15% test — all chronological (no leakage).
    Scaler fitted on train only to prevent data leakage from val/test into
    training statistics.
    """
    features = build_features(df, regime_label=0)
    prices   = df["Close"].reindex(features.index)
    X, y     = make_sequences(features, prices, LOOKBACK, HORIZON)

    if len(X) < 100:
        return None

    n       = len(X)
    n_tr    = int(n * 0.70)
    n_va    = int(n * 0.85)

    X_tr, y_tr = X[:n_tr],       y[:n_tr]
    X_va, y_va = X[n_tr:n_va],   y[n_tr:n_va]
    X_te, y_te = X[n_va:],       y[n_va:]

    # StandardScaler: z = (x - μ) / σ  — fit on train only
    sc = StandardScaler()
    shape_tr = X_tr.shape
    X_tr_flat = X_tr.reshape(-1, N_FEATURES)
    sc.fit(X_tr_flat)

    def scale(arr):
        s = arr.shape
        return sc.transform(arr.reshape(-1, N_FEATURES)).reshape(s).astype(np.float32)

    X_tr = scale(X_tr); X_va = scale(X_va); X_te = scale(X_te)

    def to_tensor(x, y_arr):
        return (torch.tensor(x, device=DEVICE),
                torch.tensor(y_arr.astype(np.float32), device=DEVICE))

    return {
        "train": to_tensor(X_tr, y_tr),
        "val":   to_tensor(X_va, y_va),
        "test":  to_tensor(X_te, y_te),
        "n_tr":  len(X_tr),
        "n_va":  len(X_va),
        "n_te":  len(X_te),
        "scaler": sc,
    }


def make_loaders(data: dict, batch: int = LAB_BATCH):
    def loader(split, shuffle):
        X, y = data[split]
        return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=shuffle)
    return loader("train", True), loader("val", False), loader("test", False)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one(model, dl_tr, dl_va, optimizer, epochs=LAB_EPOCHS, patience=LAB_PATIENCE):
    """
    Standard training loop with early stopping.

    Returns
    -------
    history : {"train": [...], "val": [...]}  — MSE per epoch
    best_val_mse : float
    epochs_run : int
    """
    loss_fn  = nn.MSELoss()
    # LR scheduler: reduce on plateau if val loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=False
    )

    best_val  = float("inf")
    best_state = None
    no_improve = 0
    history    = {"train": [], "val": []}

    for epoch in range(epochs):
        # ── train ────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(dl_tr)

        # ── validate ─────────────────────────────────────────────────────────
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                va_loss += loss_fn(model(xb), yb).item()
        va_loss /= len(dl_va)

        history["train"].append(round(tr_loss, 8))
        history["val"].append(round(va_loss, 8))
        scheduler.step(va_loss)

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return history, best_val, len(history["train"])


def eval_test_mse(model, dl_te):
    loss_fn = nn.MSELoss()
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in dl_te:
            total += loss_fn(model(xb), yb).item()
    return total / len(dl_te)


# ── PatchTST training (reuses existing forecaster logic) ─────────────────────
def train_patchtst(data: dict, sym_key: str, epochs: int = LAB_EPOCHS):
    """Train a PatchTST model using data already pre-scaled.  Returns result dict."""
    cfg = {**PATCHTST_CFG, "n_features": N_FEATURES, "epochs": epochs}
    model = PatchTST(cfg).to(DEVICE)

    dl_tr, dl_va, dl_te = make_loaders(data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                   weight_decay=cfg["weight_decay"])
    loss_fn   = nn.HuberLoss(delta=cfg["huber_delta"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"])

    best_val  = float("inf")
    best_state = None
    no_improve = 0
    history    = {"train": [], "val": []}

    for epoch in range(cfg["epochs"]):
        model.train()
        tr = 0.0
        for xb, yb in dl_tr:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr += loss.item()
        tr /= len(dl_tr)
        scheduler.step()

        model.eval()
        va = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                va += loss_fn(model(xb), yb).item()
        va /= len(dl_va)

        history["train"].append(round(float(tr), 8))
        history["val"].append(round(float(va), 8))

        if va < best_val:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    test_mse = eval_test_mse(model, dl_te)

    # Save checkpoint
    ckpt_path = LAB_DIR / f"patchtst_{sym_key}.pt"
    torch.save({"model_state": model.state_dict(), "cfg": cfg}, ckpt_path)

    return {
        "algo":           "PatchTST",
        "optimizer":      "AdamW",
        "activation":     "GELU",
        "dropout":        cfg["dropout"],
        "val_mse":        round(best_val, 8),
        "test_mse":       round(test_mse, 8),
        "epochs_run":     len(history["train"]),
        "loss_history":   history,
        "checkpoint":     str(ckpt_path),
    }


# ── Build and train a lab model for one config ────────────────────────────────
def build_model(algo: str, activation_cls, dropout: float) -> nn.Module:
    """Instantiate model for given algorithm name."""
    if algo == "MLP":
        return MLPBaseline(
            lookback=LOOKBACK, n_features=N_FEATURES,
            hidden_sizes=[256, 128, 64],
            activation=activation_cls,
            dropout=dropout,
        ).to(DEVICE)
    elif algo == "CNN":
        return CNN1DModel(
            n_features=N_FEATURES,
            channels=[32, 64, 128],
            kernel_size=5,
            dropout=dropout,
            activation=activation_cls,
        ).to(DEVICE)
    elif algo == "LSTM":
        return LSTMModel(
            n_features=N_FEATURES,
            hidden_size=128,
            num_layers=2,
            dropout=dropout,
            bidirectional=False,
            activation=activation_cls,
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown algo: {algo}")


def run_config(algo, opt_name, act_name, dropout, data, sym_key, epochs=LAB_EPOCHS):
    """Train one (algo, optimizer, activation, dropout) config. Returns result dict."""
    act_cls = make_activation(act_name)
    model   = build_model(algo, act_cls, dropout)
    optimizer = make_optimizer(opt_name, model.parameters())
    dl_tr, dl_va, dl_te = make_loaders(data)

    t0 = time.time()
    history, best_val, n_epochs = train_one(model, dl_tr, dl_va, optimizer, epochs=epochs)
    test_mse = eval_test_mse(model, dl_te)
    elapsed  = round(time.time() - t0, 1)

    # Save checkpoint
    ckpt_name = f"{algo.lower()}_{sym_key}_{opt_name}_{act_name}_do{int(dropout*10)}.pt"
    ckpt_path = LAB_DIR / ckpt_name
    torch.save({"model_state": model.state_dict(),
                "algo": algo, "opt": opt_name,
                "act": act_name, "dropout": dropout}, ckpt_path)

    return {
        "algo":         algo,
        "optimizer":    opt_name,
        "activation":   act_name,
        "dropout":      dropout,
        "val_mse":      round(best_val, 8),
        "test_mse":     round(test_mse, 8),
        "epochs_run":   n_epochs,
        "train_secs":   elapsed,
        "loss_history": history,
        "checkpoint":   str(ckpt_path),
        "overfit_ratio": round(
            # ratio of final val/train loss — >2.0 indicates overfitting
            (history["val"][-1] / (history["train"][-1] + 1e-12)), 3
        ),
    }


# ── Incremental JSON writer ───────────────────────────────────────────────────
class IncrementalWriter:
    """Appends results to a JSON file one record at a time (crash-safe)."""

    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            path.write_text("[]")

    def append(self, record: dict):
        try:
            existing = json.loads(self.path.read_text())
        except Exception:
            existing = []
        existing.append(record)
        self.path.write_text(json.dumps(existing, indent=2))

    def load(self):
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return []


def make_done_set(results: list) -> set:
    """Build a set of (sym, algo, optimizer, activation, dropout) already done."""
    done = set()
    for r in results:
        done.add((
            r["symbol"],
            r["algo"],
            r.get("optimizer", ""),
            r.get("activation", ""),
            str(r.get("dropout", "")),
        ))
    return done


# ── Analysis ──────────────────────────────────────────────────────────────────
def analyse(results_path: str):
    """
    Load results JSON and print multi-dimensional analysis.
    Answers the key learning questions:
      1. Which optimizer converges faster? (SGD vs Adam mean val_mse)
      2. Which activation is better? (ReLU vs Tanh)
      3. Does dropout help? (overfit_ratio with/without)
      4. Which architecture wins? (MLP vs CNN vs LSTM vs PatchTST)
      5. Best config per asset and per asset class.
    """
    path = Path(results_path)
    if not path.exists():
        print(f"File not found: {results_path}")
        return

    results = json.loads(path.read_text())
    if not results:
        print("No results to analyse yet.")
        return

    print(f"\n{'='*70}")
    print(f"  DL LAB ANALYSIS  —  {len(results)} experiment records")
    print(f"{'='*70}\n")

    # Group by dimension
    from collections import defaultdict

    def bucket(key_fn):
        d = defaultdict(list)
        for r in results:
            k = key_fn(r)
            if k is not None:
                d[k].append(r)
        return d

    def mean_mse(records):
        vals = [r["test_mse"] for r in records if r.get("test_mse") is not None]
        return round(np.mean(vals), 8) if vals else None

    def mean_overfit(records):
        vals = [r.get("overfit_ratio", 1.0) for r in records]
        return round(np.mean(vals), 3) if vals else None

    # ── 1. Optimizer ──────────────────────────────────────────────────────────
    print("── 1. Optimizer comparison (mean test MSE across all assets) ──")
    by_opt = bucket(lambda r: r.get("optimizer"))
    for opt, recs in sorted(by_opt.items()):
        print(f"   {opt:<8}  test_mse={mean_mse(recs):<12}  n={len(recs)}")

    # ── 2. Activation ─────────────────────────────────────────────────────────
    print("\n── 2. Activation comparison ──")
    by_act = bucket(lambda r: r.get("activation"))
    for act, recs in sorted(by_act.items()):
        print(f"   {act:<8}  test_mse={mean_mse(recs):<12}  n={len(recs)}")

    # ── 3. Regularisation ─────────────────────────────────────────────────────
    print("\n── 3. Dropout (regularisation) effect ──")
    by_do = bucket(lambda r: str(r.get("dropout", "")))
    for do, recs in sorted(by_do.items()):
        print(f"   dropout={do:<5}  test_mse={mean_mse(recs):<12}  "
              f"overfit_ratio={mean_overfit(recs)}")
    print("  (overfit_ratio = final_val_loss / final_train_loss; "
          ">2.0 = strong overfitting)")

    # ── 4. Architecture ───────────────────────────────────────────────────────
    print("\n── 4. Architecture comparison (mean test MSE) ──")
    by_algo = bucket(lambda r: r.get("algo"))
    for algo, recs in sorted(by_algo.items(), key=lambda x: mean_mse(x[1]) or 999):
        print(f"   {algo:<12}  test_mse={mean_mse(recs):<12}  n={len(recs)}")

    # ── 5. Best config per asset ──────────────────────────────────────────────
    print("\n── 5. Best config per asset (lowest test MSE) ──")
    by_sym = bucket(lambda r: r.get("symbol"))
    for sym in sorted(by_sym.keys()):
        recs    = by_sym[sym]
        best    = min(recs, key=lambda r: r.get("test_mse") or 999)
        worst   = max(recs, key=lambda r: r.get("test_mse") or 0)
        print(f"   {sym:<12}  best={best['algo']:<12}({best.get('optimizer','')}"
              f"/{best.get('activation','')}/do{best.get('dropout','')})"
              f"  mse={best['test_mse']:.2e}"
              f"  worst={worst['algo']} mse={worst['test_mse']:.2e}")

    # ── 6. Best config per asset class ────────────────────────────────────────
    print("\n── 6. Best architecture per asset class ──")
    by_class = bucket(lambda r: r.get("asset_class"))
    for cls, recs in sorted(by_class.items()):
        by_algo2 = bucket(lambda r: r.get("algo"))
        for algo, sub in by_algo2.items():
            _ = sub  # populated from recs
        # recompute per class
        cls_by_algo = defaultdict(list)
        for r in recs:
            cls_by_algo[r.get("algo")].append(r)
        ranked = sorted(cls_by_algo.items(),
                        key=lambda x: mean_mse(x[1]) or 999)
        print(f"\n   {cls.upper()}:")
        for algo, sub in ranked:
            print(f"     {algo:<12}  mean_test_mse={mean_mse(sub):.4e}")

    # ── 7. Convergence speed ─────────────────────────────────────────────────
    print("\n── 7. Convergence speed (mean epochs to early stop) ──")
    by_algo3 = bucket(lambda r: r.get("algo"))
    for algo, recs in sorted(by_algo3.items()):
        ep = [r.get("epochs_run", 0) for r in recs]
        print(f"   {algo:<12}  mean_epochs={round(np.mean(ep), 1)}")

    print(f"\n{'='*70}")
    print("Tip: load the JSON into a notebook and call:")
    print("  import json, pandas as pd")
    print("  df = pd.DataFrame(json.load(open('lab_results_....json')))")
    print("  df.groupby(['algo','optimizer'])['test_mse'].mean().sort_values()")
    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DL Laboratory — train and compare")
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Subset of symbols, e.g. --assets NVDA BTC-USD")
    parser.add_argument("--resume", action="store_true",
                        help="Skip combos already present in the latest results file")
    parser.add_argument("--analyse", default=None, metavar="JSON_PATH",
                        help="Analyse an existing results JSON and exit")
    parser.add_argument("--epochs", type=int, default=LAB_EPOCHS,
                        help=f"Training epochs per config (default {LAB_EPOCHS})")
    args = parser.parse_args()

    if args.analyse:
        analyse(args.analyse)
        return

    run_epochs = args.epochs

    # Output file
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path    = OUT_DIR / f"lab_results_{ts}.json"
    writer      = IncrementalWriter(out_path)
    done_set    = set()

    if args.resume:
        # Find latest results file to resume from
        existing = sorted(OUT_DIR.glob("lab_results_*.json"), reverse=True)
        if existing:
            latest_data = json.loads(existing[0].read_text())
            done_set    = make_done_set(latest_data)
            print(f"  Resuming from {existing[0].name} "
                  f"({len(done_set)} configs already done)")
            # Write existing records to new file
            for rec in latest_data:
                writer.append(rec)

    universe = [(s, n, c) for s, n, c in UNIVERSE
                if args.assets is None or s in args.assets]

    # Grid configs for sweepable architectures (MLP, CNN, LSTM)
    SWEEP_ALGOS = ["MLP", "CNN", "LSTM"]
    grid_keys   = list(GRID.keys())
    grid_vals   = list(GRID.values())
    configs     = list(product(*grid_vals))   # Cartesian product

    total_runs  = len(universe) * (len(SWEEP_ALGOS) * len(configs) + 1)
    print(f"\n{'='*70}")
    print(f"  DEEP LEARNING LABORATORY")
    print(f"  Device : {DEVICE}")
    print(f"  Assets : {len(universe)}")
    print(f"  Configs: {len(SWEEP_ALGOS)} algos × {len(configs)} grid = "
          f"{len(SWEEP_ALGOS)*len(configs)} + 1 PatchTST = "
          f"{len(SWEEP_ALGOS)*len(configs)+1} per asset")
    print(f"  Total  : {total_runs} runs")
    print(f"  Output : {out_path}")
    print(f"{'='*70}\n")

    completed = 0
    failed    = []

    asset_bar = tqdm(universe, desc="Assets", position=0)
    for sym, name, asset_class in asset_bar:
        asset_bar.set_description(f"{sym}")
        sym_key = sym.replace("^","").replace("-","_").replace("=","")

        # ── Fetch data ────────────────────────────────────────────────────────
        try:
            df = yf.download(sym, start="2004-01-01",
                             end=datetime.now().strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
            if hasattr(df.columns, "droplevel"):
                try:
                    df.columns = df.columns.droplevel(1)
                except Exception:
                    pass
            df = df.dropna()
            if len(df) < 200:
                print(f"\n  ⚠️  {sym}: only {len(df)} rows — skipping")
                failed.append(sym)
                continue
        except Exception as e:
            print(f"\n  ❌ {sym}: data fetch failed — {e}")
            failed.append(sym)
            continue

        # ── Prepare sequences ─────────────────────────────────────────────────
        try:
            data = prepare_data(df)
            if data is None:
                print(f"\n  ⚠️  {sym}: too few sequences — skipping")
                failed.append(sym)
                continue
        except Exception as e:
            print(f"\n  ❌ {sym}: feature build failed — {e}")
            failed.append(sym)
            continue

        # ── PatchTST (one per asset, fixed config) ────────────────────────────
        ptst_key = (sym, "PatchTST", "AdamW", "GELU", "0.2")
        if ptst_key not in done_set:
            try:
                ptst_result = train_patchtst(data, sym_key, epochs=run_epochs)
                record = {"symbol": sym, "name": name, "asset_class": asset_class,
                          **ptst_result}
                writer.append(record)
                done_set.add(ptst_key)
                completed += 1
            except Exception as e:
                print(f"\n  ❌ {sym} PatchTST: {e}")

        # ── Sweep: MLP, CNN, LSTM × {optimizer, activation, dropout} ─────────
        cfg_bar = tqdm(configs, desc=f"  {sym} grid", position=1, leave=False)
        for cfg_vals in cfg_bar:
            cfg_dict = dict(zip(grid_keys, cfg_vals))
            opt, act, do = cfg_dict["optimizer"], cfg_dict["activation"], cfg_dict["dropout"]

            for algo in SWEEP_ALGOS:
                key = (sym, algo, opt, act, str(do))
                if key in done_set:
                    continue

                cfg_bar.set_postfix(algo=algo, opt=opt, act=act, drop=do)
                try:
                    result = run_config(algo, opt, act, do, data, sym_key, epochs=run_epochs)
                    record = {"symbol": sym, "name": name, "asset_class": asset_class,
                              **result}
                    writer.append(record)
                    done_set.add(key)
                    completed += 1
                except Exception as e:
                    print(f"\n  ❌ {sym} {algo}/{opt}/{act}/do={do}: {e}")
                    failed.append(f"{sym}_{algo}_{opt}_{act}")

    print(f"\n{'='*70}")
    print(f"  LAB COMPLETE")
    print(f"  Finished  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Completed : {completed} runs")
    print(f"  Failed    : {len(failed)}")
    print(f"  Results   → {out_path}")
    print(f"{'='*70}")
    print(f"\nTo analyse:\n"
          f"  python scripts/run_dl_lab.py --analyse {out_path}\n"
          f"\nTo plot learning curves in Python:\n"
          f"  import json, matplotlib.pyplot as plt\n"
          f"  data = json.load(open('{out_path}'))\n"
          f"  # filter to one asset+config, then:\n"
          f"  plt.plot(r['loss_history']['train'], label='train')\n"
          f"  plt.plot(r['loss_history']['val'],   label='val')\n"
          f"  plt.legend(); plt.title(r['symbol'] + ' — ' + r['algo'])\n"
          f"  plt.show()")


if __name__ == "__main__":
    main()
