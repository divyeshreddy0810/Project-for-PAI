#!/usr/bin/env python3
"""
Regime Classification Evaluator
================================
Compares 4 regime detectors using proper ML classification metrics.

What "regime" means here:
  Bull     = next 20 trading days return > +2%
  Bear     = next 20 trading days return < -2%
  Sideways = between -2% and +2%

Why this matters:
  If your regime detector is wrong, you're applying the wrong risk
  profile to the wrong market state — the equivalent of wearing a
  raincoat on a sunny day or shorts in a blizzard.

Models compared:
  1. HMM (your existing model) — probabilistic sequence model
  2. SMA Crossover — rule-based baseline (price > 50-SMA = bull)
  3. Logistic Regression — linear classifier on technical features
  4. Neural Net — 2-layer MLP on same features

Metrics used (and WHY each matters):
  Accuracy   — overall % correct (misleading if classes imbalanced)
  Precision  — of all "bull" calls, how many were actually bull?
                (measures false alarm rate)
  Recall     — of all actual bull periods, how many did we catch?
                (measures miss rate)
  F1 score   — harmonic mean of precision and recall
                (better than accuracy for imbalanced classes)
  Confusion  — full breakdown: what did we call, what was true?

The confusion matrix is the most informative for trading:
  A missed BEAR (predicted HOLD, was BEAR) costs you real money.
  A false BULL (predicted BULL, was BEAR) is even worse.

Usage:
    python scripts/evaluate_regime.py --sym NVDA
    python scripts/evaluate_regime.py --sym BTC-USD
    python scripts/evaluate_regime.py --sym EURUSD=X --horizon 10
"""

import os, sys, warnings, argparse
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Technical features for regime classification.
    Same features used by all non-HMM classifiers for fair comparison.
    """
    close  = df["Close"]
    ret    = close.pct_change().fillna(0)
    ret5   = close.pct_change(5).fillna(0)
    ret20  = close.pct_change(20).fillna(0)

    # RSI — momentum oscillator (0-100)
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rsi    = (100 - (100 / (1 + gain / (loss + 1e-9)))).fillna(50)

    # MACD — trend direction and momentum
    ema12  = close.ewm(span=12).mean()
    ema26  = close.ewm(span=26).mean()
    macd   = (ema12 - ema26) / (close + 1e-9)

    # Rolling volatility — risk level
    vol10  = ret.rolling(10).std().fillna(0.01)
    vol20  = ret.rolling(20).std().fillna(0.01)

    # Distance from 50/200-day SMA — trend position
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    d50    = ((close - sma50) / (sma50 + 1e-9)).fillna(0)
    d200   = ((close - sma200) / (sma200 + 1e-9)).fillna(0)

    # Bollinger Band position — stretched/compressed
    roll_m = close.rolling(20).mean()
    roll_s = close.rolling(20).std().replace(0, np.nan)
    bb_pos = ((close - roll_m) / (2 * roll_s)).clip(-2, 2).fillna(0)

    feats = np.column_stack([
        ret.values, ret5.values, ret20.values,
        rsi.values / 100.0,   # normalise to [0,1]
        macd.values,
        vol10.values, vol20.values,
        d50.values, d200.values,
        bb_pos.values,
    ])
    return feats


def build_labels(df: pd.DataFrame, horizon: int = 20,
                 bull_thresh: float = 0.02,
                 bear_thresh: float = -0.02) -> np.ndarray:
    """
    Ground-truth regime labels from forward returns.

    Args:
        horizon    : Days ahead to compute the forward return (default 20)
        bull_thresh: Forward return above this → BULL  (default +2%)
        bear_thresh: Forward return below this → BEAR  (default -2%)

    Labels: 1=bull, 0=sideways, -1=bear

    Why forward returns, not HMM states?
    Because we want to evaluate how well each model predicts what
    ACTUALLY HAPPENED, not whether it agrees with another model.
    This is supervised evaluation against reality.
    """
    close   = df["Close"].values
    fwd_ret = (pd.Series(close).shift(-horizon) / pd.Series(close) - 1).values
    labels  = np.where(fwd_ret > bull_thresh,  1,
              np.where(fwd_ret < bear_thresh, -1, 0))
    return labels


# ══════════════════════════════════════════════════════════════════
# CLASSIFIERS
# ══════════════════════════════════════════════════════════════════

class RegimeNet(nn.Module):
    """
    Simple 2-layer neural network for 3-class regime classification.
    Input: technical features
    Output: softmax over [bear, sideways, bull] → 3 classes

    Architecture decisions:
      - 2 layers: enough for tabular data (deeper = usually worse for small datasets)
      - BatchNorm: stabilises training on different-scale features
      - Dropout 0.3: prevents overfitting on small regime datasets
      - CrossEntropyLoss: standard for multi-class classification
        (internally applies softmax, so don't apply it twice)
    """
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3),    # 3 output classes
        )

    def forward(self, x): return self.net(x)


def train_regime_net(X_train: np.ndarray, y_train: np.ndarray,
                     epochs: int = 100) -> RegimeNet:
    """
    Train the neural net regime classifier.
    Uses CrossEntropyLoss — standard for multi-class classification.
    """
    # Remap labels -1,0,1 → 0,1,2 (PyTorch needs 0-indexed classes)
    y_mapped = y_train + 1   # -1→0, 0→1, 1→2

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_mapped, dtype=torch.long).to(DEVICE)

    model   = RegimeNet(X_train.shape[1]).to(DEVICE)
    # CrossEntropyLoss = log_softmax + NLLLoss
    # It penalises confident wrong predictions more than uncertain wrong ones
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)

    model.train()
    for ep in range(epochs):
        logits = model(X_t)
        loss   = criterion(logits, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def predict_regime_net(model: RegimeNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t    = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        logits = model(X_t)
        preds  = logits.argmax(-1).cpu().numpy()
    return preds - 1   # remap 0,1,2 → -1,0,1


def predict_sma_crossover(df: pd.DataFrame, n: int = 50) -> np.ndarray:
    """
    Simplest possible regime rule: price above N-day SMA = bull.
    This is the baseline every fancier model must beat.
    If your HMM can't beat a 50-day moving average, something is wrong.
    """
    close = df["Close"]
    sma   = close.rolling(n).mean()
    preds = np.where(close > sma, 1, np.where(close < sma * 0.98, -1, 0))
    return preds


def predict_hmm(df: pd.DataFrame, asset_class: str = "equity") -> np.ndarray:
    """Run the existing HMM on the dataframe, return per-row regime labels."""
    try:
        import joblib
        hmm_path = f"data/models/hmm_{asset_class}.pkl"
        if os.path.exists(hmm_path):
            hmm = joblib.load(hmm_path)
        else:
            from src.regime.pretrained_hmm import get_pretrained_hmm
            hmm = get_pretrained_hmm()

        # HMM expects a df with Close; predict rolling windows
        label_map = {"bull": 1, "sideways": 0, "bear": -1}
        preds     = []
        WINDOW    = 60
        for i in range(len(df)):
            start = max(0, i - WINDOW)
            chunk = df.iloc[start:i+1]
            try:
                label = hmm.predict(chunk)
                preds.append(label_map.get(label, 0))
            except Exception:
                preds.append(0)
        return np.array(preds)
    except Exception as e:
        print(f"  ⚠️  HMM predict failed: {e} — returning all sideways")
        return np.zeros(len(df), dtype=int)


# ══════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════

def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray,
                        name: str) -> dict:
    """
    Full classification evaluation with all metrics.
    Returns dict of results AND prints formatted report.
    """
    label_names = ["Bear (-1)", "Sideways (0)", "Bull (+1)"]
    labels_int  = [-1, 0, 1]

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, labels=labels_int,
                    average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, labels=labels_int,
                           average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, labels=labels_int,
                        average="macro", zero_division=0)

    # Per-class F1
    f1_per = f1_score(y_true, y_pred, labels=labels_int,
                      average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels_int)

    print(f"\n  ── {name} ──────────────────────────────────────────")
    print(f"     Accuracy : {acc:.3f}  (warning: can be misleading if imbalanced)")
    print(f"     Macro F1 : {f1:.3f}  (primary metric — balanced across classes)")
    print(f"     Precision: {prec:.3f}  (of our signals, how many were right?)")
    print(f"     Recall   : {rec:.3f}  (of actual events, how many did we catch?)")
    print(f"     F1 Bear  : {f1_per[0]:.3f}  ← most important for risk management")
    print(f"     F1 Side  : {f1_per[1]:.3f}")
    print(f"     F1 Bull  : {f1_per[2]:.3f}")

    # Confusion matrix
    print(f"\n     Confusion matrix (rows=actual, cols=predicted):")
    print(f"                 Pred:Bear  Pred:Side  Pred:Bull")
    for i, label in enumerate(["Act:Bear ", "Act:Side ", "Act:Bull "]):
        row = cm[i]
        # Highlight the diagonal (correct predictions)
        row_str = "  ".join(
            f"[{v:6d}]" if j == i else f" {v:6d} "
            for j, v in enumerate(row)
        )
        print(f"     {label}   {row_str}")

    return {
        "model":      name,
        "accuracy":   round(acc,  4),
        "macro_f1":   round(f1,   4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1_bear":    round(f1_per[0], 4),
        "f1_side":    round(f1_per[1], 4),
        "f1_bull":    round(f1_per[2], 4),
    }


# ══════════════════════════════════════════════════════════════════
# BINARY EVALUATION (bull vs not-bull)
# ══════════════════════════════════════════════════════════════════

def binary_evaluation(y_true: np.ndarray, y_pred: np.ndarray, name: str):
    """
    Simplify to binary: bull (1) vs not-bull (0).
    This is the most actionable question: "should I be long?"

    Precision = if I go long, how often do I make money?
    Recall    = of all profitable long opportunities, how many did I catch?
    """
    y_true_bin = (y_true == 1).astype(int)
    y_pred_bin = (y_pred == 1).astype(int)

    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    print(f"  {name:<35} Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
    return {"model": name, "binary_prec": prec, "binary_rec": rec, "binary_f1": f1}


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym",         default="NVDA",
                        help="Asset symbol")
    parser.add_argument("--asset_class", default="equity",
                        help="equity / crypto / forex / commodity")
    parser.add_argument("--horizon",     type=int, default=20,
                        help="Forward return horizon in days (default 20)")
    parser.add_argument("--bull_thresh", type=float, default=0.02,
                        help="Bull threshold, e.g. 0.02 = +2%%")
    parser.add_argument("--bear_thresh", type=float, default=-0.02,
                        help="Bear threshold, e.g. -0.02 = -2%%")
    args = parser.parse_args()

    sym = args.sym
    print("=" * 65)
    print(f"  REGIME CLASSIFICATION EVALUATION — {sym}")
    print(f"  Horizon: {args.horizon}d  |  "
          f"Bull: >{args.bull_thresh*100:.0f}%  |  "
          f"Bear: <{args.bear_thresh*100:.0f}%")
    print("=" * 65)

    # ── Fetch data ─────────────────────────────────────────────
    print(f"\n  Fetching {sym}...")
    df = yf.download(sym, start="2010-01-01",
                     end=datetime.now().strftime("%Y-%m-%d"),
                     progress=False)
    if hasattr(df.columns, "droplevel"):
        df.columns = df.columns.droplevel(1)
    df = df.dropna().reset_index(drop=True)
    print(f"  {len(df)} rows")

    # ── Build features and labels ──────────────────────────────
    feats  = build_features(df)
    labels = build_labels(df, horizon=args.horizon,
                          bull_thresh=args.bull_thresh,
                          bear_thresh=args.bear_thresh)

    # Only keep rows where labels are valid (last `horizon` rows have no label)
    valid  = ~np.isnan(labels.astype(float))
    valid[:200] = False   # need enough history for all features
    feats  = feats[valid]
    labels = labels[valid]
    df_v   = df[valid].reset_index(drop=True)

    # Train/test split
    n_train  = int(len(feats) * 0.70)
    X_train, X_test   = feats[:n_train],  feats[n_train:]
    y_train, y_test   = labels[:n_train], labels[n_train:]
    df_test            = df_v.iloc[n_train:].reset_index(drop=True)

    # Label distribution
    bull_pct = (y_test == 1).mean()
    bear_pct = (y_test == -1).mean()
    side_pct = (y_test == 0).mean()
    print(f"\n  Test set class distribution:")
    print(f"    Bull: {bull_pct:.1%}  Sideways: {side_pct:.1%}  Bear: {bear_pct:.1%}")
    print(f"  (if one class dominates, accuracy is misleading — use F1)")

    # Standardise features (important for LR and NN, not for HMM/SMA)
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Run all classifiers ────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FULL 3-CLASS EVALUATION (bull / sideways / bear)")
    print(f"{'='*65}")

    all_results = []

    # 1. SMA Crossover baseline
    sma_preds_full = predict_sma_crossover(df_v)
    sma_preds      = sma_preds_full[n_train:]
    all_results.append(evaluate_classifier(y_test, sma_preds,
                                           "SMA-50 Crossover (baseline)"))

    # 2. HMM
    print(f"\n  Running HMM on test set ({len(df_test)} rows)...", end=" ", flush=True)
    hmm_preds_full = predict_hmm(df_v, args.asset_class)
    hmm_preds      = hmm_preds_full[n_train:]
    print("done")
    all_results.append(evaluate_classifier(y_test, hmm_preds,
                                           "HMM (existing model)"))

    # 3. Logistic Regression
    # A linear classifier — finds a hyperplane in feature space.
    # Good baseline for neural net: if NN can't beat LR on same features,
    # the data isn't complex enough to justify the extra architecture.
    lr_model = LogisticRegression(max_iter=1000, C=1.0,
                                   class_weight="balanced",
                                   random_state=42)
    lr_model.fit(X_train_s, y_train)
    lr_preds = lr_model.predict(X_test_s)
    all_results.append(evaluate_classifier(y_test, lr_preds,
                                           "Logistic Regression"))

    # 4. Neural Network
    print(f"\n  Training regime neural net ({100} epochs)...", end=" ", flush=True)
    nn_model = train_regime_net(X_train_s, y_train, epochs=100)
    nn_preds = predict_regime_net(nn_model, X_test_s)
    print("done")
    all_results.append(evaluate_classifier(y_test, nn_preds,
                                           "Neural Net (2-layer MLP)"))

    # ── Binary evaluation ──────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  BINARY EVALUATION: Bull vs Not-Bull")
    print(f"  (most actionable: 'should I be long?')")
    print(f"{'='*65}")
    print(f"\n  {'Model':<35} {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*35} {'-'*6}  {'-'*6}  {'-'*6}")

    binary_results = []
    for preds, name in [
        (sma_preds, "SMA-50 Crossover"),
        (hmm_preds, "HMM"),
        (lr_preds,  "Logistic Regression"),
        (nn_preds,  "Neural Net"),
    ]:
        binary_results.append(binary_evaluation(y_test, preds, name))

    # ── Summary table ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY — ranked by Macro F1 (3-class)")
    print(f"{'='*65}")
    all_results.sort(key=lambda x: x["macro_f1"], reverse=True)
    print(f"\n  {'Model':<35} {'Acc':>6} {'F1':>6} {'F1-Bear':>8} {'F1-Bull':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for r in all_results:
        print(f"  {r['model']:<35} {r['accuracy']:>6.3f} {r['macro_f1']:>6.3f} "
              f"{r['f1_bear']:>8.3f} {r['f1_bull']:>8.3f}")

    # ── Learning notes ─────────────────────────────────────────
    best_model   = all_results[0]["model"]
    best_f1      = all_results[0]["macro_f1"]
    hmm_f1       = next(r["macro_f1"] for r in all_results
                        if "HMM" in r["model"])
    baseline_f1  = next(r["macro_f1"] for r in all_results
                        if "SMA" in r["model"])

    print(f"\n  Key observations:")
    print(f"    Best model: {best_model}  (F1={best_f1:.3f})")
    if hmm_f1 > baseline_f1:
        print(f"    HMM beats SMA baseline by {hmm_f1 - baseline_f1:.3f} F1 ✅")
    else:
        print(f"    HMM does NOT beat SMA baseline "
              f"({hmm_f1:.3f} vs {baseline_f1:.3f}) ⚠️")
        print(f"    → The HMM may need retraining on this asset class")
    print(f"\n  Remember: F1 >> Accuracy when classes are imbalanced.")
    print(f"  A model that always predicts SIDEWAYS gets ~{side_pct:.0%} accuracy")
    print(f"  but 0.0 F1 on bull/bear — useless for trading.\n")


if __name__ == "__main__":
    main()
