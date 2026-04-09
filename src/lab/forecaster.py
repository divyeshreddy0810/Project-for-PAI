"""
LabForecaster — drop-in replacement for PatchTSTForecaster
===========================================================
Wraps MLPBaseline, CNN1DModel, or LSTMModel with the exact same interface
as PatchTSTForecaster so the production system (daily_advisor_v2.py,
train_all_models.py) can use any winning architecture without code changes.

Interface contract (matches PatchTSTForecaster):
  forecaster.fit_from_df(df, regime_label, verbose)
  forecaster.predict_return(df, regime_label)  → float
  forecaster.predict_with_meta(df, current_price, regime_label)  → dict
  forecaster.save(path)
  forecaster.load(path)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.forecast.patchtst_forecast import build_features, make_sequences
from src.lab.models import MLPBaseline, CNN1DModel, LSTMModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default training config — same as lab winners
LAB_FORECASTER_CFG = {
    "lookback":           60,
    "prediction_horizon":  5,
    "n_features":         10,
    "batch_size":         32,
    "epochs":             60,
    "patience":           10,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "dropout":           0.3,
    "activation":       "relu",
    "optimizer":        "adam",
}

_ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _make_model(algo: str, cfg: dict) -> nn.Module:
    act = _ACTIVATION_MAP.get(cfg.get("activation", "relu"), nn.ReLU)
    do  = cfg.get("dropout", 0.3)
    nf  = cfg["n_features"]
    lb  = cfg["lookback"]

    if algo == "LSTM":
        return LSTMModel(n_features=nf, hidden_size=128, num_layers=2,
                         dropout=do, activation=act)
    elif algo == "CNN":
        return CNN1DModel(n_features=nf, channels=[32, 64, 128],
                          kernel_size=5, dropout=do, activation=act)
    elif algo == "MLP":
        return MLPBaseline(lookback=lb, n_features=nf,
                           hidden_sizes=[256, 128, 64],
                           activation=act, dropout=do)
    else:
        raise ValueError(f"Unknown algo for LabForecaster: {algo}")


class LabForecaster:
    """
    Production-ready wrapper for MLP / CNN / LSTM lab models.
    Same interface as PatchTSTForecaster — swap in, no other changes needed.
    """

    def __init__(self, algo: str = "LSTM", cfg: dict = None):
        """
        Parameters
        ----------
        algo : "LSTM" | "CNN" | "MLP"
        cfg  : override any key in LAB_FORECASTER_CFG
        """
        self.algo    = algo
        self.cfg     = {**LAB_FORECASTER_CFG, **(cfg or {})}
        self.model   = None
        self.scaler  = StandardScaler()
        self.device  = DEVICE
        self.trained = False
        self.target_std  = 0.01
        self.target_mean = 0.0

    # ── Training ────────────────────────────────────────────────────────────

    def fit_from_df(self, df, regime_label: int = 0, verbose: bool = True):
        """
        Train on a full OHLCV dataframe.
        Identical signature to PatchTSTForecaster.fit_from_df().
        """
        if verbose:
            print(f"  🔧 [{self.algo}] Building features from {len(df)} rows...")

        features = build_features(df, regime_label)
        prices   = df["Close"].reindex(features.index)
        X, y     = make_sequences(features, prices,
                                   self.cfg["lookback"],
                                   self.cfg["prediction_horizon"])

        if len(X) < 100:
            if verbose:
                print(f"  ⚠️  Too few sequences ({len(X)}) — skipping")
            return False

        # Chronological split: 70% train | 15% val | 15% test
        n     = len(X)
        n_tr  = int(n * 0.70)
        n_va  = int(n * 0.85)

        X_tr, y_tr = X[:n_tr],       y[:n_tr]
        X_va, y_va = X[n_tr:n_va],   y[n_tr:n_va]

        # Scale — fit on train only
        shape_tr = X_tr.shape
        self.scaler.fit(X_tr.reshape(-1, self.cfg["n_features"]))

        def scale(arr):
            s = arr.shape
            return self.scaler.transform(
                arr.reshape(-1, self.cfg["n_features"])
            ).reshape(s).astype(np.float32)

        X_tr = scale(X_tr); X_va = scale(X_va)

        self.target_std  = float(np.std(y_tr))  if np.std(y_tr) > 0 else 0.01
        self.target_mean = float(np.mean(y_tr))

        if verbose:
            print(f"  📊 [{self.algo}] train={len(X_tr)}  val={len(X_va)}")

        # Dataloaders
        bs   = self.cfg["batch_size"]
        dl_tr = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=bs, shuffle=True)
        dl_va = DataLoader(
            TensorDataset(torch.tensor(X_va), torch.tensor(y_va)),
            batch_size=bs)

        # Build model
        self.model = _make_model(self.algo, self.cfg).to(self.device)

        # Optimizer
        opt_name = self.cfg.get("optimizer", "adam")
        if opt_name == "sgd":
            optim = torch.optim.SGD(self.model.parameters(),
                                    lr=0.01, momentum=0.9, nesterov=True)
        else:
            optim = torch.optim.Adam(self.model.parameters(),
                                     lr=self.cfg["lr"],
                                     weight_decay=self.cfg["weight_decay"])

        sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.5, patience=3)
        loss_fn = nn.HuberLoss(delta=0.001)

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        if verbose:
            print(f"  🚀 [{self.algo}] Training on {self.device}...")

        for epoch in range(self.cfg["epochs"]):
            # Train
            self.model.train()
            tr = 0.0
            for xb, yb in dl_tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optim.step()
                tr += loss.item()
            tr /= len(dl_tr)

            # Validate
            self.model.eval()
            va = 0.0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    va += loss_fn(self.model(xb), yb).item()
            va /= len(dl_va)
            sched.step(va)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}/{self.cfg['epochs']} "
                      f"train={tr:.6f} val={va:.6f}")

            if va < best_val:
                best_val   = va
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.cfg["patience"]:
                    if verbose:
                        print(f"    ⏹  Early stop at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})

        self.trained = True
        if verbose:
            print(f"  ✅ [{self.algo}] best_val_loss={best_val:.6f}")
        return True

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_return(self, df, regime_label: int = 0) -> float:
        """Predict 5-day return for the most recent window."""
        if not self.trained or self.model is None:
            return 0.0

        features = build_features(df, regime_label)
        if len(features) < self.cfg["lookback"]:
            return 0.0

        last_window = features.values[-self.cfg["lookback"]:]
        last_scaled = self.scaler.transform(last_window).astype(np.float32)
        x = torch.tensor(last_scaled).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = float(self.model(x).cpu().item())

        return pred * self.target_std

    def predict_with_meta(self, df, current_price: float,
                          regime_label: int = 0) -> dict:
        """
        Returns dict matching PatchTSTForecaster.predict_with_meta() interface.
        Drop-in compatible with daily_advisor_v2.py.
        """
        exp_ret = self.predict_return(df, regime_label)
        pred_px = current_price * (1 + exp_ret)
        return {
            5: {
                "predicted_price": pred_px,
                "expected_return": exp_ret,
                "up_probability":  0.6 if exp_ret > 0 else 0.4,
                "agreement":       1 if exp_ret > 0.003 else 0,
                "model":           f"Lab-{self.algo}",
            }
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        import os
        if self.model is None:
            raise RuntimeError("No model to save — call fit_from_df first")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "algo":         self.algo,
            "cfg":          self.cfg,
            "model_state":  self.model.state_dict(),
            "trained":      self.trained,
            "target_mean":  self.target_mean,
            "target_std":   self.target_std,
            "scaler":       self.scaler,
        }, path)
        print(f"  💾 Lab-{self.algo} saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.algo        = ckpt["algo"]
        self.cfg         = ckpt["cfg"]
        self.trained     = ckpt["trained"]
        self.target_mean = ckpt.get("target_mean", 0.0)
        self.target_std  = ckpt.get("target_std",  0.01)
        self.scaler      = ckpt["scaler"]
        self.model       = _make_model(self.algo, self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"  📂 Lab-{self.algo} loaded ← {path}")
