"""
PatchTST Forecaster
-------------------
Time Series Transformer (Nie et al. 2023) for financial price forecasting.
Trained on full historical data (2004-present) per asset class.
Replaces/complements LightGBM as primary forecaster.

Reference: "A Time Series is Worth 64 Words" — Nie et al., ICLR 2023
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────
CONFIG = {
    "patch_length":  16,
    "stride":         8,
    "lookback":      60,
    "d_model":       64,
    "n_heads":        4,
    "n_layers":       2,
    "dropout":      0.2,
    "prediction_horizon": 5,
    "batch_size":    32,
    "epochs":        50,
    "patience":      10,
    "lr":          1e-4,
    "weight_decay": 1e-4,
    "huber_delta": 0.001,
    "use_revin":   True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ── RevIN: Reversible Instance Normalisation ──────────────────────
class RevIN(nn.Module):
    """
    Normalises each input window independently.
    Critical for 22-year data where 2004 volatility ≠ 2024 volatility.
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        # x: (batch, seq_len, features)
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std  = x.std(dim=1, keepdim=True).detach() + self.eps
            x = (x - self.mean) / self.std
            x = x * self.affine_weight + self.affine_bias
        elif mode == "denorm":
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.std + self.mean
        return x


# ── Patch Embedding ───────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    Splits time series into overlapping patches.
    Each patch = one "token" for the Transformer.
    patch_length=16 means each token covers 16 trading days (~3 weeks).
    """
    def __init__(self, patch_length, stride, d_model, n_features, dropout):
        super().__init__()
        self.patch_length = patch_length
        self.stride       = stride
        # Project each patch to d_model dimensions
        self.projection   = nn.Linear(patch_length * n_features, d_model)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, features)
        B, L, C = x.shape
        # Create patches with stride
        patches = []
        for i in range(0, L - self.patch_length + 1, self.stride):
            patch = x[:, i:i+self.patch_length, :]  # (B, patch_len, C)
            patches.append(patch.reshape(B, -1))      # (B, patch_len*C)
        patches = torch.stack(patches, dim=1)          # (B, n_patches, patch_len*C)
        # Project to d_model
        out = self.projection(patches)                 # (B, n_patches, d_model)
        return self.dropout(out)


# ── PatchTST Model ────────────────────────────────────────────────
class PatchTST(nn.Module):
    """
    PatchTST: Transformer for time series using patch tokenisation.
    Each channel (feature) processed independently then aggregated.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n_feat   = cfg["n_features"]
        d_model  = cfg["d_model"]

        # RevIN normalisation
        if cfg["use_revin"]:
            self.revin = RevIN(n_feat)

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            cfg["patch_length"], cfg["stride"],
            d_model, n_feat, cfg["dropout"]
        )

        # Compute number of patches
        n_patches = (cfg["lookback"] - cfg["patch_length"]) // cfg["stride"] + 1

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = cfg["n_heads"],
            dim_feedforward= d_model * 4,
            dropout        = cfg["dropout"],
            activation     = "gelu",
            batch_first    = True,
            norm_first     = True,   # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                  num_layers=cfg["n_layers"])

        # Output head — predict 5-day ahead price change
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_patches * d_model, 128),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (batch, lookback, n_features)
        if self.cfg["use_revin"]:
            x = self.revin(x, "norm")

        patches = self.patch_embed(x)         # (B, n_patches, d_model)
        patches = patches + self.pos_enc      # add positional info
        enc_out = self.transformer(patches)   # (B, n_patches, d_model)
        out     = self.head(enc_out)          # (B, 1)
        return out.squeeze(-1)                # (B,)


# ── Dataset builder ───────────────────────────────────────────────
def build_features(df, regime_label=0):
    """
    Build feature matrix from OHLCV dataframe.
    Adds technical indicators + regime label as a channel.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Returns and price features
    df["ret"]     = df["Close"].pct_change()
    df["hl_range"]= (df["High"] - df["Low"]) / df["Close"]
    df["sma5"]    = df["Close"].rolling(5).mean() / df["Close"] - 1
    df["sma20"]   = df["Close"].rolling(20).mean() / df["Close"] - 1

    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    # MACD (normalised)
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = (ema12 - ema26) / df["Close"]

    # ATR (normalised)
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"] -df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean() / df["Close"]

    # Volume ratio
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1)

    # Momentum
    df["mom10"] = df["Close"].pct_change(10)

    # HMM regime label as channel
    df["regime"] = regime_label  # 1=bull, 0=sideways, -1=bear

    feat_cols = ["ret","hl_range","sma5","sma20","rsi",
                 "macd","atr","vol_ratio","mom10","regime"]
    df = df[feat_cols].dropna()
    return df


def make_sequences(features_df, target_series, lookback, horizon):
    """
    Create (X, y) sequences for training.
    X: lookback window of features
    y: price horizon days ahead (as return)
    """
    X, y = [], []
    feat = features_df.values
    tgt  = target_series.values

    for i in range(lookback, len(feat) - horizon):
        X.append(feat[i-lookback:i])
        # Target: return over next `horizon` days
        current = tgt[i]
        future  = tgt[i + horizon]
        y.append((future - current) / (current + 1e-8))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Main Forecaster class ─────────────────────────────────────────
class PatchTSTForecaster:
    """
    Drop-in replacement for LightGBMForecaster.
    Same interface: fit(X, y) → predict_with_meta(X, current_price)
    """

    def __init__(self, cfg=None):
        self.cfg   = cfg or CONFIG
        self.cfg["n_features"] = 10  # matches build_features output
        self.model = None
        self.scaler= StandardScaler()
        self.device= torch.device(self.cfg["device"])
        self.trained    = False
        self.target_std  = 0.01
        self.target_mean = 0.0

    def fit_from_df(self, df, regime_label=0, verbose=True):
        """
        Train PatchTST on a full OHLCV dataframe.
        This is the main training method — uses full history.
        """
        if verbose:
            print(f"  🔧 Building features from {len(df)} rows...")

        features = build_features(df, regime_label)
        prices   = df["Close"].reindex(features.index)

        X, y = make_sequences(features,
                               prices,
                               self.cfg["lookback"],
                               self.cfg["prediction_horizon"])

        if len(X) < 100:
            print(f"  ⚠️  Too few sequences ({len(X)}) — skipping")
            return False

        # Chronological split
        n      = len(X)
        n_train= int(n * 0.70)
        n_val  = int(n * 0.85)

        X_tr, y_tr = X[:n_train],   y[:n_train]
        X_va, y_va = X[n_train:n_val], y[n_train:n_val]

        # Scale features (fit on train only)
        shape = X_tr.shape
        X_tr_flat = X_tr.reshape(-1, shape[-1])
        self.scaler.fit(X_tr_flat)
        X_tr = self.scaler.transform(X_tr_flat).reshape(shape)
        X_va = self.scaler.transform(
            X_va.reshape(-1, shape[-1])).reshape(X_va.shape)

        # Store target std for denormalisation at prediction time
        self.target_std = float(np.std(y_tr)) if np.std(y_tr) > 0 else 0.01
        self.target_mean = float(np.mean(y_tr))

        if verbose:
            print(f"  📊 Sequences — train: {len(X_tr)}  val: {len(X_va)}")
            print(f"  📊 Target stats — mean: {self.target_mean:.4f}  std: {self.target_std:.4f}")

        # Build dataloaders
        dl_tr = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=self.cfg["batch_size"], shuffle=True)
        dl_va = DataLoader(
            TensorDataset(torch.tensor(X_va), torch.tensor(y_va)),
            batch_size=self.cfg["batch_size"])

        # Build model
        self.model = PatchTST(self.cfg).to(self.device)
        optim  = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"])
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.cfg["epochs"])
        loss_fn= nn.HuberLoss(delta=self.cfg["huber_delta"])

        best_val = float("inf"); patience_ctr = 0
        best_state = None

        if verbose:
            print(f"  🚀 Training PatchTST on {self.device}...")

        for epoch in range(self.cfg["epochs"]):
            # Train
            self.model.train()
            tr_loss = 0
            for xb, yb in dl_tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optim.step()
                tr_loss += loss.item()
            sched.step()

            # Validate
            self.model.eval()
            va_loss = 0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    va_loss += loss_fn(self.model(xb), yb).item()

            tr_loss /= len(dl_tr); va_loss /= len(dl_va)

            if verbose and (epoch+1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}/{self.cfg['epochs']} "
                      f"train={tr_loss:.6f} val={va_loss:.6f}")

            # Early stopping
            if va_loss < best_val:
                best_val   = va_loss
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg["patience"]:
                    if verbose:
                        print(f"    ⏹  Early stop at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})

        self.trained = True
        if verbose:
            print(f"  ✅ PatchTST trained  best_val_loss={best_val:.6f}")
        return True

    def predict_return(self, df, regime_label=0):
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
            pred_return = float(self.model(x).cpu().item())

        # Scale output back — model outputs are in normalised space
        # Multiply by target_std stored during training
        pred_return = pred_return * self.target_std

        return float(pred_return)

    def predict_with_meta(self, df, current_price, regime_label=0):
        """
        Returns dict matching LightGBMForecaster.predict_with_meta() interface.
        """
        exp_ret = self.predict_return(df, regime_label)
        pred_px = current_price * (1 + exp_ret)

        return {
            5: {
                "predicted_price":  pred_px,
                "expected_return":  exp_ret,
                "up_probability":   0.6 if exp_ret > 0 else 0.4,
                "agreement":        1 if exp_ret > 0.003 else 0,
                "model":            "PatchTST",
            }
        }

    def save(self, path: str):
        """Save trained PatchTST model to disk."""
        import torch, os, pickle
        if self.model is None:
            raise RuntimeError("No model to save — call fit_from_df first")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state":  self.model.state_dict(),
            "cfg":          self.cfg,
            "trained":      self.trained,
            "target_mean":  getattr(self, "target_mean", None),
            "target_std":   getattr(self, "target_std",  None),
            "scaler":       getattr(self, "scaler",      None),
        }, path)
        print(f"  💾 PatchTST saved → {path}")

    def load(self, path: str):
        """Load trained PatchTST model from disk."""
        import torch
        from src.forecast.patchtst_forecast import PatchTST
        ckpt = torch.load(path, map_location=self.device)
        self.cfg         = ckpt["cfg"]
        self.trained     = ckpt["trained"]
        if ckpt.get("target_mean") is not None:
            self.target_mean = ckpt["target_mean"]
        if ckpt.get("target_std") is not None:
            self.target_std  = ckpt["target_std"]
        if ckpt.get("scaler") is not None:
            self.scaler      = ckpt["scaler"]
        self.model = PatchTST(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"  📂 PatchTST loaded ← {path}")

