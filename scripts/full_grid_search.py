#!/usr/bin/env python3
"""
Full Grid Search — All 48 Assets × All Deep Learning Concepts
==============================================================
Covers every core ML/DL concept not yet in the system:

  ACTIVATIONS  : ReLU, GELU, Tanh, LeakyReLU, ELU, SiLU/Swish, Mish, SELU
  NORMALISATION: LayerNorm, BatchNorm, InstanceNorm, RMSNorm, GroupNorm, None
  WEIGHT INIT  : Kaiming (He), Xavier (Glorot), Orthogonal, Normal, Sparse
  OPTIMISERS   : Adam, AdamW, SGD+Nesterov, RMSprop, Adagrad
  LR SCHEDULERS: None, Cosine Annealing, StepLR, Warmup+Cosine, ReduceOnPlateau
  REGULARISA.  : Dropout (0→0.3), L1, L2, ElasticNet, DropPath (stochastic depth)
  CRITIC LOSS  : MSE, MAE (L1), Huber, Log-Cosh, Quantile
  ARCHITECTURE : width (64/128/256), depth (1/2/3), residual connections
  RL REWARD    : Sharpe, Sortino, Calmar, Simple, DrawdownPenalised
  REPLAY BUFFER: Uniform, Prioritised (PER)
  RETURN ESTIM.: 1-step TD, N-step (N=3,5)
  GAMMA        : 0.95, 0.99, 1.0

Each config is trained on the 70% train split and evaluated on the 30% holdout.
Results are saved incrementally (crash-safe) to CSV.

Usage:
    python scripts/full_grid_search.py                    # all assets, all configs
    python scripts/full_grid_search.py --assets NVDA BTC-USD   # specific assets
    python scripts/full_grid_search.py --episodes 30           # more episodes
    python scripts/full_grid_search.py --resume               # skip already done
    python scripts/full_grid_search.py --analyse              # analyse existing CSV
"""

import os, sys, csv, time, warnings, argparse, random
from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional, Tuple
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from src.rl.trading_env  import TradingEnvironment, build_signals

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
OUT_CSV  = "data/output/grid_search_results.csv"
os.makedirs("data/output", exist_ok=True)

# Full 48-asset universe  (mirrors train_all_models.py)
UNIVERSE = [
    ("^GSPC","S&P 500","equity","SAC"),    ("^IXIC","NASDAQ","equity","SAC"),
    ("^DJI","Dow Jones","equity","SAC"),   ("AAPL","Apple","equity","SAC"),
    ("MSFT","Microsoft","equity","SAC"),   ("NVDA","NVIDIA","equity","SAC"),
    ("TSLA","Tesla","equity","SAC"),       ("AMZN","Amazon","equity","SAC"),
    ("META","Meta","equity","SAC"),        ("GOOGL","Alphabet","equity","SAC"),
    ("JPM","JPMorgan","equity","SAC"),
    ("CRWV","CoreWeave","equity","TD3"),   ("NBIS","Nebius","equity","TD3"),
    ("IREN","IREN","equity","TD3"),        ("MU","Micron","equity","TD3"),
    ("SMCI","Super Micro","equity","TD3"), ("ARM","Arm Holdings","equity","TD3"),
    ("TSM","TSMC","equity","TD3"),         ("VRT","Vertiv","equity","TD3"),
    ("MRVL","Marvell","equity","TD3"),
    ("BTC-USD","Bitcoin","crypto","PPO"),  ("ETH-USD","Ethereum","crypto","PPO"),
    ("SOL-USD","Solana","crypto","PPO"),   ("BNB-USD","BNB","crypto","PPO"),
    ("XRP-USD","XRP","crypto","PPO"),
    ("EURUSD=X","EUR/USD","forex","SAC"),  ("GBPUSD=X","GBP/USD","forex","SAC"),
    ("USDJPY=X","USD/JPY","forex","SAC"),  ("USDCHF=X","USD/CHF","forex","SAC"),
    ("AUDUSD=X","AUD/USD","forex","SAC"),  ("NZDUSD=X","NZD/USD","forex","SAC"),
    ("USDCAD=X","USD/CAD","forex","SAC"),  ("EURGBP=X","EUR/GBP","forex","SAC"),
    ("EURJPY=X","EUR/JPY","forex","SAC"),  ("GBPJPY=X","GBP/JPY","forex","SAC"),
    ("AUDNZD=X","AUD/NZD","forex","SAC"),  ("USDNGN=X","USD/NGN","forex","SAC"),
    ("EURNGN=X","EUR/NGN","forex","SAC"),  ("USDZAR=X","USD/ZAR","forex","SAC"),
    ("USDKES=X","USD/KES","forex","SAC"),  ("USDGHS=X","USD/GHS","forex","SAC"),
    ("GC=F","Gold","commodity","SAC"),     ("SI=F","Silver","commodity","SAC"),
    ("CL=F","Crude Oil","commodity","SAC"),("NG=F","Natural Gas","commodity","SAC"),
    ("HG=F","Copper","commodity","SAC"),   ("ZW=F","Wheat","commodity","SAC"),
    ("ZC=F","Corn","commodity","SAC"),
]


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Cfg:
    name:         str
    # Architecture
    hidden:       int   = 128
    n_layers:     int   = 2
    activation:   str   = "gelu"
    norm:         str   = "layer"
    residual:     bool  = False
    # Regularisation
    dropout:      float = 0.10
    weight_decay: float = 0.0
    l1_lambda:    float = 0.0
    droppath:     float = 0.0   # stochastic depth rate
    # Init
    init:         str   = "kaiming"
    # Optimiser
    opt:          str   = "adam"
    lr:           float = 3e-4
    # LR scheduler
    scheduler:    str   = "none"
    # Critic loss
    loss:         str   = "mse"
    # RL
    gamma:        float = 0.99
    reward_fn:    str   = "sharpe"
    per:          bool  = False   # prioritised experience replay
    n_steps:      int   = 1       # N-step returns
    episodes:     int   = 20


def build_grid(episodes: int) -> List[Cfg]:
    """
    35-config grid: one change at a time from baseline.
    This is ablation study design — isolating one variable per experiment
    so you can attribute any performance difference to that variable alone.
    """
    e = episodes
    return [
        # ── Baseline ─────────────────────────────────────────────────────────
        Cfg("baseline",                                              episodes=e),

        # ── Activations ─────────────────────────────────────────────────────
        # Theory: activation introduces non-linearity. Without it, N stacked
        # linear layers = still just one linear layer.
        Cfg("act_relu",       activation="relu",                    episodes=e),
        # ReLU: max(0,x). Simple, fast, can cause "dying ReLU" (neuron stuck at 0)
        Cfg("act_tanh",       activation="tanh",                    episodes=e),
        # Tanh: outputs (-1,1). Symmetric, but saturates → vanishing gradients
        Cfg("act_leaky_relu", activation="leaky_relu",              episodes=e),
        # LeakyReLU: small slope for x<0, fixes dying ReLU
        Cfg("act_elu",        activation="elu",                     episodes=e),
        # ELU: smooth for negative inputs, self-normalising tendency
        Cfg("act_silu",       activation="silu",                    episodes=e),
        # SiLU (Swish): x*sigmoid(x). Used in EfficientNet, modern networks
        Cfg("act_mish",       activation="mish",                    episodes=e),
        # Mish: x*tanh(softplus(x)). Smoother than SiLU, often marginally better

        # ── Normalisation ────────────────────────────────────────────────────
        # Normalisation stabilises training by keeping activations in a
        # sensible range. Different types suit different architectures.
        Cfg("norm_batch",     norm="batch",                         episodes=e),
        # BatchNorm: norm over batch dimension. Bad for small batches (RL)
        Cfg("norm_rms",       norm="rms",                           episodes=e),
        # RMSNorm: simpler than LayerNorm (no mean subtraction), used in LLaMA
        Cfg("norm_group",     norm="group",                         episodes=e),
        # GroupNorm: between batch and instance norm. Stable for small batches
        Cfg("norm_none",      norm="none",                          episodes=e),
        # No norm: fastest, but training can be unstable for deep networks

        # ── Architecture: width and depth ────────────────────────────────────
        # Width (neurons): more = higher capacity, more compute, overfitting risk
        # Depth (layers): deeper = more abstract features, vanishing gradient risk
        Cfg("width_64",       hidden=64,                            episodes=e),
        Cfg("width_256",      hidden=256,                           episodes=e),
        Cfg("depth_1",        n_layers=1,                           episodes=e),
        Cfg("depth_3",        n_layers=3,                           episodes=e),

        # ── Residual connections ─────────────────────────────────────────────
        # Skip connections: output = f(x) + x  (identity shortcut)
        # Why it helps: gradients flow directly back, solving vanishing gradient.
        # Introduced by ResNet (He et al. 2015). Now used everywhere.
        Cfg("residual",       residual=True,                        episodes=e),

        # ── Regularisation ───────────────────────────────────────────────────
        # Regularisation prevents overfitting by penalising model complexity.
        Cfg("dropout_0",      dropout=0.0,                          episodes=e),
        Cfg("dropout_0.3",    dropout=0.3,                          episodes=e),
        Cfg("l1_reg",         l1_lambda=1e-4,                       episodes=e),
        # L1: penalises |w|, produces sparse weights. Bayesian: Laplace prior.
        Cfg("l2_reg",         weight_decay=1e-3,                    episodes=e),
        # L2: penalises w², produces small weights. Bayesian: Gaussian prior.
        Cfg("elastic_net",    weight_decay=1e-4, l1_lambda=1e-4,    episodes=e),
        # ElasticNet: L1+L2 combined. Best of both: sparse + small weights.
        Cfg("droppath_0.1",   droppath=0.1,                         episodes=e),
        # DropPath (Stochastic Depth): drops entire residual branches randomly.
        # Each layer only trains on ~(1-p) fraction of samples → ensemble effect.

        # ── Weight initialisation ────────────────────────────────────────────
        # Init sets starting point. Bad init → vanishing/exploding gradients.
        Cfg("init_xavier",    init="xavier",                         episodes=e),
        # Xavier/Glorot: var = 2/(fan_in+fan_out). Designed for tanh/sigmoid.
        Cfg("init_orthogonal",init="orthogonal",                     episodes=e),
        # Orthogonal: preserves gradient norms through layers. Good for RNNs.
        Cfg("init_normal",    init="normal",                         episodes=e),
        # Normal(0, 0.01): small random init. Simple but can be suboptimal.

        # ── Optimisers ───────────────────────────────────────────────────────
        # Optimiser controls HOW gradients update weights.
        Cfg("adamw",          opt="adamw",  weight_decay=1e-4,       episodes=e),
        # AdamW: Adam + true L2 decay (not the broken Adam+weight_decay).
        # Preferred for most modern architectures. Decouples decay from LR.
        Cfg("sgd",            opt="sgd",                             episodes=e),
        # SGD+Nesterov: looks ahead before computing gradient.
        # Slower than Adam but often better final generalisation.
        Cfg("rmsprop",        opt="rmsprop",                         episodes=e),
        # RMSprop: divides by running avg of squared gradient. Original DQN used this.
        Cfg("adagrad",        opt="adagrad",                         episodes=e),
        # Adagrad: adapts LR per-parameter based on historical gradient magnitude.
        # Good for sparse gradients (NLP), but LR shrinks monotonically.

        # ── Learning rate ────────────────────────────────────────────────────
        Cfg("lr_1e3",         lr=1e-3,                               episodes=e),
        Cfg("lr_1e4",         lr=1e-4,                               episodes=e),

        # ── LR Schedulers ────────────────────────────────────────────────────
        # Schedulers change LR over training. Often improves final performance.
        Cfg("sched_cosine",   scheduler="cosine",                    episodes=e),
        # Cosine Annealing: LR decreases as cos curve → converges smoothly
        Cfg("sched_step",     scheduler="step",                      episodes=e),
        # StepLR: halves LR every N episodes → discrete drops in LR
        Cfg("sched_warmup",   scheduler="warmup_cosine",             episodes=e),
        # Warmup: LR linearly increases for first ~10% then cosine decay.
        # Critical for Transformers; also helps RL agents stabilise early.
        Cfg("sched_plateau",  scheduler="plateau",                   episodes=e),
        # ReduceLROnPlateau: cuts LR when training stalls. Adaptive.

        # ── Critic loss functions ────────────────────────────────────────────
        # Q-network loss: how we measure "how wrong is the Q estimate?"
        Cfg("loss_mae",       loss="mae",                            episodes=e),
        # MAE (L1): |Q_pred - Q_target|. Robust to reward outliers.
        Cfg("loss_huber",     loss="huber",                          episodes=e),
        # Huber: MSE for small errors, MAE for large. Best of both worlds.
        # Financial data has fat tails → Huber/MAE > MSE theoretically.
        Cfg("loss_logcosh",   loss="logcosh",                        episodes=e),
        # Log-Cosh: log(cosh(error)). Smooth everywhere, ~MAE for large errors.
        Cfg("loss_quantile",  loss="quantile",                       episodes=e),
        # Quantile (pinball): asymmetric loss. Penalises under/over-prediction
        # differently. Good for capturing distribution asymmetry in finance.

        # ── RL discount factor ───────────────────────────────────────────────
        # γ controls time horizon. Low γ = myopic, High γ = far-sighted.
        Cfg("gamma_0.95",     gamma=0.95,                            episodes=e),
        # γ=0.95: cares ~20 steps ahead. Better for intraday trading.
        Cfg("gamma_1.0",      gamma=1.0,                             episodes=e),
        # γ=1.0: infinite horizon. Can cause instability but works for stable envs.

        # ── Reward functions ─────────────────────────────────────────────────
        # The reward signal shapes WHAT the agent learns to optimise.
        # This is one of the most underexplored dimensions in RL for finance.
        Cfg("reward_sortino", reward_fn="sortino",                   episodes=e),
        # Sortino: Sharpe but only penalises DOWNSIDE volatility.
        # Better aligned with investor preferences (nobody minds upside vol).
        Cfg("reward_calmar",  reward_fn="calmar",                    episodes=e),
        # Calmar: return / max_drawdown. Penalises large drawdowns heavily.
        Cfg("reward_simple",  reward_fn="simple",                    episodes=e),
        # Simple return: just the P&L. No risk adjustment. Often overtrades.
        Cfg("reward_dd_pen",  reward_fn="dd_penalised",              episodes=e),
        # Drawdown-penalised: Sharpe minus extra penalty when in drawdown.

        # ── Prioritised Experience Replay (PER) ──────────────────────────────
        # Uniform replay: samples all past transitions equally.
        # PER: samples high-TD-error transitions more frequently.
        # Why: agent learns MORE from transitions it got very wrong.
        # Introduced by Schaul et al. (2016), used in Rainbow DQN.
        Cfg("per",            per=True,                              episodes=e),

        # ── N-step returns ───────────────────────────────────────────────────
        # 1-step TD: r + γV(s') — low variance, high bias
        # N-step:    Σ γ^i*r_i + γ^N*V(s_N) — lower bias, higher variance
        # N=3-5 is the sweet spot for most RL problems.
        Cfg("nstep_3",        n_steps=3,                             episodes=e),
        Cfg("nstep_5",        n_steps=5,                             episodes=e),

        # ── Combo: educated best guess ────────────────────────────────────────
        # Based on what generally works well across financial RL:
        Cfg("combo_best",
            activation="silu", hidden=128, n_layers=2,
            norm="layer", residual=True, dropout=0.1,
            opt="adamw", weight_decay=1e-4, lr=3e-4,
            scheduler="cosine", loss="huber", gamma=0.99,
            reward_fn="sortino", per=False, n_steps=3,   episodes=e),

        # Wider+deeper with residual
        Cfg("combo_deep",
            activation="gelu", hidden=256, n_layers=3,
            norm="layer", residual=True, dropout=0.2,
            opt="adamw", weight_decay=1e-4,
            scheduler="warmup_cosine", loss="huber",      episodes=e),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# NEW DL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Used in LLaMA, Mistral, Qwen. Simpler than LayerNorm:
    no mean subtraction, just RMS scaling.
    Empirically as good as LayerNorm at 30% less compute.
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


def get_activation(name: str) -> nn.Module:
    return {
        "relu":       nn.ReLU(),
        "gelu":       nn.GELU(),
        "tanh":       nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "elu":        nn.ELU(),
        "silu":       nn.SiLU(),    # Swish: x * sigmoid(x)
        "mish":       nn.Mish(),    # x * tanh(softplus(x))
        "selu":       nn.SELU(),    # self-normalising (needs specific init)
    }.get(name, nn.GELU())


def get_norm(name: str, size: int) -> Optional[nn.Module]:
    if name == "layer":    return nn.LayerNorm(size)
    if name == "batch":    return nn.BatchNorm1d(size)
    if name == "rms":      return RMSNorm(size)
    if name == "group":    return nn.GroupNorm(min(8, size), size)
    if name == "instance": return nn.InstanceNorm1d(size)
    return None   # "none"


def apply_init(module: nn.Module, method: str):
    """
    Weight initialisation — sets the starting distribution of weights.

    Kaiming/He:    Designed for ReLU/SiLU. var = 2/fan_in.
                   Keeps signal variance stable through ReLU layers.
    Xavier/Glorot: Designed for Tanh/Sigmoid. var = 2/(fan_in+fan_out).
    Orthogonal:    W^T W = I. Preserves gradient norms exactly.
                   Good for deep networks and RNNs.
    Normal:        N(0, 0.01). Simple but can be suboptimal.
    Sparse:        Most weights zeroed, a few randomly initialised.
                   Forces the network to be sparse from the start.
    """
    for m in module.modules():
        if not isinstance(m, nn.Linear):
            continue
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif method == "orthogonal":
            nn.init.orthogonal_(m.weight)
        elif method == "normal":
            nn.init.normal_(m.weight, 0, 0.01)
        elif method == "sparse":
            nn.init.sparse_(m.weight, sparsity=0.5)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DropPath(nn.Module):
    """
    Stochastic Depth / DropPath.
    Randomly drops the entire residual branch during training.
    At training: branch output × Bernoulli(1-p) / (1-p)
    At inference: full branch (no dropping)
    Used in DeiT, Swin Transformer, ConvNeXt.
    Trains an implicit ensemble of networks with different depths.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.p = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        keep = torch.rand(x.shape[0], 1, device=x.device) > self.p
        return x * keep / (1 - self.p)


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block: output = x + f(Norm(x))
    Pre-activation (He et al. 2016) is more stable than post-activation.
    The identity shortcut x allows gradients to bypass layers entirely,
    solving the vanishing gradient problem for deep networks.
    """
    def __init__(self, dim: int, cfg: 'Cfg'):
        super().__init__()
        self.fc       = nn.Linear(dim, dim)
        norm          = get_norm(cfg.norm, dim)
        self.norm     = norm if norm is not None else nn.Identity()
        self.act      = get_activation(cfg.activation)
        self.drop     = nn.Dropout(cfg.dropout)
        self.droppath = DropPath(cfg.droppath)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.droppath(self.drop(self.act(self.norm(self.fc(x)))))


def build_net(in_dim: int, out_dim: int, cfg: 'Cfg') -> nn.Sequential:
    """Build a fully configurable MLP with optional residual connections."""
    layers  = []
    current = in_dim

    for i in range(cfg.n_layers):
        if cfg.residual and i > 0 and current == cfg.hidden:
            # Add residual block after the first layer (dims must match)
            layers.append(ResidualBlock(cfg.hidden, cfg))
        else:
            layers.append(nn.Linear(current, cfg.hidden))
            norm = get_norm(cfg.norm, cfg.hidden)
            if norm is not None:
                layers.append(norm)
            layers.append(get_activation(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            if cfg.droppath > 0:
                layers.append(DropPath(cfg.droppath))
            current = cfg.hidden

    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class ConfigNet(nn.Module):
    """Fully configurable actor + twin critics."""
    def __init__(self, state_dim: int, action_dim: int, cfg: 'Cfg'):
        super().__init__()
        self.actor = build_net(state_dim, action_dim, cfg)
        self.q1    = build_net(state_dim, action_dim, cfg)
        self.q2    = build_net(state_dim, action_dim, cfg)

    def get_actor(self, x): return self.actor(x)
    def get_q(self, x):     return self.q1(x), self.q2(x)


def get_optimizer(cfg: 'Cfg', params):
    """
    Optimiser factory.
    The optimiser controls HOW the gradient signal updates weights.

    Adam:    Momentum + adaptive LR per parameter. Fast, works out of the box.
    AdamW:   Decouples weight decay from gradient update (Loshchilov & Hutter 2019).
             True L2 regularisation. Better generalisation than Adam.
    SGD:     Stochastic gradient descent + Nesterov momentum.
             Slower to converge than Adam, but sometimes reaches better minima.
    RMSprop: Divides gradient by running RMS. Used in original DQN (Mnih 2015).
    Adagrad: Per-parameter adaptive LR. LR shrinks monotonically → can stall.
    """
    wd = cfg.weight_decay
    lr = cfg.lr
    if cfg.opt == "adam":    return torch.optim.Adam(params, lr=lr)
    if cfg.opt == "adamw":   return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if cfg.opt == "sgd":     return torch.optim.SGD(params, lr=lr, momentum=0.9,
                                                     weight_decay=wd, nesterov=True)
    if cfg.opt == "rmsprop": return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    if cfg.opt == "adagrad": return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr)


def get_scheduler(cfg: 'Cfg', optimizer, episodes: int):
    """
    LR Scheduler factory.

    None:            Constant LR throughout.
    Cosine:          LR decreases as a cosine curve (smooth decay).
    Step:            LR halves every ⌊episodes/3⌋ steps (discrete drops).
    Warmup+Cosine:   LR linearly increases for 10% of training, then cosine decay.
                     Critical for Transformers. Helps RL agents stabilise early.
    ReduceOnPlateau: Cuts LR when monitored metric stops improving. Adaptive.
    """
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=episodes, eta_min=cfg.lr * 0.01)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, episodes // 3), gamma=0.5)
    if cfg.scheduler == "warmup_cosine":
        warmup = max(1, episodes // 10)
        def lr_fn(ep):
            if ep < warmup:
                return ep / warmup          # linear warmup
            progress = (ep - warmup) / max(1, episodes - warmup)
            return 0.5 * (1 + np.cos(np.pi * progress))  # cosine decay
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5)
    return None


def compute_critic_loss(pred, target, cfg: 'Cfg') -> torch.Tensor:
    """
    Critic (Q-network) regression loss.

    MSE:      (y_pred - y_true)²        Penalises large errors heavily.
    MAE:      |y_pred - y_true|         Robust to outliers (fat-tailed rewards).
    Huber:    MSE for |err|<δ, MAE for |err|≥δ. Best of both.
    Log-Cosh: log(cosh(err)) ≈ (err²/2) for small, |err| for large.
              Smooth everywhere, second-order differentiable.
    Quantile: Pinball loss. Asymmetric — penalises over/under separately.
              α=0.5 gives MAE. Used in quantile regression.
              Useful when reward distribution is skewed (as in finance).
    """
    if cfg.loss == "mse":      return F.mse_loss(pred, target)
    if cfg.loss == "mae":      return F.l1_loss(pred, target)
    if cfg.loss == "huber":    return F.huber_loss(pred, target, delta=1.0)
    if cfg.loss == "logcosh":
        err = pred - target
        return torch.mean(torch.log(torch.cosh(err + 1e-8)))
    if cfg.loss == "quantile":
        alpha = 0.5   # median regression (equivalent to MAE)
        err   = target - pred
        return torch.mean(torch.max(alpha * err, (alpha - 1) * err))
    return F.mse_loss(pred, target)


def compute_l1_penalty(model: nn.Module) -> torch.Tensor:
    """
    L1 penalty on all weight tensors.
    Adds Σ|w| to the loss → encourages sparse weight matrices.
    L1 is the LASSO in supervised learning.
    From a Bayesian perspective: equivalent to a Laplace prior on weights.
    """
    return sum(p.abs().sum() for p in model.parameters() if p.requires_grad)


# ── Reward functions ──────────────────────────────────────────────────────────

def compute_reward(returns: list, portfolio_vals: list, step_ret: float,
                   reward_fn: str) -> float:
    """
    Reward function factory.

    The reward signal defines WHAT the agent learns to optimise.
    Changing it fundamentally changes the agent's behaviour.

    Sharpe:    E[r] / std(r)          — classic risk-adjusted return
    Sortino:   E[r] / std(r<0)        — only penalises DOWNSIDE volatility
               Better aligned with investor preferences (upside vol is good)
    Calmar:    E[r] / max_drawdown    — penalises large peak-to-trough drops
    Simple:    r directly             — no risk adjustment, often overtrades
    DD-pen:    Sharpe - k*drawdown    — Sharpe with extra drawdown penalty
    """
    W = 20
    if len(returns) < 2:
        return float(step_ret * 100)

    recent = np.array(returns[-W:])

    if reward_fn == "sharpe":
        std = np.std(recent)
        return float(np.mean(recent) / (std + 1e-8))

    if reward_fn == "sortino":
        downside = recent[recent < 0]
        d_std    = np.std(downside) if len(downside) > 1 else 1e-8
        return float(np.mean(recent) / (d_std + 1e-8))

    if reward_fn == "calmar":
        vals = np.array(portfolio_vals[-W:])
        peak = np.maximum.accumulate(vals)
        dd   = np.min((vals - peak) / (peak + 1e-9))
        return float(np.mean(recent) / (abs(dd) + 1e-8))

    if reward_fn == "simple":
        return float(step_ret * 100)

    if reward_fn == "dd_penalised":
        vals = np.array(portfolio_vals[-W:])
        peak = np.maximum.accumulate(vals)
        dd   = float(np.min((vals - peak) / (peak + 1e-9)))
        std  = np.std(recent)
        sharpe = float(np.mean(recent) / (std + 1e-8))
        return sharpe + 2.0 * dd   # penalise drawdown twice

    return float(step_ret * 100)


# ── Prioritised Experience Replay ─────────────────────────────────────────────

class PrioritisedBuffer:
    """
    Prioritised Experience Replay (PER) — Schaul et al. (2016).

    KEY IDEA: not all transitions are equally informative.
    Transitions with high TD-error (things the agent got very wrong)
    are sampled more frequently → agent learns from its mistakes faster.

    Priority: p_i = (|δ_i| + ε)^α
      |δ_i| = TD error (how wrong was the Q estimate?)
      ε     = small constant to ensure all transitions have some chance
      α     = how much to prioritise (0=uniform, 1=full prioritisation)

    Importance sampling weights correct for the sampling bias:
      w_i = (1/N * 1/p_i)^β
    β anneals from 0.4 → 1.0 over training.
    """
    def __init__(self, capacity: int = 50000, alpha: float = 0.6):
        self.cap      = capacity
        self.alpha    = alpha
        self.buffer   = []
        self.prios    = np.zeros(capacity, dtype=np.float32)
        self.pos      = 0
        self.max_prio = 1.0

    def push(self, s, a, r, ns, done):
        if len(self.buffer) < self.cap:
            self.buffer.append((s, a, r, ns, done))
        else:
            self.buffer[self.pos] = (s, a, r, ns, done)
        self.prios[self.pos] = self.max_prio ** self.alpha
        self.pos = (self.pos + 1) % self.cap

    def sample(self, batch_size: int, beta: float = 0.4):
        n     = len(self.buffer)
        prios = self.prios[:n]
        probs = prios / prios.sum()
        idxs  = np.random.choice(n, batch_size, p=probs, replace=False
                                  if n >= batch_size else True)
        weights = (n * probs[idxs]) ** (-beta)
        weights /= weights.max()
        batch   = [self.buffer[i] for i in idxs]
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r),
                np.array(ns), np.array(d),
                torch.tensor(weights, dtype=torch.float32).to(DEVICE),
                idxs)

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray):
        for i, e in zip(idxs, td_errors):
            prio = (abs(e) + 1e-5) ** self.alpha
            self.prios[i]  = prio
            self.max_prio  = max(self.max_prio, prio)

    def __len__(self): return len(self.buffer)


# ── N-step returns ────────────────────────────────────────────────────────────

class NStepBuffer:
    """
    N-step return buffer.

    1-step TD: target = r_0 + γ * V(s_1)
    N-step TD: target = r_0 + γ*r_1 + γ²*r_2 + ... + γ^(N-1)*r_{N-1} + γ^N * V(s_N)

    Trade-off:
      1-step: low variance, high bias (V(s') may be wrong at start)
      N-step: lower bias (more of the actual reward used), higher variance
      N=3-5: empirically best for most tasks

    Rainbow DQN uses N=3. SAC papers use N=1 but N=5 often helps.
    """
    def __init__(self, n: int, gamma: float):
        self.n      = n
        self.gamma  = gamma
        self.window = deque(maxlen=n)

    def add(self, s, a, r, ns, done) -> Optional[Tuple]:
        self.window.append((s, a, r, ns, done))
        if len(self.window) < self.n and not done:
            return None
        # Compute discounted return over window
        g     = 0.0
        for i, (_, _, ri, _, _) in enumerate(self.window):
            g += ri * (self.gamma ** i)
        s0, a0     = self.window[0][0],  self.window[0][1]
        s_n, done_n = self.window[-1][3], self.window[-1][4]
        if done:
            self.window.clear()
        return s0, a0, g, s_n, done_n

    def flush(self):
        self.window.clear()


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING ONE CONFIG ON ONE ASSET
# ══════════════════════════════════════════════════════════════════════════════

def train_eval(df_train: pd.DataFrame, df_test: pd.DataFrame,
               sig_train: np.ndarray, sig_test: np.ndarray,
               cfg: Cfg) -> dict:
    """Train and evaluate one config. Returns metric dict."""
    t0        = time.time()
    env_train = TradingEnvironment(df_train, sig_train)
    state_dim = env_train.state_dim

    # Build net + target
    net    = ConfigNet(state_dim, 3, cfg).to(DEVICE)
    tgt    = ConfigNet(state_dim, 3, cfg).to(DEVICE)
    tgt.load_state_dict(net.state_dict())
    apply_init(net, cfg.init)

    # Optimisers (separate actor / critic — mirrors SAC design)
    a_opt = get_optimizer(cfg, list(net.actor.parameters()))
    c_opt = get_optimizer(cfg, list(net.q1.parameters()) +
                              list(net.q2.parameters()))
    a_sch = get_scheduler(cfg, a_opt, cfg.episodes)
    c_sch = get_scheduler(cfg, c_opt, cfg.episodes)

    # Entropy (SAC automatic tuning)
    log_alpha      = torch.tensor(np.log(0.2), dtype=torch.float32,
                                   requires_grad=True, device=DEVICE)
    target_entropy = -np.log(1.0 / 3) * 0.98
    al_opt         = torch.optim.Adam([log_alpha], lr=1e-4)

    # Replay buffer
    BATCH  = 64
    TAU    = 0.005
    buf    = PrioritisedBuffer() if cfg.per else deque(maxlen=50000)
    nstep  = NStepBuffer(cfg.n_steps, cfg.gamma) if cfg.n_steps > 1 else None
    beta   = 0.4   # PER importance sampling (anneals to 1)

    best_sharpe = -999
    best_state  = None
    ep_sharpes  = []

    for ep in range(cfg.episodes):
        state        = env_train.reset()
        returns_hist = []
        port_vals    = [env_train.initial_capital]

        while True:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = net.get_actor(s_t)
                action = torch.distributions.Categorical(logits=logits).sample().item()

            ns, raw_r, done, info = env_train.step(action)
            returns_hist.append(info.get("step_return", raw_r))
            port_vals.append(info.get("capital", port_vals[-1]))

            # Custom reward
            r = compute_reward(returns_hist, port_vals,
                               info.get("step_return", raw_r), cfg.reward_fn)

            # N-step buffer
            if nstep is not None:
                result = nstep.add(state, action, r, ns, done)
                if result is not None:
                    s0, a0, g, sn, dn = result
                    if cfg.per:
                        buf.push(s0, a0, g, sn, dn)
                    else:
                        buf.append((s0, a0, g, sn, dn))
            else:
                if cfg.per:
                    buf.push(state, action, r, ns, done)
                else:
                    buf.append((state, action, r, ns, done))

            state = ns
            if done:
                if nstep is not None:
                    nstep.flush()
                break

            if len(buf) < BATCH:
                continue

            # ── Sample batch ───────────────────────────────────────
            is_weights = None
            per_idxs   = None
            if cfg.per:
                beta_now = min(1.0, beta + ep / cfg.episodes * 0.6)
                s, a, rw, ns_, d, is_weights, per_idxs = buf.sample(BATCH, beta_now)
            else:
                batch  = random.sample(buf, BATCH)
                s, a, rw, ns_, d = zip(*batch)
                s  = np.array(s);  a  = np.array(a)
                rw = np.array(rw); ns_= np.array(ns_); d = np.array(d)

            s   = torch.tensor(s,   dtype=torch.float32).to(DEVICE)
            a   = torch.tensor(a,   dtype=torch.int64).to(DEVICE)
            rw  = torch.tensor(rw,  dtype=torch.float32).to(DEVICE)
            ns_ = torch.tensor(ns_, dtype=torch.float32).to(DEVICE)
            d   = torch.tensor(d,   dtype=torch.float32).to(DEVICE)
            alpha = log_alpha.exp().detach()

            # ── Critic update ──────────────────────────────────────
            with torch.no_grad():
                tl    = tgt.get_actor(ns_)
                tp    = torch.softmax(tl, -1)
                tlp   = torch.log_softmax(tl, -1)
                tq1, tq2 = tgt.get_q(ns_)
                v_n   = (tp * (torch.min(tq1, tq2) - alpha * tlp)).sum(-1)
                tgt_q = rw + cfg.gamma * (1 - d) * v_n

            q1, q2 = net.get_q(s)
            q1a    = q1.gather(1, a.unsqueeze(1)).squeeze()
            q2a    = q2.gather(1, a.unsqueeze(1)).squeeze()

            c_loss = compute_critic_loss(q1a, tgt_q, cfg) + \
                     compute_critic_loss(q2a, tgt_q, cfg)

            # PER: weight the loss by importance weights
            if is_weights is not None:
                td_err = (q1a - tgt_q).abs().detach().cpu().numpy()
                buf.update_priorities(per_idxs, td_err)
                c_loss = (c_loss * is_weights.to(DEVICE)).mean()

            # L1 regularisation penalty
            if cfg.l1_lambda > 0:
                c_loss = c_loss + cfg.l1_lambda * compute_l1_penalty(net)

            c_opt.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(list(net.q1.parameters()) +
                                     list(net.q2.parameters()), 1.0)
            c_opt.step()

            # ── Actor update ───────────────────────────────────────
            logits   = net.get_actor(s)
            probs    = torch.softmax(logits, -1)
            lp       = torch.log_softmax(logits, -1)
            q1c, q2c = net.get_q(s)
            a_loss   = (probs * (alpha * lp - torch.min(q1c, q2c))).sum(-1).mean()
            if cfg.l1_lambda > 0:
                a_loss = a_loss + cfg.l1_lambda * compute_l1_penalty(net)

            a_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(net.actor.parameters(), 1.0)
            a_opt.step()

            # ── Entropy ────────────────────────────────────────────
            ent     = -(probs.detach() * lp.detach()).sum(-1).mean()
            al_loss = -(log_alpha * (ent - target_entropy).detach())
            al_opt.zero_grad()
            al_loss.backward()
            al_opt.step()

            # ── Soft target update ─────────────────────────────────
            for p, tp_ in zip(net.parameters(), tgt.parameters()):
                tp_.data.copy_(TAU * p.data + (1 - TAU) * tp_.data)

        # LR scheduler step
        perf = env_train.get_performance()
        ep_sharpes.append(perf["sharpe_ratio"])
        if perf["sharpe_ratio"] > best_sharpe:
            best_sharpe = perf["sharpe_ratio"]
            best_state  = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        if a_sch:
            if cfg.scheduler == "plateau":
                a_sch.step(perf["sharpe_ratio"])
                c_sch.step(perf["sharpe_ratio"])
            else:
                a_sch.step()
                if c_sch: c_sch.step()

    if best_state:
        net.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # ── Holdout evaluation ────────────────────────────────────────
    env_test = TradingEnvironment(df_test, sig_test)
    state    = env_test.reset()
    net.eval()
    with torch.no_grad():
        while True:
            s_t    = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action = net.get_actor(s_t).argmax(-1).item()
            state, _, done, _ = env_test.step(action)
            if done: break

    perf      = env_test.get_performance()
    vals      = np.array(env_test.portfolio_val)
    daily_ret = np.diff(vals) / (vals[:-1] + 1e-10)

    return {
        "sharpe":       round(perf["sharpe_ratio"],  4),
        "total_return": round(perf["total_return"],  4),
        "max_drawdown": round(perf["max_drawdown"],  4),
        "trade_count":  perf["trade_count"],
        "train_sharpe": round(best_sharpe, 4),
        "train_time_s": round(time.time() - t0, 1),
        "n_params":     sum(p.numel() for p in net.parameters()),
        "learning_curve": ep_sharpes,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse(csv_path: str):
    """Load saved CSV and print comprehensive analysis."""
    if not os.path.exists(csv_path):
        print(f"  No results file found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"\n{'='*70}")
    print(f"  GRID SEARCH ANALYSIS")
    print(f"  {len(df)} results  |  "
          f"{df['sym'].nunique()} assets  |  "
          f"{df['config'].nunique()} configs")
    print(f"{'='*70}")

    # 1. Best config overall (mean Sharpe across all assets)
    summary = (df.groupby("config")["sharpe"]
                 .agg(["mean","std","min","max"])
                 .round(4)
                 .sort_values("mean", ascending=False))
    print(f"\n── Top 10 Configs (mean Sharpe across all assets) ──")
    print(f"  {'Config':<28} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
    print(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for cfg, row in summary.head(10).iterrows():
        bar = "█" * int(row["mean"] * 10 + 5)
        print(f"  {cfg:<28} {row['mean']:>+7.3f} {row['std']:>7.3f} "
              f"{row['min']:>+7.3f} {row['max']:>+7.3f}  {bar}")

    # 2. Most robust config (highest minimum Sharpe)
    robust = (df.groupby("config")["sharpe"].min()
                .sort_values(ascending=False))
    print(f"\n── Most Robust Config (highest MINIMUM Sharpe — best worst-case) ──")
    for cfg, val in robust.head(5).items():
        print(f"  {cfg:<30} min_sharpe={val:+.3f}")

    # 3. Best config per asset class
    print(f"\n── Best Config per Asset Class ──")
    for cls in ["equity","crypto","forex","commodity"]:
        sub = df[df["asset_class"] == cls]
        if sub.empty: continue
        best = (sub.groupby("config")["sharpe"].mean()
                   .idxmax())
        mean_s = sub.groupby("config")["sharpe"].mean()[best]
        print(f"  {cls:<12}: {best:<28}  mean_sharpe={mean_s:+.3f}")

    # 4. Best config per individual asset
    print(f"\n── Best Config per Asset (top 3) ──")
    for sym in sorted(df["sym"].unique()):
        sub   = df[df["sym"] == sym].sort_values("sharpe", ascending=False)
        top3  = sub.head(3)
        bests = "  |  ".join(
            f"{r['config']}({r['sharpe']:+.3f})"
            for _, r in top3.iterrows()
        )
        print(f"  {sym:<12}: {bests}")

    # 5. Which hyperparameter dimension matters most?
    print(f"\n── Hyperparameter Sensitivity (std of mean Sharpe) ──")
    print(f"  Higher std = this dimension matters more\n")
    groups = {
        "activation": ["act_relu","act_tanh","act_leaky_relu","act_elu","act_silu","act_mish"],
        "normalisation": ["norm_batch","norm_rms","norm_group","norm_none"],
        "width": ["width_64","width_256"],
        "depth": ["depth_1","depth_3"],
        "optimizer": ["adamw","sgd","rmsprop","adagrad"],
        "loss_fn": ["loss_mae","loss_huber","loss_logcosh","loss_quantile"],
        "scheduler": ["sched_cosine","sched_step","sched_warmup","sched_plateau"],
        "reward_fn": ["reward_sortino","reward_calmar","reward_simple","reward_dd_pen"],
        "regularisation": ["dropout_0","dropout_0.3","l1_reg","l2_reg","elastic_net"],
        "replay": ["per"],
        "n_step": ["nstep_3","nstep_5"],
    }
    sensitivity = {}
    for dim, cfgs in groups.items():
        subset = df[df["config"].isin(cfgs + ["baseline"])]
        if subset.empty: continue
        means  = subset.groupby("config")["sharpe"].mean()
        sensitivity[dim] = float(means.std())

    for dim, std in sorted(sensitivity.items(), key=lambda x: -x[1]):
        bar = "█" * int(std * 30)
        print(f"  {dim:<18}: std={std:.4f}  {bar}")

    print(f"\n  Full results: {csv_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets",   nargs="+", default=None,
                        help="Specific symbols to run (default: all 48)")
    parser.add_argument("--episodes", type=int,  default=20,
                        help="Episodes per config (default: 20)")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip configs already in CSV")
    parser.add_argument("--analyse",  action="store_true",
                        help="Only run analysis on existing CSV, no training")
    args = parser.parse_args()

    if args.analyse:
        analyse(OUT_CSV)
        return

    universe = [(s, n, c, a) for s, n, c, a in UNIVERSE
                if args.assets is None or s in args.assets]

    configs = build_grid(args.episodes)

    # Load already-done combos for resume mode
    done = set()
    if args.resume and os.path.exists(OUT_CSV):
        with open(OUT_CSV) as f:
            for row in csv.DictReader(f):
                done.add((row["sym"], row["config"]))
        print(f"  Resume mode: {len(done)} results already in CSV — skipping")

    total_runs = len(universe) * len(configs) - len(done)
    print("=" * 70)
    print(f"  FULL GRID SEARCH — ALL ASSETS × ALL CONFIGS")
    print(f"  {len(universe)} assets  ×  {len(configs)} configs  =  "
          f"{len(universe)*len(configs)} runs  ({len(done)} skipped)")
    print(f"  Episodes per run: {args.episodes}  |  Device: {DEVICE}")
    print(f"  Est. time: {total_runs * args.episodes * 0.5 / 3600:.1f}–"
          f"{total_runs * args.episodes * 1.0 / 3600:.1f}h")
    print(f"  Results: {OUT_CSV}  (saved incrementally)")
    print("=" * 70)

    FIELDNAMES = [
        "sym","name","asset_class","config",
        "activation","hidden","n_layers","norm","residual",
        "dropout","weight_decay","l1_lambda","droppath","init",
        "opt","lr","scheduler","loss","gamma","reward_fn","per","n_steps",
        "sharpe","total_return","max_drawdown","trade_count",
        "train_sharpe","train_time_s","n_params",
    ]

    # Open CSV for incremental writing
    csv_exists = os.path.exists(OUT_CSV)
    csv_f      = open(OUT_CSV, "a", newline="")
    writer     = csv.DictWriter(csv_f, fieldnames=FIELDNAMES)
    if not csv_exists:
        writer.writeheader()
        csv_f.flush()

    run_idx = 0
    for sym, name, asset_class, _ in universe:
        print(f"\n{'─'*70}")
        print(f"  ASSET: {sym:<12}  {name:<20}  [{asset_class}]")
        print(f"{'─'*70}")

        # Fetch data
        try:
            df = yf.download(sym, start="2015-01-01",
                             end=datetime.now().strftime("%Y-%m-%d"),
                             progress=False)
            if hasattr(df.columns, "droplevel"):
                df.columns = df.columns.droplevel(1)
            df = df.dropna().reset_index(drop=True)
            if len(df) < 200:
                print(f"  ⚠️  Only {len(df)} rows — skipping")
                continue
        except Exception as e:
            print(f"  ❌ Data fetch failed: {e}")
            continue

        n_train  = int(len(df) * 0.70)
        df_train = df.iloc[:n_train].reset_index(drop=True)
        df_test  = df.iloc[n_train:].reset_index(drop=True)

        sig_train = build_signals(df_train)
        sig_test  = build_signals(df_test)

        for cfg in configs:
            if (sym, cfg.name) in done:
                continue

            run_idx += 1
            print(f"  [{run_idx:4d}] {cfg.name:<28}", end=" ", flush=True)

            try:
                r = train_eval(df_train, df_test, sig_train, sig_test, cfg)
                marker = ("✅" if r["sharpe"] > 0.5 else
                          "⚠️ " if r["sharpe"] > 0 else "❌")
                print(f"Sharpe={r['sharpe']:+.3f}  "
                      f"Ret={r['total_return']:+.1%}  "
                      f"DD={r['max_drawdown']:.1%}  "
                      f"t={r['train_time_s']:.0f}s  {marker}")

                row = {
                    "sym": sym, "name": name, "asset_class": asset_class,
                    "config": cfg.name,
                    "activation": cfg.activation, "hidden": cfg.hidden,
                    "n_layers": cfg.n_layers, "norm": cfg.norm,
                    "residual": cfg.residual, "dropout": cfg.dropout,
                    "weight_decay": cfg.weight_decay, "l1_lambda": cfg.l1_lambda,
                    "droppath": cfg.droppath, "init": cfg.init,
                    "opt": cfg.opt, "lr": cfg.lr, "scheduler": cfg.scheduler,
                    "loss": cfg.loss, "gamma": cfg.gamma,
                    "reward_fn": cfg.reward_fn, "per": cfg.per,
                    "n_steps": cfg.n_steps,
                    **{k: v for k, v in r.items()
                       if k not in ("learning_curve",)},
                }
                writer.writerow(row)
                csv_f.flush()

            except Exception as e:
                print(f"FAILED: {e}")

        # Clear GPU cache between assets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_f.close()
    print(f"\n{'='*70}")
    print(f"  GRID SEARCH COMPLETE — running analysis...")
    print(f"{'='*70}")
    analyse(OUT_CSV)


if __name__ == "__main__":
    main()
