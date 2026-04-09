#!/usr/bin/env python3
"""
Experiment Lab — Hyperparameter & Architecture Comparison
==========================================================
Purpose: Test every core ML/DL concept against real financial data
         so you can see, measure, and understand the tradeoffs.

What you can vary per experiment:
  - Activation function  : GELU vs ReLU vs Tanh vs LeakyReLU vs ELU
  - Network width        : hidden neurons (64 / 128 / 256)
  - Network depth        : number of layers (1 / 2 / 3)
  - Regularization       : dropout rate (0.0 → 0.3)
  - Normalization        : LayerNorm vs BatchNorm vs None
  - Optimizer            : Adam vs AdamW vs SGD+momentum
  - Learning rate        : 1e-3 / 3e-4 / 1e-4
  - Critic loss function : MSE vs MAE (L1) vs Huber
  - Discount factor γ    : 0.95 / 0.99 / 1.0
  - Weight decay (L2 reg): 0.0 / 1e-4 / 1e-3

Each config trains a SAC agent on 70% of the asset history,
evaluates on the 30% holdout, and logs:
  - Sharpe ratio, total return, max drawdown (trading performance)
  - Win rate (% profitable trades)
  - Training time (efficiency)

Results saved to data/output/experiment_results.csv for comparison.

Usage:
    python scripts/experiment_lab.py --sym NVDA
    python scripts/experiment_lab.py --sym BTC-USD --episodes 80
    python scripts/experiment_lab.py --sym AAPL --quick   # 20 episodes
"""

import os, sys, time, warnings, argparse, csv
from dataclasses import dataclass, field, asdict
from typing import List, Optional
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from src.rl.trading_env              import TradingEnvironment, build_signals
from src.evaluation.metrics          import sharpe_ratio, max_drawdown, win_rate, total_return

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGURATION
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """
    One experiment = one set of hyperparameters.

    Theory notes inline so you know WHY each parameter matters.
    """
    name: str

    # ── Architecture ───────────────────────────────────────────
    hidden_size: int   = 128
    # Width: more neurons = higher capacity, but more compute + risk of overfitting
    # The bias-variance tradeoff: wider = lower bias, potentially higher variance

    n_layers: int      = 2
    # Depth: deeper networks can represent more complex functions,
    # but vanishing/exploding gradients become harder to manage

    activation: str    = "gelu"
    # Options: "relu", "gelu", "tanh", "leaky_relu", "elu"
    # ReLU: simple, fast, can "die" (neuron always outputs 0)
    # GELU: smoother ReLU, used in Transformers (GPT, BERT)
    # Tanh: outputs in (-1,1), smooth, prone to vanishing gradients
    # LeakyReLU: fixes dying ReLU by allowing small negative slope
    # ELU: like LeakyReLU but smooth, handles negative inputs better

    # ── Regularization ─────────────────────────────────────────
    dropout: float     = 0.1
    # Randomly zeros neurons during training → prevents co-adaptation
    # = a form of ensemble learning (trains many sub-networks)
    # Too high (>0.5): underfitting. Too low (0.0): overfitting risk.

    norm_type: str     = "layer"
    # "layer" = LayerNorm (normalises across features, stable for small batches)
    # "batch" = BatchNorm (normalises across batch, better for CNN/large batches)
    # "none"  = no normalisation (faster, less stable training)

    weight_decay: float = 0.0
    # L2 regularisation: penalises large weights (||w||²) in the loss.
    # Equivalent to putting a Gaussian prior on weights → Bayesian view.
    # AdamW implements true L2 decay; Adam with weight_decay≠0 is L2-ish but not exact.

    # ── Optimisation ───────────────────────────────────────────
    optimizer: str     = "adam"
    # "adam"  : adaptive lr per param, fast convergence (default)
    # "adamw" : Adam + true L2 weight decay (better generalisation)
    # "sgd"   : momentum-based, slower but often better final performance
    # Key insight: Adam converges faster but AdamW generalises better

    lr: float          = 3e-4
    # Too high → loss oscillates, doesn't converge
    # Too low  → extremely slow convergence
    # 3e-4 is the "Karpathy constant" — empirically good for Adam in deep RL

    # ── RL-specific ────────────────────────────────────────────
    gamma: float       = 0.99
    # Discount factor: how much the agent cares about future rewards vs now.
    # γ=0.99 → cares about ~100 steps ahead (1/(1-0.99))
    # γ=0.95 → ~20 steps ahead (shorter horizon, faster but myopic)
    # γ=1.0  → infinite horizon, can cause instability

    loss_fn: str       = "mse"
    # Critic (Q-network) loss function:
    # "mse"   = MSE — penalises large errors quadratically, sensitive to outliers
    # "mae"   = MAE — linear penalty, robust to outlier reward spikes
    # "huber" = Huber — MSE for small errors, MAE for large (best of both)
    # Financial data has fat tails → Huber or MAE often outperforms MSE

    # ── Training ───────────────────────────────────────────────
    episodes: int      = 50


# ══════════════════════════════════════════════════════════════════
# DEFAULT EXPERIMENT SUITE
# ══════════════════════════════════════════════════════════════════

def default_experiments(episodes: int) -> List[ExperimentConfig]:
    """
    Canonical set of experiments — one change at a time from baseline.
    This is how proper ablation studies work: change one variable,
    hold everything else constant, measure the delta.
    """
    e = episodes
    return [
        # ── Baseline ──────────────────────────────────────────
        ExperimentConfig("baseline",              episodes=e),

        # ── Activation comparison ─────────────────────────────
        ExperimentConfig("act_relu",              activation="relu",       episodes=e),
        ExperimentConfig("act_tanh",              activation="tanh",       episodes=e),
        ExperimentConfig("act_leaky_relu",        activation="leaky_relu", episodes=e),
        ExperimentConfig("act_elu",               activation="elu",        episodes=e),

        # ── Width (neurons per layer) ─────────────────────────
        ExperimentConfig("width_64",              hidden_size=64,          episodes=e),
        ExperimentConfig("width_256",             hidden_size=256,         episodes=e),

        # ── Depth (number of layers) ──────────────────────────
        ExperimentConfig("depth_1",               n_layers=1,              episodes=e),
        ExperimentConfig("depth_3",               n_layers=3,              episodes=e),

        # ── Dropout (regularisation) ──────────────────────────
        ExperimentConfig("dropout_0",             dropout=0.0,             episodes=e),
        ExperimentConfig("dropout_0.2",           dropout=0.2,             episodes=e),
        ExperimentConfig("dropout_0.3",           dropout=0.3,             episodes=e),

        # ── Normalisation ─────────────────────────────────────
        ExperimentConfig("norm_batch",            norm_type="batch",       episodes=e),
        ExperimentConfig("norm_none",             norm_type="none",        episodes=e),

        # ── Optimiser ─────────────────────────────────────────
        ExperimentConfig("opt_adamw",             optimizer="adamw", weight_decay=1e-4, episodes=e),
        ExperimentConfig("opt_adamw_strong",      optimizer="adamw", weight_decay=1e-3, episodes=e),
        ExperimentConfig("opt_sgd",               optimizer="sgd",         episodes=e),

        # ── Learning rate ─────────────────────────────────────
        ExperimentConfig("lr_high",               lr=1e-3,                 episodes=e),
        ExperimentConfig("lr_low",                lr=1e-4,                 episodes=e),

        # ── Critic loss function ──────────────────────────────
        ExperimentConfig("loss_mae",              loss_fn="mae",           episodes=e),
        ExperimentConfig("loss_huber",            loss_fn="huber",         episodes=e),

        # ── Discount factor (time horizon) ────────────────────
        ExperimentConfig("gamma_0.95",            gamma=0.95,              episodes=e),
        ExperimentConfig("gamma_1.0",             gamma=1.0,               episodes=e),

        # ── Combined: best guesses ────────────────────────────
        ExperimentConfig("combo_best_guess",
                         activation="gelu", hidden_size=128, dropout=0.1,
                         optimizer="adamw", weight_decay=1e-4, lr=3e-4,
                         loss_fn="huber", gamma=0.99,          episodes=e),
    ]


# ══════════════════════════════════════════════════════════════════
# CONFIGURABLE NETWORK
# ══════════════════════════════════════════════════════════════════

def get_activation(name: str) -> nn.Module:
    """
    Factory for activation functions.
    Each has a different shape, derivative, and gradient flow behaviour.
    """
    acts = {
        "relu":       nn.ReLU(),
        "gelu":       nn.GELU(),
        "tanh":       nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.1),
        "elu":        nn.ELU(alpha=1.0),
    }
    if name not in acts:
        raise ValueError(f"Unknown activation '{name}'. Choose: {list(acts)}")
    return acts[name]


def get_norm(norm_type: str, size: int) -> Optional[nn.Module]:
    """
    Factory for normalisation layers.
    LayerNorm: normalises per sample across features — stable, good for RL.
    BatchNorm: normalises per feature across batch — can be unstable for small batches.
    """
    if norm_type == "layer": return nn.LayerNorm(size)
    if norm_type == "batch": return nn.BatchNorm1d(size)
    return None   # "none" — no normalisation


def build_mlp(in_dim: int, out_dim: int, cfg: ExperimentConfig) -> nn.Sequential:
    """
    Build a configurable MLP from an ExperimentConfig.
    This is the building block for both actor and critics.
    """
    layers = []
    current = in_dim

    for i in range(cfg.n_layers):
        layers.append(nn.Linear(current, cfg.hidden_size))
        norm = get_norm(cfg.norm_type, cfg.hidden_size)
        if norm is not None:
            layers.append(norm)
        layers.append(get_activation(cfg.activation))
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        current = cfg.hidden_size

    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class ConfigurableNet(nn.Module):
    """
    SAC network with fully configurable architecture.
    Replaces the hardcoded SACNetwork for experiments.
    """
    def __init__(self, state_dim: int, action_dim: int, cfg: ExperimentConfig):
        super().__init__()
        # Actor and critics share the same config but have separate weights
        self.actor_net = build_mlp(state_dim, action_dim, cfg)
        self.q1_net    = build_mlp(state_dim, action_dim, cfg)
        self.q2_net    = build_mlp(state_dim, action_dim, cfg)

    def get_actor(self, x): return self.actor_net(x)
    def get_q(self, x):     return self.q1_net(x), self.q2_net(x)


def get_optimizer(name: str, params, lr: float, weight_decay: float):
    """
    Factory for optimisers.
    Adam:  adaptive per-parameter LR — fast convergence
    AdamW: Adam + true L2 weight decay — better generalisation
    SGD:   gradient descent + momentum — classic, often strong final perf
    """
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9,
                                weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer '{name}'")


def compute_critic_loss(pred: torch.Tensor, target: torch.Tensor,
                        loss_fn: str) -> torch.Tensor:
    """
    Critic (Q-network) loss.
    MSE:   (pred - target)²  — sensitive to outliers, common default
    MAE:   |pred - target|   — robust to reward spikes, less smooth gradient
    Huber: MSE when |err|<δ, MAE when |err|≥δ  — best of both
    """
    if loss_fn == "mse":   return F.mse_loss(pred, target)
    if loss_fn == "mae":   return F.l1_loss(pred, target)
    if loss_fn == "huber": return F.huber_loss(pred, target, delta=1.0)
    raise ValueError(f"Unknown loss_fn '{loss_fn}'")


# ══════════════════════════════════════════════════════════════════
# TRAINING ONE CONFIG
# ══════════════════════════════════════════════════════════════════

def train_and_evaluate(df_train: pd.DataFrame,
                       df_test:  pd.DataFrame,
                       sig_train: np.ndarray,
                       sig_test:  np.ndarray,
                       cfg: ExperimentConfig) -> dict:
    """
    Train a SAC agent with the given config on df_train,
    evaluate on df_test holdout.
    Returns a dict of metrics.
    """
    t0 = time.time()

    env_train = TradingEnvironment(df_train, sig_train)
    state_dim = env_train.state_dim

    # Build configurable network + target
    net        = ConfigurableNet(state_dim, 3, cfg).to(DEVICE)
    target_net = ConfigurableNet(state_dim, 3, cfg).to(DEVICE)
    target_net.load_state_dict(net.state_dict())

    # Separate optimisers for actor and critics
    actor_params  = list(net.actor_net.parameters())
    critic_params = list(net.q1_net.parameters()) + list(net.q2_net.parameters())
    actor_optim   = get_optimizer(cfg.optimizer, actor_params,
                                  cfg.lr, cfg.weight_decay)
    critic_optim  = get_optimizer(cfg.optimizer, critic_params,
                                  cfg.lr, cfg.weight_decay)

    # SAC entropy tuning
    log_alpha      = torch.tensor(np.log(0.2), dtype=torch.float32,
                                   requires_grad=True, device=DEVICE)
    target_entropy = -np.log(1.0 / 3) * 0.98
    alpha_optim    = torch.optim.Adam([log_alpha], lr=1e-4)

    from collections import deque
    import random
    buffer = deque(maxlen=50000)
    BATCH  = 64
    TAU    = 0.005

    best_sharpe = -999
    best_state  = None

    # ── Training loop ──────────────────────────────────────────
    for ep in range(cfg.episodes):
        state = env_train.reset()
        while True:
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = net.get_actor(s_t)
                action = torch.distributions.Categorical(logits=logits).sample().item()
            ns, r, done, _ = env_train.step(action)
            buffer.append((state, action, r, ns, done))
            state = ns
            if done: break

            if len(buffer) < BATCH:
                continue

            # Sample batch
            batch = random.sample(buffer, BATCH)
            s, a, rw, ns_, d = zip(*batch)
            s   = torch.tensor(np.array(s),   dtype=torch.float32).to(DEVICE)
            a   = torch.tensor(np.array(a),   dtype=torch.int64).to(DEVICE)
            rw  = torch.tensor(np.array(rw),  dtype=torch.float32).to(DEVICE)
            ns_ = torch.tensor(np.array(ns_), dtype=torch.float32).to(DEVICE)
            d   = torch.tensor(np.array(d),   dtype=torch.float32).to(DEVICE)
            alpha = log_alpha.exp().detach()

            # Critic update
            with torch.no_grad():
                tlogits    = target_net.get_actor(ns_)
                tprobs     = torch.softmax(tlogits, -1)
                tlp        = torch.log_softmax(tlogits, -1)
                tq1, tq2   = target_net.get_q(ns_)
                min_tq     = torch.min(tq1, tq2)
                v_next     = (tprobs * (min_tq - alpha * tlp)).sum(-1)
                target_q   = rw + cfg.gamma * (1 - d) * v_next

            q1, q2 = net.get_q(s)
            q1_a   = q1.gather(1, a.unsqueeze(1)).squeeze()
            q2_a   = q2.gather(1, a.unsqueeze(1)).squeeze()
            c_loss = compute_critic_loss(q1_a, target_q, cfg.loss_fn) + \
                     compute_critic_loss(q2_a, target_q, cfg.loss_fn)

            critic_optim.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(critic_params, 1.0)
            critic_optim.step()

            # Actor update
            logits   = net.get_actor(s)
            probs    = torch.softmax(logits, -1)
            lp       = torch.log_softmax(logits, -1)
            q1c, q2c = net.get_q(s)
            min_qc   = torch.min(q1c, q2c)
            a_loss   = (probs * (alpha * lp - min_qc)).sum(-1).mean()

            actor_optim.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(actor_params, 1.0)
            actor_optim.step()

            # Alpha (entropy coefficient) update
            ent      = -(probs.detach() * lp.detach()).sum(-1).mean()
            al_loss  = -(log_alpha * (ent - target_entropy).detach())
            alpha_optim.zero_grad()
            al_loss.backward()
            alpha_optim.step()

            # Soft target update
            for p, tp in zip(net.parameters(), target_net.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        perf = env_train.get_performance()
        if perf["sharpe_ratio"] > best_sharpe:
            best_sharpe = perf["sharpe_ratio"]
            best_state  = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    if best_state:
        net.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    train_time = time.time() - t0

    # ── Holdout evaluation ─────────────────────────────────────
    env_test = TradingEnvironment(df_test, sig_test)
    state    = env_test.reset()
    net.eval()
    with torch.no_grad():
        while True:
            s_t    = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            logits = net.get_actor(s_t)
            action = logits.argmax(-1).item()
            state, _, done, _ = env_test.step(action)
            if done: break

    perf_test = env_test.get_performance()
    vals      = np.array(env_test.portfolio_val)
    daily_ret = np.diff(vals) / (vals[:-1] + 1e-10)

    n_params  = sum(p.numel() for p in net.parameters())

    return {
        "config":          cfg.name,
        "activation":      cfg.activation,
        "hidden_size":     cfg.hidden_size,
        "n_layers":        cfg.n_layers,
        "dropout":         cfg.dropout,
        "norm_type":       cfg.norm_type,
        "optimizer":       cfg.optimizer,
        "lr":              cfg.lr,
        "weight_decay":    cfg.weight_decay,
        "loss_fn":         cfg.loss_fn,
        "gamma":           cfg.gamma,
        "episodes":        cfg.episodes,
        "sharpe":          round(perf_test["sharpe_ratio"],  4),
        "total_return":    round(perf_test["total_return"],  4),
        "max_drawdown":    round(perf_test["max_drawdown"],  4),
        "trade_count":     perf_test["trade_count"],
        "train_sharpe":    round(best_sharpe, 4),
        "train_time_s":    round(train_time, 1),
        "n_params":        n_params,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter experiments on a single asset")
    parser.add_argument("--sym",      default="NVDA",
                        help="Asset symbol (default: NVDA)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="RL training episodes per config (default: 50)")
    parser.add_argument("--quick",    action="store_true",
                        help="Quick mode: 20 episodes, 5 key configs only")
    parser.add_argument("--configs",  nargs="+", default=None,
                        help="Run specific config names only")
    args = parser.parse_args()

    sym      = args.sym
    episodes = 20 if args.quick else args.episodes

    print("=" * 65)
    print(f"  EXPERIMENT LAB — {sym}")
    print(f"  Device: {DEVICE}  |  Episodes per config: {episodes}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Fetch data ─────────────────────────────────────────────
    print(f"\n  Fetching {sym} data...")
    df = yf.download(sym, start="2015-01-01",
                     end=datetime.now().strftime("%Y-%m-%d"),
                     progress=False)
    if hasattr(df.columns, "droplevel"):
        df.columns = df.columns.droplevel(1)
    df = df.dropna().reset_index(drop=True)
    print(f"  {len(df)} rows")

    n_train  = int(len(df) * 0.70)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_test  = df.iloc[n_train:].reset_index(drop=True)

    # Build technical signals (no ML models needed — uses build_signals fallback)
    print("  Building signals...")
    sig_train = build_signals(df_train)
    sig_test  = build_signals(df_test)
    print(f"  Signal shape: {sig_train.shape}  state_dim={sig_train.shape[1]+3}")

    # ── Select experiments ─────────────────────────────────────
    all_cfgs = default_experiments(episodes)
    if args.quick:
        # Quick mode: 5 representative configs
        quick_names = {"baseline", "act_relu", "loss_huber",
                       "opt_adamw", "depth_1"}
        all_cfgs = [c for c in all_cfgs if c.name in quick_names]
    elif args.configs:
        all_cfgs = [c for c in all_cfgs if c.name in set(args.configs)]

    print(f"\n  Running {len(all_cfgs)} experiment configs...\n")

    # ── Run experiments ────────────────────────────────────────
    results = []
    for i, cfg in enumerate(all_cfgs, 1):
        print(f"  [{i:2d}/{len(all_cfgs)}] {cfg.name:<28} ", end="", flush=True)
        try:
            r = train_and_evaluate(df_train, df_test,
                                   sig_train, sig_test, cfg)
            results.append(r)
            sharpe_marker = ("✅" if r["sharpe"] > 0.5 else
                             "⚠️ " if r["sharpe"] > 0 else "❌")
            print(f"Sharpe={r['sharpe']:+.3f}  "
                  f"Return={r['total_return']:+.1%}  "
                  f"DD={r['max_drawdown']:.1%}  "
                  f"t={r['train_time_s']:.0f}s  {sharpe_marker}")
        except Exception as e:
            print(f"FAILED: {e}")

    if not results:
        print("No results — check errors above.")
        return

    # ── Sort and display results ────────────────────────────────
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n{'='*65}")
    print(f"  RESULTS — ranked by Sharpe ratio (holdout set)")
    print(f"  Asset: {sym}  |  Train: {len(df_train)} rows  |  Test: {len(df_test)} rows")
    print(f"{'='*65}")
    print(f"  {'Config':<28} {'Sharpe':>7} {'Return':>8} {'DD':>7} "
          f"{'Trades':>7} {'Params':>8}")
    print(f"  {'-'*28} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*8}")

    baseline_sharpe = next((r["sharpe"] for r in results
                            if r["config"] == "baseline"), 0.0)
    for r in results:
        delta  = r["sharpe"] - baseline_sharpe
        marker = f" (+{delta:.3f})" if delta > 0.01 else (
                 f" ({delta:.3f})" if delta < -0.01 else "")
        print(f"  {r['config']:<28} {r['sharpe']:>+7.3f}{marker:<10} "
              f"{r['total_return']:>+7.1%} {r['max_drawdown']:>7.1%} "
              f"{r['trade_count']:>7d} {r['n_params']:>8,}")

    # ── Key insights ───────────────────────────────────────────
    best = results[0]
    worst = results[-1]
    print(f"\n  Best:  {best['config']} → Sharpe {best['sharpe']:+.3f}")
    print(f"  Worst: {worst['config']} → Sharpe {worst['sharpe']:+.3f}")
    print(f"  Gap:   {best['sharpe'] - worst['sharpe']:.3f} "
          f"(larger = hyperparams matter more for this asset)")

    # ── Save to CSV ────────────────────────────────────────────
    os.makedirs("data/output", exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"data/output/experiment_results_{sym}_{ts}.csv"
    latest   = f"data/output/experiment_results_latest.csv"

    fieldnames = list(results[0].keys())
    for path in [out_path, latest]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)

    print(f"\n  💾 Results saved → {out_path}")
    print(f"  💾 Latest copy  → {latest}")
    print(f"\n  Tip: compare columns in the CSV to understand each")
    print(f"  hyperparameter's effect on this specific asset.\n")


if __name__ == "__main__":
    main()
