# MRAT-RL: Multimodal Regime-Adaptive Transformer and Reinforcement Learning

**MSc Artificial Intelligence — NCI Dublin — Programming for AI (MSCAIJAN26I)**

Group: Taiwo Alabi · Divyesh Reddy Ellasiri · Sai Vivek Yerninti · Mukesh Saren Ramu

---

## What This System Does

An AI trading signal generator that scans 39 global financial assets every morning and recommends the top trades for the day. It combines four independent AI models into a consensus voting system — a trade only happens when at least 3 out of 4 models agree.

**The four voters:**

| Model | What it does |
|-------|-------------|
| PatchTST Transformer | Predicts price direction over the next 5 days using 22 years of price history |
| Hidden Markov Model (HMM) | Detects whether the market is bull, bear, or sideways — trained separately per asset class |
| SAC / PPO RL Agent | Per-asset agent that learned trading strategy from thousands of simulated trades |
| RSI Divergence | Independent mean-reversion signal — spots exhausted trends |

---

## Key Results (7,800 Anti-Cheat Walk-Forward Tests)

200 random 52-week windows per asset × 39 assets. Models never see future data.

| Metric | Result |
|--------|--------|
| Average annual return | +1.85% per window |
| Sharpe ratio | 0.909 |
| Profitable windows | 59% |
| 2022 bear market loss | −1.2% vs S&P 500 −20%, BTC −65% |
| Bull market periods | 9/10 profitable, avg Sharpe 1.189 |

**Top performing assets:**

| Asset | Sharpe | Win Rate |
|-------|--------|---------|
| XRP-USD | 3.244 | 80% |
| Amazon (AMZN) | 1.888 | 92% |
| Gold (GC=F) | 1.727 | 77% |
| GBP/USD | 1.582 | 61% |
| Meta (META) | 1.518 | 84% |
| NVDA | 1.268 | 84% |

**2022 bear market — system vs buy-and-hold:**

| Asset | System | Buy & Hold | Edge |
|-------|--------|-----------|------|
| S&P 500 | −1.2% | −20.0% | +18.8% |
| NASDAQ | −1.1% | −33.9% | +32.8% |
| Bitcoin | −1.4% | −65.2% | +63.8% |
| Tech stocks avg | −1.2% | −43.2% | +42.0% |

**Year by year ($100,000 starting capital each year):**

| Year | Return | Sharpe |
|------|--------|--------|
| 2020 | +3.0% | +1.669 |
| 2021 | +0.3% | +0.632 |
| 2022 | −2.0% | −0.553 |
| 2023 | +1.5% | +1.592 |
| 2024 | +0.5% | +0.764 |
| 2025 | +1.9% | +1.343 |

Compounded: $100,000 in 2020 → **$104,438 by 2026** (+4.4%)

---

## Quick Start
```bash
# Install dependencies
pip install yfinance scikit-learn lightgbm hmmlearn transformers torch --break-system-packages

# Train all 39 models — first time only (3-4 hours on GPU)
python3 scripts/train_all_models.py

# Run daily advisor — scan all 39 assets, get today's top trades
python3 scripts/daily_advisor_v2.py
```

The daily advisor will ask:
1. Portfolio amount — type `₦100000` or `€500` or `$300`
2. Risk profile — 1 = Conservative, 2 = Moderate, 3 = Aggressive
3. Press Enter to scan

---

## Sample Output
```
┌─ #1 SOL-USD  Solana
│  ▲ BUY  · SIDEWAYS · ◆ HIGH · conf 75% · PPO
│  Entry      : €86.44
│  TP (daily) : €93.14  (+7.74%)
│  SL (daily) : €81.98  (-5.16%)
│  TP (swing) : €91.80  (+6.19%)
│  SL (swing) : €79.75  (-7.74%)
│  Position   : 20.0% → €200  (₦340,000)
│  5d forecast : +1.68%
│  Votes      : BUY=3 SELL=0 HOLD=1
└─ PatchTST:BUY  HMM:HOLD  RL:BUY  RSI:BUY

Market Summary:
  Signals: BUY=8  SELL=0  HOLD=31
  Regimes: BULL=34  BEAR=0  SIDEWAYS=5
  39 assets scanned — 0 skipped
```

---

## Scripts Reference

| Script | What it does |
|--------|-------------|
| `daily_advisor_v2.py` | **MAIN** — scan all 39 assets, top daily + swing trades in EUR and NGN |
| `train_all_models.py` | Train all 39 PatchTST + SAC/PPO models from scratch |
| `robust_validation.py` | 7,800 anti-cheat walk-forward validation tests |
| `backtest_v2.py` | Backtest March 2025 → March 2026 |
| `yearly_backtest.py` | Year-by-year results 2020–2026 |
| `bull_regime_backtest.py` | Test across 10 historical bull market periods |
| `enhanced_pipeline.py` | Live signal pipeline for a subset of assets |
| `signal_validator.py` | Post-generation validation (TP/SL, correlation, duplicates) |
| `daily_advisor.py` | Original v1 advisor — legacy, 3 forex pairs only |
| `backtest.py` | Original backtester — legacy |
| `live_paper_trade.py` | Real-time monitor, auto-closes on TP/SL |
| `paper_trade.py` | Quick one-shot signal report |
| `trade_log.py` | View trade history and accuracy stats |

---

## Project Structure
```
Project-for-PAI/
├── scripts/                          ← All runnable scripts (see table above)
├── src/
│   ├── forecast/
│   │   ├── patchtst_forecast.py      ← PatchTST Transformer (save/load enabled)
│   │   └── lgbm_forecast.py          ← LightGBM forecaster (legacy v1)
│   ├── regime/
│   │   ├── hmm_regime.py             ← Per-class HMM (4 models: equity/crypto/commodity/forex)
│   │   └── pretrained_hmm.py         ← Legacy fallback only — not used for live signals
│   ├── rl/
│   │   ├── trading_env.py            ← Trading simulation environment
│   │   ├── sac_agent.py              ← SAC agent (equities, commodities, forex)
│   │   ├── ppo_agent.py              ← PPO agent (cryptocurrencies)
│   │   ├── macro_sentiment_features.py  ← VIX + interest rates + FinBERT sentiment
│   │   ├── integrated_pipeline.py    ← PatchTST signal builder
│   │   ├── ablation_study.py         ← Component importance analysis
│   │   ├── compare_agents.py         ← PPO vs SAC vs Rule-based comparison
│   │   └── regime_stress_test.py     ← Bull/Bear/Random regime stress test
│   ├── trading/
│   │   ├── ensemble_trader.py        ← Ensemble signal combiner (legacy)
│   │   └── rule_trader.py            ← Rule-based baseline
│   └── utils/
│       ├── currency.py               ← Live EUR/NGN/USD exchange rates
│       ├── sentiment_loader.py       ← FinBERT sentiment reader
│       └── trade_logger.py           ← Persistent trade log
├── data/
│   ├── models/                       ← 78+ saved models
│   │   ├── patchtst_*.pt             ← PatchTST per asset (39 files)
│   │   ├── sac_*.pt                  ← SAC agents (equities/commodities/forex)
│   │   ├── ppo_*.pt                  ← PPO agents (crypto)
│   │   └── hmm_*.pkl                 ← 4 HMM models (equity/crypto/commodity/forex)
│   ├── output/                       ← Daily signals, backtest results, validation
│   └── cache/                        ← Per-asset macro feature cache
├── configs/                          ← Backtest experiment configs
├── MRAT_RL_Beginners_Guide.docx      ← Plain English guide — start here if new to the system
├── MRAT_RL_Complete_Report.docx      ← Full IEEE academic report with all results
├── Multi-Modal Regime AI Trading Bot.docx  ← Project overview document
├── MRAT_RL_IEEE_Short_Paper.docx     ← Original short paper
├── FRONTEND_INTEGRATION.md           ← Web UI integration guide (for Divyesh)
└── README.md                         ← This file
```

---

## Asset Universe (39 Assets)

| Category | Assets |
|----------|--------|
| US Indices | ^GSPC (S&P 500), ^IXIC (NASDAQ), ^DJI (Dow Jones) |
| US Stocks | AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, JPM |
| Crypto | BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD |
| Forex Major | EURUSD=X, GBPUSD=X, USDJPY=X, USDCHF=X, AUDUSD=X, NZDUSD=X, USDCAD=X |
| Forex Cross | EURGBP=X, EURJPY=X, GBPJPY=X, AUDNZD=X |
| African Forex | USDNGN=X, EURNGN=X, USDZAR=X, USDKES=X, USDGHS=X |
| Commodities | GC=F (Gold), SI=F (Silver), CL=F (Crude Oil), NG=F (Natural Gas), HG=F (Copper), ZW=F (Wheat), ZC=F (Corn) |

---

## Agent Routing

| Asset Class | Agent | Why |
|------------|-------|-----|
| Equities, Commodities, Forex | SAC | More predictable — exploration finds better strategies |
| Cryptocurrencies | PPO | High volatility requires cautious policy updates |

---

## Risk Profiles

| Profile | Max per trade | Min confidence | Best for |
|---------|--------------|----------------|---------|
| 1 — Conservative | 5% | 100% unanimous | Small portfolio, capital preservation |
| 2 — Moderate | 10% | 75% (3/4 agree) | Balanced risk/reward |
| 3 — Aggressive | 20% | 50% (2/4 agree) | Higher risk tolerance |

---

## Ablation Study (Corrected)

| Removed | Avg Sharpe Impact | Verdict |
|---------|------------------|---------|
| No PatchTST | −0.783 | **Most critical** |
| No Sentiment | −0.485 | Second |
| No Macro features | −0.245 | Third |
| No HMM regime | −0.130 | Least critical |

> Earlier version incorrectly found macro features most critical — caused by cross-asset contamination where Bitcoin was receiving S&P 500 macro data. Fixed in v2.

---

## System Corrections (v1 → v2)

| Issue | Before | After |
|-------|--------|-------|
| HMM models | Single EUR/USD model for all 39 assets | 4 separate models per asset class |
| Macro benchmark | S&P 500 SMA for all assets | Per-asset benchmarks |
| Sentiment | 0.05 stub for all assets | Real FinBERT per asset |
| PatchTST | Retrained every run | Saved/loaded — reproducible |
| Ablation result | Macro most critical (wrong) | PatchTST most critical (correct) |

---


---

## Training Guide — Full Context for Anyone Picking Up This Project

### Hardware Used (Original Training)
- **Machine**: Taiwo's laptop — NVIDIA GeForce RTX 3050 (6GB VRAM), 20-core CPU, 16GB RAM
- **OS**: Ubuntu 24, Python 3.11, CUDA 12.8
- **Training time**: ~8 hours total for all 39 assets (first full run)
- **Per asset**: ~8-12 minutes (PatchTST ~5 min + RL agent ~3-7 min)

---

### What Was Trained and Why

The system has **three types of models** — all must be trained before the daily advisor works:

| Model Type | Files saved | Trained on | Time |
|-----------|-------------|-----------|------|
| PatchTST (39 files) | `data/models/patchtst_*.pt` | Each asset individually, 2004–2024 | ~5 min/asset |
| SAC/PPO RL agents (39 files) | `data/models/sac_*.pt` / `ppo_*.pt` | Each asset individually, 150-300 episodes | ~3-7 min/asset |
| HMM regime (4 files) | `data/models/hmm_*.pkl` | Per asset class (equity/crypto/commodity/forex) | ~1 min total |

**Total: 82 model files** in `data/models/`

---

### Issues We Encountered During Training

| Issue | What happened | How we fixed it |
|-------|--------------|-----------------|
| ETH/BNB PPO stuck in HOLD | PPO converged to always doing nothing — 0% win rate | Retrained ETH with SAC instead, BNB with 300 episodes + higher entropy |
| Cross-asset HMM contamination | Original single EUR/USD HMM applied to Bitcoin — wrong regime signals | Trained 4 separate HMMs per asset class |
| S&P 500 macro for all assets | Bitcoin was using S&P 500 SMA as its benchmark — contamination | Added per-asset macro benchmarks |
| PatchTST retraining every run | Models retrained from scratch each time — results varied | Added save/load — models train once, load instantly |
| GPU temperature | RTX 3050 hit 64°C during full training run | Normal range — throttles at 87°C, was safe |
| Training interrupted | Power cut mid-training | `train_all_models.py` skips already-saved models — safe to resume |

---

### When to Retrain

| Situation | What to retrain | Time needed |
|-----------|----------------|-------------|
| Every 6 months | All models | 8 hours overnight |
| Major market event (crash, new bull run) | All models | 8 hours overnight |
| Adding a new asset | That asset only | 15 minutes |
| HMM giving wrong regimes | HMM models only | 5 minutes |
| One asset performing badly | That asset's RL agent only | 10 minutes |

---

### How to Retrain — Simple Commands

**Retrain everything from scratch (run overnight):**
```bash
cd ~/Desktop/NCI/programming_for_ai/Project-for-PAI
nohup python3 scripts/train_all_models.py > data/output/training.log 2>&1 &
echo "Training started — monitor with: tail -f data/output/training.log"
```

**Resume interrupted training (skips already-saved models):**
```bash
python3 scripts/train_all_models.py
# Automatically skips assets that already have saved .pt files
```

**Retrain one specific asset (e.g. Bitcoin):**
```bash
python3 -c "
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf, torch, numpy as np
from src.forecast.patchtst_forecast import PatchTSTForecaster
from src.rl.ppo_agent import PPOAgent
from src.rl.trading_env import TradingEnvironment
from src.rl.integrated_pipeline import build_patchtst_signals
from src.rl.macro_sentiment_features import build_macro_signals

sym = 'BTC-USD'
key = 'BTC_USD'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Fetch data
df = yf.download(sym, start='2004-01-01', progress=False)
if hasattr(df.columns,'droplevel'): df.columns=df.columns.droplevel(1)
df = df.dropna()
df_train = df.iloc[:int(len(df)*0.70)].reset_index(drop=True)

# Train PatchTST
ptst = PatchTSTForecaster()
ptst.fit_from_df(df_train, verbose=True)
ptst.save(f'data/models/patchtst_{key}.pt')

# Train PPO
sig = build_patchtst_signals(df_train, ptst)
mac = build_macro_signals(df_train, asset_sym=sym)
import numpy; sig = numpy.concatenate([sig, mac], axis=1)
env = TradingEnvironment(df_train, sig)
agent = PPOAgent(state_dim=env.state_dim, device=DEVICE, n_steps=256)
for ep in range(200):
    state = env.reset()
    while True:
        a,lp,v = agent.select_action(state)
        ns,r,done,_ = env.step(a)
        agent.store(state,a,lp,r,v,done)
        state=ns
        if done: break
torch.save(agent.policy.state_dict(), f'data/models/ppo_{key}.pt')
print('Done — Bitcoin retrained')
"
```

**Retrain HMM models only (5 minutes):**
```bash
python3 -c "
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf, joblib
from src.regime.hmm_regime import HMMRegimeDetector

pairs = [
    ('hmm_equity',    ['^GSPC', '^IXIC']),
    ('hmm_crypto',    ['BTC-USD', 'ETH-USD']),
    ('hmm_commodity', ['GC=F', 'CL=F']),
    ('hmm_forex',     ['EURUSD=X', 'GBPUSD=X']),
]
for name, syms in pairs:
    frames = []
    for sym in syms:
        df = yf.download(sym, start='2004-01-01', progress=False)
        if hasattr(df.columns,'droplevel'): df.columns=df.columns.droplevel(1)
        frames.append(df.dropna())
    import pandas as pd
    combined = pd.concat(frames).sort_index()
    hmm = HMMRegimeDetector()
    hmm.fit(combined)
    joblib.dump(hmm, f'data/models/{name}.pkl')
    print(f'Saved {name}.pkl')
print('All HMMs retrained')
"
```

---

### Checking Training Progress
```bash
# How many models are saved?
ls data/models/patchtst_*.pt | wc -l   # Should be 39
ls data/models/sac_*.pt | wc -l        # Should be ~30
ls data/models/ppo_*.pt | wc -l        # Should be ~7
ls data/models/hmm_*.pkl | wc -l       # Should be 5 (4 + 1 fallback)

# Watch training live
tail -f data/output/training.log

# Monitor GPU during training
watch -n 5 "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
```

---

### Recommended Retraining Schedule

| When | Action |
|------|--------|
| **Every 6 months** | Full retrain — `python3 scripts/train_all_models.py` |
| **After a major crash** (>15% drop) | Full retrain to learn new bear patterns |
| **After a major rally** (>20% gain) | Full retrain to learn new bull patterns |
| **If an asset suddenly performs badly** | Retrain that asset only |
| **Never** | Delete all models and retrain from nothing — always use the resume feature |

---

### After Retraining — Verify It Worked
```bash
# Run validation (takes 8-12 hours — run overnight after retraining)
python3 scripts/robust_validation.py

# Quick sanity check (30 minutes)
python3 scripts/backtest_v2.py

# Run daily advisor to confirm signals generate correctly
python3 scripts/daily_advisor_v2.py
```

---
## Dependencies
```bash
pip install yfinance scikit-learn lightgbm hmmlearn transformers torch --break-system-packages
```

---

> This project is for educational purposes only. Not financial advice.
