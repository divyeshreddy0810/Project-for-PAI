# MRAT-RL: Multimodal Regime-Adaptive Transformer and Reinforcement Learning

**MSc Artificial Intelligence — NCI Dublin — Programming for AI (MSCAIJAN26I)**

Group: Taiwo Alabi · Divyesh Reddy Ellasiri · Sai Vivek Yerninti · Mukesh Saren Ramu

---

## What This System Does

An AI trading signal generator that scans 48 global financial assets every morning and recommends the top trades for the day. It combines multiple AI models into a consensus voting system; a trade only goes through when at least 2 of the 3 RL agents agree on a direction.

**The signal pipeline:**

| Model | What it does |
|-------|-------------|
| PatchTST Transformer | Predicts price direction over the next 5 days using price history |
| LightGBM | A second forecaster using technical indicators; blended 50/50 with PatchTST |
| Hidden Markov Model (HMM) | Detects whether the market is bull, bear, or sideways; trained per asset class |
| SAC Agent | Soft Actor-Critic RL agent; learned trading from simulated trades |
| TD3 Agent | Twin Delayed DDPG RL agent; cautious and stable, uses two internal scorers |
| PPO Agent | Proximal Policy Optimisation agent; updates in small steps to avoid overreacting |
| ADX | Trend strength indicator; confirms whether a trend is strong enough to trade |

The three RL agents vote; if 2 or more agree, the system acts. Otherwise it holds.

---

## Key Results (Latest Run — 9 April 2026)

| Metric | Result |
|--------|--------|
| Assets scanned | 47 of 48 (CRWV skipped; too new) |
| BUY signals | 17 |
| SELL signals | 2 |
| HOLD signals | 28 |
| Average confidence | 59% |
| Forecast range (5-day) | +1.5% to -0.4% |
| Regimes detected | BULL=33, SIDEWAYS=14, BEAR=0 |

**Previous backtest results (7,800 walk-forward tests):**

| Metric | Result |
|--------|--------|
| Average annual return | +1.85% per window |
| Sharpe ratio | 0.909 |
| Profitable windows | 59% |
| 2022 bear market loss | -1.2% vs S&P 500 -20%, BTC -65% |

---

## Deep Learning Laboratory

To find the best forecasting model for each asset, the system ran a competition across 1,200 training experiments.

**What was tested:**

| Architecture | Description |
|-------------|-------------|
| MLP | Simple stacked layers; no sense of time order |
| CNN | Scans the price chart with a sliding window; good at short patterns |
| LSTM | Reads day by day and carries a memory forward |
| PatchTST | Reads chunks of history all at once using attention |

Each architecture was tested with different optimisers (Adam, SGD), activation functions (ReLU, Tanh), and dropout rates (0.0, 0.3) across all 48 assets.

**Results:**

| Asset Class | Total | Promoted (CNN/LSTM won) | Kept PatchTST |
|-------------|-------|------------------------|---------------|
| Equity | 20 | 17 | 3 |
| Crypto | 5 | 5 | 0 |
| Commodity | 7 | 5 | 2 |
| Forex | 16 | 12 | 4 |
| **Total** | **48** | **39** | **9** |

A model was promoted only if it beat PatchTST by more than 5% on held-out test data. CNN and LSTM won most assets; PatchTST held its ground on a few equities and commodities.

---

## Quick Start
```bash
# Install dependencies
pip install yfinance scikit-learn lightgbm hmmlearn torch joblib --break-system-packages

# Train all 48 models — first time only (takes several hours on GPU)
python3 scripts/train_all_models.py

# Run the Deep Learning Laboratory — finds best model per asset
python3 scripts/run_dl_lab.py

# Promote lab winners to production
python3 scripts/promote_best.py

# Run daily advisor — scan all 48 assets, get today's top trades
python3 scripts/daily_advisor_v2.py
```

The daily advisor will ask:
1. Portfolio amount — type `€500` or `$300` or `₦100000`
2. Risk profile — 1 = Conservative, 2 = Moderate, 3 = Aggressive
3. Press Enter to scan

---

## Sample Output
```
┌─ #1 USDGHS=X  USD/GHS
│  ▲ BUY  · BULL · ★ UNANIMOUS · conf 100%
│  Entry      : €11.0000
│  TP (daily) : €11.0211  (+0.19%)
│  SL (daily) : €10.9578  (-0.38%)
│  Position   : 10.0% → €100  (₦170,000)
│  5d forecast : +1.00%  → €11.1101
│  Votes      : BUY=4 SELL=0 HOLD=0
└─ PatchTST:BUY  HMM:BUY  RL:BUY  ADX:BUY

Market Summary:
  Signals: BUY=17  SELL=2  HOLD=28
  Regimes: BULL=33  BEAR=0  SIDEWAYS=14
  47 assets scanned — 1 skipped (CRWV)
```

---

## Scripts Reference

| Script | What it does |
|--------|-------------|
| `daily_advisor_v2.py` | **MAIN** — scan all 48 assets, top daily and swing trades |
| `train_all_models.py` | Train all 48 PatchTST + SAC/TD3/PPO + LightGBM + HMM models |
| `run_dl_lab.py` | Deep Learning Laboratory — 1,200 experiments across MLP/CNN/LSTM/PatchTST |
| `promote_best.py` | Read lab results and promote winning architectures to production |
| `robust_validation.py` | 7,800 anti-cheat walk-forward validation tests |
| `backtest_v2.py` | Backtest on historical data |
| `yearly_backtest.py` | Year-by-year results 2020–2026 |
| `live_paper_trade.py` | Real-time monitor; auto-closes on TP/SL |
| `paper_trade.py` | Quick one-shot signal report |
| `trade_log.py` | View trade history and accuracy stats |
| `daily_advisor.py` | Original v1 advisor — legacy, do not use |

---

## Project Structure
```
Project-for-PAI/
├── scripts/                              ← All runnable scripts
├── src/
│   ├── forecast/
│   │   ├── patchtst_forecast.py          ← PatchTST Transformer forecaster
│   │   └── lgbm_forecast.py              ← LightGBM forecaster (blended with PatchTST)
│   ├── lab/
│   │   ├── models.py                     ← MLP, CNN, LSTM model definitions
│   │   └── forecaster.py                 ← LabForecaster; same interface as PatchTST
│   ├── regime/
│   │   ├── hmm_regime.py                 ← Per-class HMM (equity/crypto/commodity/forex/semiconductor)
│   │   └── pretrained_hmm.py             ← Legacy fallback only
│   ├── rl/
│   │   ├── trading_env.py                ← Trading simulation environment
│   │   ├── sac_agent.py                  ← SAC agent
│   │   ├── td3_agent.py                  ← TD3 agent
│   │   ├── ppo_agent.py                  ← PPO agent
│   │   ├── macro_sentiment_features.py   ← VIX, interest rates, sentiment, peer momentum
│   │   └── integrated_pipeline.py        ← PatchTST signal builder
│   ├── trading/
│   │   ├── ensemble_trader.py            ← Ensemble signal combiner
│   │   └── consensus.py                  ← Consensus voting logic
│   └── utils/
│       ├── currency.py                   ← Live EUR/NGN/USD exchange rates
│       └── trade_logger.py               ← Persistent trade log
├── data/
│   ├── models/                           ← All saved models (150+ files)
│   │   ├── patchtst_*.pt                 ← PatchTST or promoted lab model per asset (48 files)
│   │   ├── sac_*.pt                      ← SAC agents
│   │   ├── td3_*.pt                      ← TD3 agents
│   │   ├── ppo_*.pt                      ← PPO agents
│   │   ├── lgbm_*.pkl                    ← LightGBM per asset (48 files)
│   │   └── hmm_*.pkl                     ← 5 HMM models (equity/crypto/commodity/forex/semiconductor)
│   ├── output/                           ← Daily signals, lab results, backtest output
│   └── cache/                            ← Per-asset macro feature cache
└── README.md                             ← This file
```

---

## Asset Universe (48 Assets)

| Category | Assets |
|----------|--------|
| US Indices | ^GSPC (S&P 500), ^IXIC (NASDAQ), ^DJI (Dow Jones) |
| US Stocks | AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, JPM, CRWV* |
| Crypto | BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD |
| Forex Major | EURUSD=X, GBPUSD=X, USDJPY=X, USDCHF=X, AUDUSD=X, NZDUSD=X, USDCAD=X |
| Forex Cross | EURGBP=X, EURJPY=X, GBPJPY=X, AUDNZD=X |
| African Forex | USDNGN=X, EURNGN=X, USDZAR=X, USDKES=X, USDGHS=X |
| Commodities | GC=F (Gold), SI=F (Silver), CL=F (Crude Oil), NG=F (Natural Gas), HG=F (Copper), ZW=F (Wheat), ZC=F (Corn) |

*CRWV (CoreWeave) is in the universe but currently skipped; not enough price history to train on yet.

---

## Agent Routing

| Asset Class | RL Agents Used | Why |
|------------|---------------|-----|
| Equities | SAC + TD3 | More predictable trends; SAC explores, TD3 stabilises |
| Crypto | PPO | High volatility; PPO's small updates prevent overreacting |
| Commodities | SAC + TD3 | Similar reasoning to equities |
| Forex | SAC + TD3 | Sequential dependencies across sessions |

All three agents (SAC, TD3, PPO) vote regardless; consensus requires 2 of 3.

---

## Risk Profiles

| Profile | Max per trade | Min confidence | Best for |
|---------|--------------|----------------|---------|
| 1 — Conservative | 5% | 100% unanimous | Small portfolio; capital preservation |
| 2 — Moderate | 10% | 75% (3 of 4 agree) | Balanced risk and reward |
| 3 — Aggressive | 20% | 50% (2 of 4 agree) | Higher risk tolerance |

---

## Position Sizing

Position size is calculated using the Kelly criterion; a formula that says bet more when the forecast is confident and the asset is calm, and bet less when the forecast is weak or the asset is volatile. The maximum allowed per asset is 25% of the portfolio. GPU-related stocks (NVDA, MSFT, META) are capped at a combined 25% to avoid concentration risk.

---

## Training Guide

### Hardware Used
- **Machine**: Taiwo's laptop — NVIDIA GeForce RTX 3050 (6GB VRAM), 20-core CPU, 16GB RAM
- **OS**: Ubuntu 24, Python 3.11, CUDA 12.8
- **Training time**: approximately 8 hours total for all 48 assets on first run

### Model Files

| Model type | Files | Trained on |
|-----------|-------|-----------|
| PatchTST or lab winner (48 files) | `patchtst_*.pt` | Each asset; 2004 to present |
| SAC agents | `sac_*.pt` | Each asset individually |
| TD3 agents | `td3_*.pt` | Each asset individually |
| PPO agents | `ppo_*.pt` | Crypto assets |
| LightGBM (48 files) | `lgbm_*.pkl` | Each asset; engineered features |
| HMM (5 files) | `hmm_*.pkl` | Per asset class |

### Retraining Commands

**Retrain everything from scratch:**
```bash
cd ~/Desktop/NCI/programming_for_ai/Project-for-PAI
nohup python3 scripts/train_all_models.py > data/output/training.log 2>&1 &
tail -f data/output/training.log
```

**Resume interrupted training (skips already saved models):**
```bash
python3 scripts/train_all_models.py
```

**Retrain LightGBM only (after market conditions change):**
```bash
rm data/models/lgbm_*.pkl && python3 scripts/train_all_models.py
```

**Re-run the Deep Learning Laboratory:**
```bash
python3 scripts/run_dl_lab.py --resume   # skips already completed runs
python3 scripts/promote_best.py          # promote winners to production
```

### When to Retrain

| Situation | What to retrain |
|-----------|----------------|
| Every 6 months | Everything |
| After a major market crash (>15% drop) | Everything |
| After a major rally (>20% gain) | Everything |
| One asset performing badly | That asset only |
| LightGBM giving extreme forecasts | LightGBM only (rm lgbm_*.pkl) |

### Monitor GPU During Training
```bash
watch -n 5 "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
```

### Check How Many Models Are Saved
```bash
ls data/models/patchtst_*.pt  | wc -l   # should be 48
ls data/models/sac_*.pt       | wc -l   # should be ~35
ls data/models/td3_*.pt       | wc -l   # should be ~5
ls data/models/ppo_*.pt       | wc -l   # should be ~5
ls data/models/lgbm_*.pkl     | wc -l   # should be 47
ls data/models/hmm_*.pkl      | wc -l   # should be 5
```

---

## Known Issues

| Issue | Status |
|-------|--------|
| CRWV (CoreWeave) skipped | Not enough price history; will include when data builds up |
| Intraday trading not supported | System is designed for daily and 5-day horizons only |

---

## Dependencies
```bash
pip install yfinance scikit-learn lightgbm hmmlearn torch joblib --break-system-packages
```

---

> This project is for educational purposes only. Not financial advice.
