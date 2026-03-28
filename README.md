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

## Dependencies
```bash
pip install yfinance scikit-learn lightgbm hmmlearn transformers torch --break-system-packages
```

---

> This project is for educational purposes only. Not financial advice.
