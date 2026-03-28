# MRAT-RL: Multimodal Regime-Adaptive Trading System

**MSc Artificial Intelligence — NCI Dublin — Programming for AI (MSCAIJAN26I)**
**Group:** Alabi Taiwo · Divyesh Reddy Ellasiri · Sai Vivek Yerninti · Mukesh Saren Ramu

---

## What This System Does

An AI trading signal generator that combines five data sources to decide
when to buy, sell, or hold an asset:

- **PatchTST Transformer** — price forecasting trained on 22 years of data
- **Hidden Markov Model** — detects bull / bear / sideways market regimes
- **SAC + PPO RL agents** — learn trading strategy from experience
- **Macroeconomic features** — VIX fear index + interest rates
- **FinBERT + VADER** — news sentiment analysis

## Key Results

| Metric | Baseline | Enhanced | Improvement |
|--------|----------|----------|-------------|
| Forecast MAE (S&P 500) | $3,674 | $79 | **97.8% better** |
| Avg Sharpe ratio | −0.40 | +0.54 | **9× better** |
| Walk-forward Sharpe | 1.387 | 1.853 | **+33.6%** |
| Combined system Sharpe | — | **+0.766** | 3 assets |

## Quick Start
```bash
# Install dependencies
pip install yfinance scikit-learn lightgbm hmmlearn transformers torch --break-system-packages

# Run enhanced pipeline (equities + crypto + gold) — ~40 seconds
python3 scripts/enhanced_pipeline.py \
  --symbols "^GSPC,BTC-USD,GC=F" \
  --risk moderate \
  --portfolio 100000

# Run original daily advisor (forex pairs)
python3 scripts/daily_advisor.py
```

## Project Structure
```
Project-for-PAI/
├── scripts/
│   ├── enhanced_pipeline.py      ← MAIN: Run this daily (enhanced system)
│   ├── daily_advisor.py          ← Original advisor (forex pairs)
│   ├── live_paper_trade.py       ← Real-time price monitor
│   ├── paper_trade.py            ← Quick signal report
│   ├── backtest.py               ← Walk-forward backtester
│   └── trade_log.py              ← View trade history
├── src/
│   ├── forecast/
│   │   ├── patchtst_forecast.py  ← PatchTST Transformer forecaster
│   │   └── lgbm_forecast.py      ← LightGBM forecaster
│   ├── regime/
│   │   ├── hmm_regime.py         ← HMM regime detector
│   │   └── pretrained_hmm.py     ← Fallback HMM only — per-class models (hmm_equity/crypto/commodity/forex.pkl) are used for all 39 assets
│   ├── rl/
│   │   ├── trading_env.py        ← Trading environment
│   │   ├── sac_agent.py          ← SAC agent (equities/gold)
│   │   ├── ppo_agent.py          ← PPO agent (crypto)
│   │   ├── rl_trainer.py         ← Training pipeline
│   │   ├── macro_sentiment_features.py  ← VIX + rates + sentiment
│   │   ├── integrated_pipeline.py       ← PatchTST + PPO pipeline
│   │   ├── ablation_study.py     ← Component importance test
│   │   ├── compare_agents.py     ← PPO vs SAC comparison
│   │   └── regime_stress_test.py ← 75 unseen window stress test
│   ├── trading/
│   │   ├── ensemble_trader.py    ← Ensemble signal combiner
│   │   └── rule_trader.py        ← Rule-based baseline
│   └── utils/
│       ├── currency.py           ← Live EUR/NGN/USD rates
│       ├── sentiment_loader.py   ← Loads FinBERT output
│       └── trade_logger.py       ← Persistent trade log
├── data/
│   ├── models/                   ← Saved RL models (SAC + PPO)
│   ├── output/                   ← Generated signals + reports
│   └── cache/                    ← Macro feature cache
├── configs/                      ← Backtest experiment configs
├── FRONTEND_INTEGRATION.md       ← How to add toggle to frontend
└── MRAT_RL_IEEE_Short_Paper.docx ← Research paper
```

## Asset Routing

| Asset | Agent | Notes |
|-------|-------|-------|
| S&P 500 (^GSPC) | SAC | Sharpe +0.68 |
| NASDAQ (^IXIC) | SAC | Sharpe +0.73 |
| Bitcoin (BTC-USD) | PPO | Sharpe +0.71 |
| Gold (GC=F) | SAC | Sharpe +0.59 |
| EUR/USD | SAC | Sharpe +0.36 |
| GBP/USD | SAC | Sharpe +0.22 |

## Frontend Integration (for Divyesh)

See **FRONTEND_INTEGRATION.md** for full details.

**Quick version:**
```javascript
// Baseline mode (existing)
exec('python3 scripts/master_pipeline_v2.py')
readFile('data/output/latest.json')

// Enhanced mode (new toggle)
exec('python3 scripts/enhanced_pipeline.py --symbols "^GSPC,BTC-USD,GC=F"')
readFile('data/output/enhanced_latest.json')
```

Both produce the same JSON structure. Enhanced adds:
`signal.confidence`, `signal.votes`, `signal.regime`, `signal.rl_agent`

## Ablation Study Findings

Testing what happens when each component is removed:

| Removed | Impact on S&P 500 Sharpe | Lesson |
|---------|--------------------------|--------|
| Macro features (VIX + rates) | −1.27 (crashed) | **Most critical component** |
| PatchTST | +0.31 (improved) | Robot learns patterns itself |
| HMM regime | +0.28 (improved) | Robot learns mood itself |
| Sentiment | +0.03 (no change) | Minimal impact |

## First Run Notes

- First run trains RL agents (~2 min per asset)
- After first run, models save to `data/models/` and load instantly
- Second run takes ~40 seconds total

## Saved Models

Pre-trained models included in repo:
- `data/models/sac_GSPC.pt` — S&P 500
- `data/models/sac_GCF.pt` — Gold
- `data/models/sac_IXIC.pt` — NASDAQ
- `data/models/sac_EURUSDX.pt` — EUR/USD
- `data/models/sac_GBPUSDX.pt` — GBP/USD
- `data/models/ppo_BTC_USD.pt` — Bitcoin

---

*This project is for educational purposes only. Not financial advice.*
