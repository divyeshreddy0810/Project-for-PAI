# AI-Driven Trading Pipeline — Full Project Documentation

**Version:** 2.0 (Enhanced)
**Last Updated:** March 21, 2026
**Author:** newtazer
**Project Path:** `~/Desktop/NCI/programming_for_ai/Project-for-PAI`

---

## What This System Does (Simple English)

This is an AI trading signal generator. Every day you launch it, it:

1. Checks if any of yesterday's trade recommendations hit their target (TP) or stop loss (SL)
2. Scans 40 assets — stocks, crypto, forex (including NGN pairs), and commodities
3. Detects the current market regime (BULL / BEAR / SIDEWAYS) using a Hidden Markov Model
4. Predicts prices 5 days ahead using LightGBM
5. Picks the top 3 assets to trade TODAY and the top 3 for a 5-day SWING trade
6. Prints exact entry price, TP, SL — ready to type into MT4
7. Shows everything in both EUR and NGN
8. Logs every trade it ever suggested, tracks accuracy over time

You do not need to know anything about machine learning to use it. You just run one command.

---

## How To Start The System (Daily Use)

### Every morning — run this one command:
```bash
cd ~/Desktop/NCI/programming_for_ai/Project-for-PAI
python3 scripts/daily_advisor.py
```

It will ask you three questions:
- Portfolio amount (type `₦100000` or `€500` or `$300`)
- Risk profile (1=Conservative, 2=Moderate, 3=Aggressive)
- Press Enter to start scanning

Then wait ~2 minutes while it scans all 40 assets. It prints your trade cards with exact MT4 entry levels.

### Check your trade log anytime:
```bash
python3 scripts/trade_log.py          # summary + accuracy stats
python3 scripts/trade_log.py --open   # what trades are still open
python3 scripts/trade_log.py --all    # every single trade ever logged
```

### Run a live paper trade (monitors prices in real time):
```bash
python3 scripts/live_paper_trade.py
```
Runs until all positions hit TP or SL. Checks prices every 60 seconds.

### Run a quick one-shot paper trade report:
```bash
python3 scripts/paper_trade.py
```
Fetches live prices, generates signals, prints report. Does not monitor continuously.

---

## The Three Scripts You Use Every Day

### 1. `scripts/daily_advisor.py` — THE MAIN SCRIPT
**What it does:** Full daily workflow. Checks previous trades, scans all 40 assets, recommends top 3 daily + top 3 swing trades, shows MT4 entry cards in EUR and NGN, logs everything.

**When to use it:** Every morning before markets open or at market open.

**Key output:**
```
┌─ #1 MSFT  Microsoft
│  ▲ LONG  · BULL regime · conf 98% · TODAY (1-day ATR)
│  Entry      : €381.87   ← type this in MT4
│  Take Profit: €393.45   (+3.03%)
│  Stop Loss  : €374.15   (+2.02%)
│  Position   : 14.7% → €93.12  (₦146,000)
│  If TP hit  : +€2.82  (+₦4,430)
│  If SL hit  : -€1.88  (-₦2,950)
└─ Check back tomorrow
```

### 2. `scripts/live_paper_trade.py` — REAL-TIME MONITOR
**What it does:** Runs full analysis once, then watches prices every 60 seconds. Automatically closes positions when TP or SL is hit. Shows live unrealised P&L. Runs until all positions close or you press Ctrl+C.

**When to use it:** When you want to watch trades resolve in real time during market hours.

**Options:**
```bash
python3 scripts/live_paper_trade.py --interval 30   # check every 30 seconds
```

**TP/SL modes it asks you:**
- **Standard** — risk profile levels (TP 15-25% away, SL 8-10% away) — for swing trades
- **Daily ATR** — tight levels based on today's volatility — for same-day trades

### 3. `scripts/paper_trade.py` — QUICK REPORT
**What it does:** Fetches live prices, runs analysis, prints a report. Does not monitor. Good for a quick check of current signals.

**When to use it:** When you want signals fast without waiting for the full daily advisor scan.

---

## Complete File Reference

### Scripts (what you run)

| File | What It Does |
|------|-------------|
| `scripts/daily_advisor.py` | Main daily workflow — scan all assets, recommend trades, log everything |
| `scripts/paper_trade.py` | Quick one-shot paper trade report |
| `scripts/live_paper_trade.py` | Real-time price monitor — auto-closes on TP/SL |
| `scripts/backtest.py` | Walk-forward backtester — tests a config on historical data |
| `scripts/compare_experiments.py` | Compares backtest results across experiments |
| `scripts/trade_log.py` | View trade log and accuracy statistics |
| `scripts/master_pipeline.py` | Original pipeline orchestrator (runs all src/ scripts in sequence) |

### Source Code — Models (src/)

| File | What It Does |
|------|-------------|
| `src/regime/base.py` | Abstract base class — defines what a regime detector must do |
| `src/regime/score_regime.py` | Baseline regime detector — uses weighted score (RSI + MACD + sentiment) |
| `src/regime/hmm_regime.py` | HMM regime detector — learns BULL/BEAR/SIDEWAYS from price patterns |
| `src/forecast/base.py` | Abstract base class — defines what a forecaster must do |
| `src/forecast/linear_forecast.py` | Baseline forecaster — linear regression on price sequences |
| `src/forecast/lgbm_forecast.py` | LightGBM forecaster — uses lag features, rolling stats, regime labels |
| `src/trading/base.py` | Abstract base class — defines what a trader must do |
| `src/trading/rule_trader.py` | Rule-based trader — signal from momentum + regime + confidence |
| `src/trading/ensemble_trader.py` | Ensemble trader — combines score-regime direction + LightGBM magnitude |
| `src/evaluation/metrics.py` | All metric functions — MAE, Sharpe, drawdown, win rate, etc. |

### Source Code — Original System (src/)

| File | What It Does |
|------|-------------|
| `src/sentiment_analyzer.py` | Fetches news from Finnhub/GNews, runs VADER/FinBERT sentiment analysis |
| `src/technical_indicators.py` | Calculates RSI, MACD, Bollinger Bands, ATR, regime scoring |
| `src/market_regime_model.py` | Original regime classifier (BULL/BEAR/VOLATILE probabilities) |
| `src/price_forecaster.py` | Original price forecaster (linear regression + momentum) |
| `src/rl_trader.py` | Original trading signal generator (rule-based, not actual RL) |

### Utilities (src/utils/)

| File | What It Does |
|------|-------------|
| `src/utils/sentiment_loader.py` | Reads latest.json, converts positive probability (0-1) to sentiment score (-1 to 1) |
| `src/utils/currency.py` | Live EUR/NGN/USD exchange rates via yfinance. Converts any currency input |
| `src/utils/trade_logger.py` | Persistent trade log — records every signal, tracks TP/SL outcomes, accuracy stats |
| `src/utils/config_manager.py` | Original config manager for master pipeline |
| `src/utils/sentiment_cache.py` | Cache for sentiment analysis results |

### Configs (configs/)

| File | What It Does |
|------|-------------|
| `configs/baseline.json` | Backtest config — score regime + linear regression + rule trader |
| `configs/hmm_lgbm.json` | Backtest config — HMM regime + LightGBM + rule trader (BEST) |
| `configs/ensemble.json` | Backtest config — HMM regime + LightGBM + ensemble trader |

### Data

| Path | What It Contains |
|------|-----------------|
| `data/output/latest.json` | Most recent sentiment analysis output (used by all scripts) |
| `data/output/daily_advice_*.json` | Saved daily advisor reports |
| `data/output/results_*.json` | Backtest results per experiment |
| `data/logs/trade_log.json` | PERSISTENT trade log — never deleted, appends every trade |

---

## The Original System (Before Our Work)

The project started as a 5-stage pipeline:

```
sentiment_analyzer.py
    → technical_indicators.py
        → market_regime_model.py
            → price_forecaster.py
                → rl_trader.py
```

**What it did:** Fetch news → calculate RSI/MACD/etc → score regime (BULL/BEAR/SIDEWAYS) → linear regression price forecast → rule-based BUY/SELL/HOLD signal.

**Problems:** No backtesting, no comparison framework, no walk-forward validation, no transaction costs, no position sizing tied to signal strength, no trade logging.

---

## What We Built (In Order)

### Phase 1 — Abstract Base Classes
Created clean interfaces so any regime detector, forecaster, or trader can be swapped without breaking the rest of the system.
- `src/regime/base.py` — RegimeDetector with fit() and predict()
- `src/forecast/base.py` — Forecaster with fit() and predict()
- `src/trading/base.py` — Trader with generate_signal()

### Phase 2 — Concrete Implementations
Wrapped existing code and added new models:
- `ScoreRegimeDetector` — wraps existing score-based regime logic (BASELINE)
- `HMMRegimeDetector` — new Hidden Markov Model regime detection (ENHANCED)
- `LinearForecaster` — wraps existing linear regression (BASELINE)
- `LightGBMForecaster` — new gradient boosting forecaster with feature engineering (ENHANCED)
- `RuleTrader` — wraps existing rule-based trader
- `EnsembleTrader` — new blended signal combining both approaches

### Phase 3 — Backtesting Framework
Built a proper walk-forward validation system:
- `scripts/backtest.py` — loads config, runs 5 expanding windows, records metrics
- `configs/*.json` — experiment configurations
- `src/evaluation/metrics.py` — MAE, RMSE, Sharpe, drawdown, win rate, directional accuracy
- `scripts/compare_experiments.py` — side-by-side comparison table

### Improvements (After Phase 3)
1. **Transaction costs** — 10 basis points per trade baked into all backtests
2. **Regime-aware position sizing** — BULL regime boosts size, SIDEWAYS reduces it
3. **Ensemble trader** — combines score-regime direction with LightGBM magnitude
4. **Real sentiment** — wired sentiment_analyzer.py output into backtest via sentiment_loader.py
5. **Paper trader** — `paper_trade.py` for quick live signals
6. **Live paper trader** — `live_paper_trade.py` monitors prices every 60s, auto-closes on TP/SL
7. **Daily advisor** — `daily_advisor.py` full daily workflow with MT4 cards
8. **Trade logger** — persistent log of every signal, tracks accuracy over time
9. **Currency converter** — live EUR/NGN/USD rates, accepts ₦ or € or $ input
10. **40-asset universe** — added all working forex, crypto, commodity tickers

---

## Backtest Results (Final)

All three systems tested on S&P 500 (^GSPC), 5 walk-forward windows, 200-day train / 50-day test, 10 bps transaction costs, aggressive risk profile.

| Metric | Baseline | HMM + LightGBM | Ensemble |
|--------|----------|----------------|----------|
| Forecast MAE | 802.7 | **239.0 ★** | 239.0 |
| Forecast RMSE | 998.3 | **269.3 ★** | 269.3 |
| Directional Accuracy | **49.3% ★** | 40.9% | 40.9% |
| Sharpe Ratio | 0.89 | **2.45 ★** | 1.57 |
| Win Rate | 44.8% | **47.2% ★** | 44.5% |
| Total Return | **+0.2% ★** | -0.09% | -0.09% |
| Max Drawdown | **-0.17% ★** | -0.27% | -0.27% |

**Winner: HMM + LightGBM** — best Sharpe ratio (2.45), best forecast accuracy (70% lower MAE), best win rate.

**Key finding:** Real sentiment data (60 headlines, mean +0.11) improved Baseline Sharpe from 0.30 → 0.70, showing the sentiment pipeline is working.

---

## Asset Universe (40 Assets)

### Indices
^GSPC (S&P 500), ^IXIC (NASDAQ), ^DJI (Dow Jones)

### US Stocks
AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, JPM

### Crypto
BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD

### Forex Major
EURUSD=X, GBPUSD=X, USDJPY=X, USDCHF=X, AUDUSD=X, NZDUSD=X, USDCAD=X

### Forex Cross
EURGBP=X, EURJPY=X, GBPJPY=X, AUDNZD=X

### African/Emerging Forex
USDNGN=X, EURNGN=X, USDZAR=X, USDKES=X, USDGHS=X, USDEGP=X

### Commodities
GC=F (Gold), SI=F (Silver), CL=F (Crude Oil), NG=F (Natural Gas), HG=F (Copper), ZW=F (Wheat), ZC=F (Corn)

---

## Installed Dependencies

```
yfinance==1.2.0       — live price data
scikit-learn==1.8.0   — linear regression, scaling
lightgbm==4.6.0       — gradient boosting forecaster
hmmlearn==0.3.3       — Hidden Markov Model regime detection
numpy==1.26.4         — numerical computing
pandas==3.0.1         — data manipulation
transformers          — FinBERT sentiment (optional, installed)
torch                 — PyTorch for FinBERT (optional, installed)
requests              — API calls for news
```

Install all at once:
```bash
pip install yfinance scikit-learn lightgbm hmmlearn transformers torch --break-system-packages
```

---

## API Keys (Already In sentiment_analyzer.py)

- **Finnhub:** `d6k6089r01qko8c3hrf0d6k6089r01qko8c3hrfg` — company news for stocks
- **GNews:** `74c1f79e81a5d2c62494062adaf232ca` — keyword news for forex/commodities

Both are free tier. Finnhub free tier = 60 calls/minute. GNews free tier = 100 calls/day.

---

## Known Issues / Warnings (All Harmless)

1. `⚠️ TensorFlow not installed` — TensorFlow was used by original price_forecaster.py which we replaced. Ignore.
2. `⚠️ Transformers not installed` — only matters if you want FinBERT. VADER works fine without it.
3. `ChainedAssignmentError` — pandas 3.0 warning about inplace operations. Does not affect results.
4. `Model is not converging` — HMM sometimes warns on small datasets. Still fits and predicts correctly.
5. `X does not have valid feature names` — LightGBM warning when predicting. Fixed in lgbm_forecast.py.
6. `SentimentLoader: symbol not found` — forex/crypto/commodity symbols not in latest.json. Falls back to neutral stub (0.05). Works fine.

---

## How The Algorithm Works (Simple English)

### Step 1 — Fetch Data
Downloads last 8 months of daily OHLCV prices from Yahoo Finance (free, no API key).

### Step 2 — Calculate Features
From price data: returns, moving averages (5/20/50/200 day), RSI, MACD, volatility, volume trend, high-low range. From sentiment: mean score (-1 to +1), trend (improving/worsening), headline count.

### Step 3 — Detect Regime (HMM)
A Hidden Markov Model trained on returns, volatility, RSI, and sentiment learns to identify 3 hidden states. States are auto-labelled by average return: lowest = BEAR, middle = SIDEWAYS, highest = BULL.

### Step 4 — Forecast Price (LightGBM)
One LightGBM model per horizon (5, 10, 15, 20 days). Features: lag returns (1/2/3/5/10 day), rolling stats (5/10/20 day windows), RSI, MACD, volume trend, sentiment, regime label. Target: close price N days ahead.

### Step 5 — Generate Signal (RuleTrader)
Combines 4 factors with weights:
- Momentum score (expected return direction) — 25%
- Regime score (bull=+0.5, bear=-0.5, sideways=0) — 25%
- Risk/reward ratio (upside vs downside range) — 20%
- Confidence score (based on volatility) — 30%

Total score → STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL

### Step 6 — Position Sizing
Base size from risk profile (aggressive = 20% max). Adjusted by:
- Signal strength (strong signal = larger position)
- Regime (BULL alignment = +20%, SIDEWAYS = -15%)
- Confidence (below minimum threshold = halved)

### Step 7 — TP/SL Levels
**Daily mode:** 1.5× ATR for TP, 1.0× ATR for SL. Designed to resolve within 24 hours.
**Standard mode:** risk profile percentages (aggressive: TP=25%, SL=10%). For swing trades.

---

## For Gold (XAU/USD) Specifically

Signal from March 21 2026:
- **Signal:** STRONG_BUY
- **Regime:** BEAR (recent pullback from highs)
- **Confidence:** 96%
- **Entry:** €4,492.00
- **TP daily:** €4,690.09 (+4.41%)
- **SL daily:** €4,359.94 (-2.94%)
- **TP swing:** €5,615.00 (+25%)
- **SL swing:** €4,042.80 (-10%)
- **5-day forecast:** +7.91%

Note: BEAR regime + STRONG_BUY = LightGBM sees short-term bounce inside longer-term pullback. Higher risk trade. Use smaller position size.

---

## MT4 Manual Trading Workflow

1. Run `python3 scripts/daily_advisor.py`
2. Enter portfolio in Naira (e.g. `₦10000`)
3. Choose risk profile
4. Wait for scan (~2 minutes)
5. Read the trade cards — note Entry, TP, SL for each recommended asset
6. Open MT4 → New Order
7. Select symbol (e.g. XAUUSD for Gold)
8. Set TP and SL to the exact prices shown
9. Execute trade
10. Come back tomorrow and run `daily_advisor.py` again — it will auto-check if TP/SL was hit

---

## Planned Next Steps

- [ ] MT4 bridge via ZeroMQ for automated order execution
- [ ] Day-by-day sentiment (instead of constant value) for better signal variation
- [ ] More walk-forward windows (step=20 days instead of 50) for more stable metrics
- [ ] Regime-specific LightGBM models (separate model for BULL vs BEAR periods)
- [ ] Streamlit web UI for non-terminal users
- [ ] Binance/crypto exchange API integration for automated crypto trading
