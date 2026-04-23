# MRAT-RL — AI Trading Signal System

**MSc Artificial Intelligence · NCI Dublin · Programming for AI 2026**

Group: Taiwo Alabi · Divyesh Reddy Ellasiri · Sai Vivek Yerninti · Mukesh Saren Ramu

**Individual Journal (Taiwo Alabi):** https://gist.github.com/iamtamilore/f44b01550d1e118faa421090a7cd01bc

---

## Run It Right Now — No Training Required

Pre-trained models are included in the repo. Clone and go:

```bash
git clone https://github.com/divyeshreddy0810/Project-for-PAI.git
cd Project-for-PAI
git checkout full_works_v2

pip install yfinance scikit-learn lightgbm hmmlearn torch joblib

python3 scripts/daily_advisor_v2.py
```

The advisor pulls **live price data from Yahoo Finance** on each run,
so you always get today's signals. When it asks:

- **Portfolio amount** — type any amount, e.g. `1000`
- **Risk profile** — type `2` (moderate) then press Enter

That is everything. No training step needed.

> **Important:** always run from the project root (`Project-for-PAI/`),
> not from inside the `scripts/` folder.

---

## What This System Does

Every morning, this system scans 48 global financial assets — stocks,
crypto, forex, commodities, and says: **BUY, SELL, or HOLD**.

It does not guess. It runs six AI models per asset, makes them vote,
and only fires a signal when at least two of three agents agree. If
they disagree, the system says HOLD and waits for a cleaner setup.

On 9 April 2026 it scanned 47 assets and produced:
- **17 BUY** signals, **2 SELL** signals, **28 HOLD** signals
- Average confidence: **59%**
- Highest confidence: USD/GHS at **100%** (unanimous across all four signal sources)

---

## Quick Start

```bash
# Install dependencies
pip install yfinance scikit-learn lightgbm hmmlearn torch joblib --break-system-packages

# Train all 48 models — first time only (~40 hours on a GPU)
python3 scripts/train_all_models.py

# Run the Deep Learning Laboratory — finds the best model per asset
python3 scripts/run_dl_lab.py

# Promote lab winners to production
python3 scripts/promote_best.py

# Run the daily advisor — scan all 48 assets, get today's top trades
python3 scripts/daily_advisor_v2.py
```

The daily advisor asks for your portfolio amount, your risk level (1–3),
then prints ranked signals with entry price, take-profit, stop-loss,
and position size in your currency.

---

## Concept Guide — Everything Explained From Scratch

> Before you read about the models, you need to understand what they
> are actually doing. Every idea below starts from the simplest possible
> version before adding any complexity.

---

### What Is a Time Series?

Imagine you weigh yourself every morning and write it down.

```
Monday:    82 kg
Tuesday:   81.8 kg
Wednesday: 82.1 kg
Thursday:  81.5 kg
```

That list is a **time series**, numbers recorded in order over time.

Stock prices are exactly this. Every trading day the market closes at
a price, and that number gets added to the list. The critical thing
about a time series is that **order matters**. Monday's weight came
before Tuesday's. Tuesday grew from Monday or fell from it. If you
shuffle the list and put Wednesday before Monday, you have destroyed
the information that makes it meaningful.

Everything in this system, the Transformer, the RL agents, the HMM —
is built around reading these ordered lists and finding patterns inside
them.

---

### What Is Training a Model?

A model starts as a blank page. It knows nothing.

You show it years of historical price data and say: "given everything
that happened up to this day, what happened next?" It makes a guess.
It was wrong. You tell it how wrong. It adjusts its internal settings
slightly. You show it the next example. It guesses again. Still wrong,
but a tiny bit less wrong. After thousands of examples and thousands of
adjustments, it starts to recognise patterns it was never explicitly
told to find.

That process, showing examples, measuring the error, adjusting,
repeating, is **training**.

---

### The Train/Test Split — The Cheat Sheet Problem

Picture an exam. The teacher writes 100 practice questions. You study
them all. You learn the answers. You ace the practice test. Then the
real exam comes, it has completely different questions. And you fail.

If we train a model on all available data and then test it on that same
data, we have the same problem. The model has already seen the answers.
It is not learning patterns, it is memorising. The test result looks
great. The live result is garbage.

The fix is simple: **split the data in time**.

```
─── 2004 ─────────────────── 2022 ──── 2024 ──── 2026 ───▶

  ████████████████████████ TRAIN  │  VAL  │  TEST
         (70%)                (15%)   (15%)
```

The model trains on the oldest 70%. It is tuned and evaluated on the
middle 15% (validation). It is tested on the final 15% — data it has
never touched. If it performs well on that final slice, the result is
honest.

---

### Walk-Forward Backtesting — The Real Exam

A single train/test split tells you how the model did on one stretch
of history. That is useful but limited. What if that particular stretch
was unusually calm? What if you happened to split at a lucky point?

**Walk-forward testing** runs the same exam hundreds of times, sliding
forward through history each time. Here is exactly how it works:

```
All available data for one asset: 5,611 days (Jan 2004 → Apr 2026)

Window 1:
  ████████████████████████████████  TRAIN (756 days = 3 years)
                                    ██████  TEST (63 days = 1 quarter)

Slide forward 21 days (1 month), then Window 2:
  ░░░░████████████████████████████████  TRAIN (756 days)
                                        ██████  TEST (63 days)

Slide forward again... Window 3, Window 4... all the way to 2026.
```

Each window is one complete exam. The model trains on three years of
data it has never seen in that window, then gets tested on the next
quarter of data it has never seen at all.

For this system: **7,800 windows** across 48 assets and 22 years of
history. 7,800 honest exams, each on a different slice of market
conditions, the 2008 crash, the 2020 pandemic, the 2021 bull run,
the 2022 rate hike sell-off.

The final Sharpe ratio of 0.91 is the average across all 7,800 of
those exam results. Not cherry-picked. Not lucky. Averaged.

---

### What Is Backtesting?

Backtesting is **replaying history** with your strategy and watching
what would have happened.

Think of it like watching a football match replay, except instead of
watching goals, you are watching whether your trades would have made
money. You already know how the game ended. You just want to see how
your strategy would have performed if it had been running at the time.

The danger is peeking at the future. If your model saw tomorrow's price
while deciding today's trade, even accidentally, through something as
subtle as fitting a scaler on the full dataset, the backtest will look
much better than reality. This is called **data leakage** and it is the
most common way AI trading backtests lie. Every result in this project
was produced under a strict no-leakage protocol.

---

### What Is a Market Regime?

A 15-degree day feels completely different in February versus August.
The temperature is the same number. The **context** — what season you
are in — changes everything.

Markets work the same way. A rising price inside a broad bull market
carries a different meaning than the same price movement during a
brief bounce inside a crash. The overall **regime**, bull, bear, or
sideways, is the season. Ignoring it means reading the temperature
without knowing the time of year.

The **Hidden Markov Model (HMM)** in this system detects the current
regime from price patterns alone, without needing to know the cause.
On 9 April 2026, all 47 scanned assets were classified as bull or
sideways. Zero bear classifications. That single output shaped the
entire signal distribution for the day.

---

### What Is Reinforcement Learning (RL)?

A dog trainer does not explain rules. They give treats when the dog
does the right thing and withhold treats when it does not. The dog
figures out the rules on its own through trial and reward.

An **RL agent** works exactly like this. It sits inside a simulated
trading environment. It takes an action, buy, sell, hold. The
environment gives it a reward (positive if the trade was good,
negative if it lost money). Over thousands of simulated trading days,
it slowly learns which actions in which market conditions tend to
produce positive rewards.

Three different RL agents are used here, SAC, TD3, and PPO, each
with slightly different learning strategies. Requiring two of three to
agree before acting means no single agent can drag the portfolio into
a bad trade on its own.

---

### What Is the Kelly Criterion?

Imagine you are at a casino and you know the coin is slightly biased,
heads comes up 55% of the time. How much of your money do you bet on
each flip?

Bet too little and you leave easy profit on the table.  
Bet everything and one bad run ruins you.

The **Kelly criterion** calculates the mathematically optimal fraction:
bet more when your edge is large and the outcome is predictable, bet
less when your edge is small or the outcome is uncertain.

This system uses **quarter-Kelly**, 25% of what the formula suggests,
because financial models are never perfectly accurate. Dialling it down
by 75% protects against the model being confidently wrong.

---

## Results

### Daily Advisory (9 April 2026)

| Signal | Count |
|--------|-------|
| BUY    | 17    |
| SELL   | 2     |
| HOLD   | 28    |

Average confidence: 59% · All regimes: BULL or SIDEWAYS · 5-day forecast range: +1.5% to -0.4%

### Walk-Forward Backtest (7,800 windows, 2004–2026)

| Metric | Result |
|--------|--------|
| Average annual return | +1.85% per window |
| Sharpe ratio | 0.91 |
| Profitable windows | 59% |
| Max drawdown, 2022 bear market | -1.2% |
| S&P 500 drawdown, 2022 | -20% |
| BTC drawdown, 2022 | -65% |

### Deep Learning Laboratory (1,200 experiments)

| Asset Class | Total | CNN/LSTM Promoted | Kept PatchTST |
|-------------|-------|-------------------|---------------|
| Equity      | 20    | 17                | 3             |
| Crypto      | 5     | 5                 | 0             |
| Commodity   | 7     | 5                 | 2             |
| Forex       | 16    | 12                | 4             |
| **Total**   | **48**| **39**            | **9**         |

No single architecture won everywhere. MLP never promoted on any asset.
CNN dominated crypto. LSTM dominated equities. PatchTST held on a focused
set of major forex pairs where long-range session patterns matter most.

---

## The Models

| Model | Role |
|-------|------|
| PatchTST Transformer | Reads 60 days of price history in 8-day chunks; predicts the 5-day forward return |
| LightGBM | Second forecaster; gradient boosted trees on technical features; blended 50/50 with PatchTST |
| HMM (5 models) | Detects bull / sideways / bear regime; one model per asset class |
| SAC Agent | RL agent for stable equities, forex, commodities; explores aggressively |
| TD3 Agent | RL agent for volatile stocks; uses two internal critics to stay cautious |
| PPO Agent | RL agent for crypto; small update steps to avoid overreacting to violent sessions |

---

## Scripts Reference

| Script | What it does |
|--------|-------------|
| `daily_advisor_v2.py` | **MAIN** — scan all 48 assets, print today's top trades |
| `train_all_models.py` | Train all 48 assets: PatchTST + SAC/TD3/PPO + LightGBM + HMM |
| `run_dl_lab.py` | Deep Learning Laboratory — 1,200 experiments across MLP/CNN/LSTM/PatchTST |
| `promote_best.py` | Read lab results and replace PatchTST with winner if it beats by >5% |
| `robust_validation.py` | 7,800 walk-forward validation windows |
| `backtest_v2.py` | Historical backtest on price data |
| `yearly_backtest.py` | Year-by-year breakdown 2020–2026 |
| `live_paper_trade.py` | Real-time monitor; auto-closes on take-profit or stop-loss |
| `paper_trade.py` | One-shot signal report |
| `trade_log.py` | View trade history and accuracy stats |

---

## Project Structure

```
Project-for-PAI/
├── scripts/                    ← All runnable scripts
├── src/
│   ├── forecast/               ← PatchTST and LightGBM forecasters
│   ├── lab/                    ← MLP, CNN, LSTM model definitions + LabForecaster
│   ├── regime/                 ← HMM regime detection (per asset class)
│   ├── rl/                     ← SAC, TD3, PPO agents + trading environment
│   │                             + macro/sentiment features
│   ├── trading/                ← Ensemble signal combiner + consensus voting
│   └── utils/                  ← Currency rates, trade logger
├── data/
│   ├── models/                 ← Saved model checkpoints (patchtst, sac, td3, ppo, lgbm, hmm)
│   ├── output/                 ← Daily signals, lab results, backtest output
│   └── cache/                  ← Per-asset macro feature cache
├── configs/                    ← Experiment configuration files
├── updated_PAI_report.tex      ← IEEE group report (LaTeX source)
├── Taiwo_Individual_Journal.txt← Individual development journal
└── README.md
```

---

## Asset Universe (48 Assets)

| Category | Assets |
|----------|--------|
| US Indices | ^GSPC (S&P 500), ^IXIC (NASDAQ), ^DJI (Dow Jones) |
| US Stocks | AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, JPM, CRWV* |
| Semiconductor | MU, SMCI, ARM, TSM, VRT, MRVL, NBIS, IREN |
| Crypto | BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD |
| Forex Major | EURUSD=X, GBPUSD=X, USDJPY=X, USDCHF=X, AUDUSD=X, NZDUSD=X, USDCAD=X |
| Forex Cross | EURGBP=X, EURJPY=X, GBPJPY=X, AUDNZD=X |
| African Forex | USDNGN=X, EURNGN=X, USDZAR=X, USDKES=X, USDGHS=X |
| Commodities | GC=F (Gold), SI=F (Silver), CL=F (Crude Oil), NG=F (Natural Gas), HG=F (Copper), ZW=F (Wheat), ZC=F (Corn) |

*CRWV (CoreWeave) listed March 2025. Only 268 rows of price history —
not enough to train reliably. Kept in the universe but signals
suppressed until data builds up.

---

## Agent Routing

| Asset Class | SAC | TD3 | PPO | Why |
|-------------|-----|-----|-----|-----|
| Equities (stable) | ✓ | | | Predictable trend structure suits SAC's exploration |
| Equities (volatile) | | ✓ | | TD3's two critics prevent overreacting to single-session spikes |
| Crypto | | | ✓ | Extreme volatility; PPO's clipped updates prevent catastrophic collapse |
| Forex | ✓ | | | Sequential session dependencies reward structured memory |
| Commodities | ✓ | | | Similar reasoning to stable equities |

All three agents vote regardless of asset class. A signal requires at
least two votes in the same direction.

---

## Training Guide

### Hardware
- GPU: NVIDIA GeForce RTX 3050 (6GB VRAM)
- CPU: 20-core Intel, 16GB RAM
- OS: Ubuntu 24, Python 3.12, CUDA 12.8
- Full cold-start training time: **~40 hours** across all 48 assets

### Run training from the project root (important)
```bash
cd ~/Desktop/NCI/programming_for_ai/Project-for-PAI
nohup python3 scripts/train_all_models.py > data/output/training.log 2>&1 &
tail -f data/output/training.log
```

> Always run from the project root. Running from inside `scripts/`
> causes models to save to `scripts/data/models/` instead of
> `data/models/`, which the advisory pipeline cannot find.

### Check how many models are saved
```bash
ls data/models/patchtst_*.pt | wc -l   # target: 48
ls data/models/sac_*.pt      | wc -l   # target: ~35
ls data/models/td3_*.pt      | wc -l   # target: ~9
ls data/models/ppo_*.pt      | wc -l   # target: ~5
ls data/models/lgbm_*.pkl    | wc -l   # target: 47
ls data/models/hmm_*.pkl     | wc -l   # target: 5
```

### When to retrain

| Situation | Action |
|-----------|--------|
| Every 6 months | Full retrain |
| After a major crash or rally (>15%) | Full retrain |
| One asset performing badly | Delete that asset's files only |
| Extreme forecasts reappearing | Check state_dim matches env dimensions |

---

## Known Issues

| Issue | Status |
|-------|--------|
| CRWV (CoreWeave) suppressed | Only 268 rows of history; Sharpe 6.009 during training is an overfit artifact, not a real result |
| No intraday support | Built for daily and 5-day horizons only |
| State dimension must match env | RL agents must be initialised with state_dim=19; env provides 19 dimensions |

---

## Dependencies

```bash
pip install yfinance scikit-learn lightgbm hmmlearn torch joblib --break-system-packages
```

---

> Educational project. Not financial advice.
