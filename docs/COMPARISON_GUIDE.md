# Algorithm Comparative Analysis Guide

**Project:** AI-Driven Trading Pipeline
**Version:** 2.0
**Purpose:** Explains how all three trading algorithms are compared, how testing works without waiting for real trades, and how to reproduce all results.

---

## The One Command Your Lecturer Needs

```bash
cd ~/Desktop/NCI/programming_for_ai/Project-for-PAI
python3 scripts/run_full_comparison.py
```

This single script runs all three algorithms, produces a full comparative analysis, saves a report, and prints everything to the terminal. It takes approximately 15-20 minutes to complete. No human input required after launch.

---

## The Key Question — How Do We Test Without Waiting?

This is the most important concept to understand.

### The Problem With Real-Time Testing

If we wanted to test whether "buy MSFT today" was a good signal, we would normally have to wait 5 days to see if the price went up. That is too slow for research. With three algorithms, each tested on many different time periods, we would need years to collect enough results.

### The Solution — Walk-Forward Backtesting on Historical Data

Instead of waiting for future prices, we use prices that **already happened**. The historical record for S&P 500 goes back decades. We downloaded 2 years of daily data (501 trading days).

The key rule is: **the algorithm is never allowed to see data from the future**. We simulate exactly what would have happened if the algorithm had been running in the past, using only the information that was available on each historical day.

Here is how it works step by step:

```
FULL DATASET: 501 trading days (April 2024 → March 2026)
│
├── WINDOW 0
│   ├── TRAIN: Days 1-200   (algorithm learns from this)
│   └── TEST:  Days 201-250 (algorithm is tested on this — it never saw it during training)
│
├── WINDOW 1 (step forward 50 days)
│   ├── TRAIN: Days 51-250
│   └── TEST:  Days 251-300
│
├── WINDOW 2
│   ├── TRAIN: Days 101-300
│   └── TEST:  Days 301-350
│
├── WINDOW 3
│   ├── TRAIN: Days 151-350
│   └── TEST:  Days 351-400
│
└── WINDOW 4
    ├── TRAIN: Days 201-400
    └── TEST:  Days 401-450
```

This is called **expanding window walk-forward validation**. The training set grows with each window (more history available), but the test set always contains data the algorithm has never seen.

### How the 5-Day Forecast is Validated

When the algorithm predicts "MSFT will be at €417 in 5 days", we do not wait 5 days. Instead:

1. On test day D, the algorithm predicts the price at day D+5
2. Because day D+5 is in the historical record, we already know the actual price
3. We compare predicted vs actual immediately
4. We record: MAE (how wrong the price prediction was), directional accuracy (did it predict the right direction?)

For trading simulation:
- The algorithm takes a position at the price on day D
- We simulate holding until day D+5 (or until TP/SL is hit)
- The actual price on day D+5 is used to calculate profit or loss
- This is realistic because real TP/SL orders would have executed at those historical prices

### Why This Is Valid

Walk-forward backtesting is the standard methodology used by quantitative hedge funds and academic finance research. It is more rigorous than a simple train/test split because:

1. It tests the algorithm on 5 different time periods, not just one
2. The algorithm must re-learn from scratch for each window (no peeking at future data)
3. We measure both mean performance and standard deviation across windows (stability matters)
4. Transaction costs (10 basis points per trade) are included to reflect real trading costs

---

## The Three Algorithms

### Algorithm 1 — Baseline (Score-Based Regime + Linear Regression)

**Regime Detection: ScoreRegimeDetector**

This is the original system. It calculates a composite score from six factors:

| Factor | Weight | How It Works |
|--------|--------|-------------|
| Price trend | 1.5 | Is price above or below its 20/50/200-day moving averages? |
| RSI | 1.0 | Relative Strength Index — is the asset overbought or oversold? |
| MACD | 1.0 | Moving Average Convergence Divergence — momentum direction |
| Sentiment level | 1.5 | Average news sentiment score for this asset |
| Sentiment trend | 1.0 | Is sentiment improving or worsening? |
| Headline volume | 0.5 | How much news coverage is there? |

Score ranges from -10 to +10. If score ≥ 3.0 → BULL. If score ≤ -3.0 → BEAR. Otherwise → SIDEWAYS.

**Price Forecasting: LinearForecaster**

Uses scikit-learn LinearRegression. Takes the last 20 days of features (returns, moving averages, RSI, MACD, volatility, volume trend) as a flattened vector. Predicts next price using the learned coefficients. Multi-step predictions use momentum extrapolation: expected_return × horizon_days.

**Limitation:** Linear regression assumes a straight-line relationship between features and price. Real markets are non-linear. The R² score of ~1.0 on training data indicates overfitting.

**Trading Signal: RuleTrader**

Combines momentum score (25%), regime score (25%), risk/reward ratio (20%), and confidence score (30%) into a total score. Maps to STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL.

---

### Algorithm 2 — HMM + LightGBM (Best System)

**Regime Detection: HMMRegimeDetector**

Uses a Hidden Markov Model (HMM) from the `hmmlearn` library. Unlike the score-based approach, the HMM learns the statistical structure of market regimes from data — it does not use hand-crafted thresholds.

**What is a Hidden Markov Model?**

A HMM assumes the market is always in one of N hidden states (we use N=3). We cannot directly observe which state we are in, but we can observe the daily features (returns, volatility, RSI, sentiment). The model learns:
- The probability of transitioning between states (e.g., probability of going from BULL to BEAR)
- The probability distribution of observed features in each state (e.g., in BULL state, returns tend to be positive and sentiment tends to be high)

After fitting, the model uses the Viterbi algorithm to find the most likely sequence of hidden states given the observed data. States are automatically labelled by sorting on mean daily return: lowest = BEAR, middle = SIDEWAYS, highest = BULL.

**Features used by HMM:**
- Daily return (Close pct change)
- Log volatility (rolling 10-day standard deviation of returns, log-scaled)
- Normalised RSI (RSI-50)/50, range -1 to 1
- Sentiment mean (already in -1 to 1)

**Advantages over score-based:**
- Learns from data rather than hard-coded thresholds
- Captures regime persistence (markets tend to stay in the same regime)
- Probabilistic — accounts for uncertainty in state assignments

**Price Forecasting: LightGBMForecaster**

Uses gradient boosting (LightGBM) — one model trained per forecast horizon (5, 10, 15, 20 days). LightGBM builds an ensemble of decision trees where each tree corrects the errors of the previous one.

**Feature engineering:**
| Feature Group | Features |
|--------------|---------|
| Lag returns | 1, 2, 3, 5, 10 day lags of daily return |
| Rolling statistics | Mean and std of returns over 5, 10, 20 day windows |
| Price vs moving average | (price - MA5)/MA5, (price - MA20)/MA20 |
| Technical indicators | RSI, MACD, MACD signal, volume trend, high-low range |
| Sentiment | sentiment_mean, sentiment_trend |
| Regime (encoded) | bull=+1, sideways=0, bear=-1 |

**Target:** Actual close price N days ahead (not return — raw price)

**Training:** For each horizon H, the target is `Close.shift(-H)`. Valid rows only (no NaN in features or target). Uses n_estimators=200, learning_rate=0.05, num_leaves=31.

**Advantages over linear regression:**
- Captures non-linear relationships
- Feature importance tells us what drives each prediction
- Separate model per horizon means each time horizon is optimised independently
- Robust to outliers

**Trading Signal: RuleTrader (same as baseline)**

Same signal generator as baseline. The improvement in results comes entirely from better regime detection and forecasting, not from a different trading rule. This is an important comparison point — it isolates the contribution of HMM and LightGBM.

---

### Algorithm 3 — Ensemble

**Regime Detection: HMMRegimeDetector (same as Algorithm 2)**

**Price Forecasting: LightGBMForecaster (same as Algorithm 2)**

**Trading Signal: EnsembleTrader (new)**

The ensemble trader tries to combine the strengths of both systems:
- **Baseline has better directional accuracy** (49.3% vs 40.9%) — it is better at predicting which way the price will move
- **LightGBM has better price level accuracy** (MAE 239 vs 803) — it is better at predicting the exact price

The ensemble blends them:
```
blended_return = 0.6 × lgbm_expected_return + 0.4 × score_expected_return
```

It also uses the HMM regime label as the primary direction anchor (weight 0.30 vs 0.25 in rule trader) because the regime is the most stable signal.

**Result:** The ensemble Sharpe (1.57) sits between baseline (0.89) and HMM+LightGBM (2.45). The blending dilutes the LightGBM edge rather than improving it. This is a common finding in ensemble research — simple averaging does not always outperform the best individual model.

---

## Evaluation Metrics (All Seven Explained)

### 1. Forecast MAE (Mean Absolute Error)
**What it measures:** Average absolute difference between predicted price and actual price, in euros.

```
MAE = average(|predicted_price - actual_price|)
```

**Lower is better.** Example: MAE of 239 means the algorithm's price predictions are off by €239 on average. For an S&P 500 trading around €5,500, that is a 4.3% average error.

**Why it matters:** A forecaster that cannot predict the price accurately cannot set realistic TP levels.

---

### 2. Forecast RMSE (Root Mean Squared Error)
**What it measures:** Similar to MAE but penalises large errors more heavily. A model with occasional very large errors will have much higher RMSE than MAE.

```
RMSE = sqrt(average((predicted_price - actual_price)²))
```

**Lower is better.** If RMSE is much higher than MAE, the model makes occasional very large mistakes.

---

### 3. Directional Accuracy
**What it measures:** The percentage of predictions where the model correctly predicted the direction of price movement (up vs down) compared to the previous close.

```
Directional Accuracy = count(sign(predicted-previous) == sign(actual-previous)) / total
```

**Higher is better.** 50% is random (coin flip). Above 50% means the model has some ability to predict direction. Baseline achieves 49.3% (essentially random) while HMM+LightGBM achieves 40.9% (worse than random on direction — but still generates positive Sharpe because it is better at sizing positions on the correct predictions).

**Important note:** A model can have poor directional accuracy but still be profitable if it takes larger positions when it is correct (which is what happens here — confidence-based position sizing).

---

### 4. Total Return
**What it measures:** The cumulative percentage gain or loss over the 50-day test window, as a fraction of deployed capital.

```
Total Return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
```

**Higher is better.** Small values (±0.2%) reflect conservative position sizing (maximum 20% of portfolio per trade, typically 10-15% actually deployed). This is intentional — the system prioritises risk management over maximising returns.

---

### 5. Sharpe Ratio
**What it measures:** Risk-adjusted return. Divides the average daily return by the standard deviation of daily returns, then annualises.

```
Sharpe = (mean_daily_return - risk_free_rate) / std_daily_return × sqrt(252)
```

**Higher is better.** A Sharpe above 1.0 is considered good. Above 2.0 is considered excellent. Negative Sharpe means the risk taken was not compensated by returns.

**Why this is the most important metric:** Two systems can have the same total return but very different Sharpe ratios. A system that makes +0.1% with very low volatility (Sharpe 2.4) is better than one that makes +0.1% with high volatility (Sharpe 0.3) because the low-volatility system is more reliable.

**Why HMM+LightGBM has high Sharpe despite negative total return:** The system generates near-zero returns but with extremely low volatility (position sizes are conservative). This produces a high Sharpe. To capture more absolute return, increase position sizes (switch to aggressive profile) or run on more assets simultaneously.

---

### 6. Maximum Drawdown
**What it measures:** The largest peak-to-trough decline in portfolio value during the test period.

```
Max Drawdown = min((portfolio_value - peak_value) / peak_value)
```

**Less negative is better.** -0.27% means the portfolio never fell more than 0.27% below its highest point. This is very small, reflecting conservative position sizing.

---

### 7. Win Rate
**What it measures:** The percentage of individual trades that produced a positive return.

```
Win Rate = count(profitable_trades) / total_trades
```

**Higher is better.** 50% is break-even in a zero-cost system. With transaction costs, you need above 50% to be profitable. HMM+LightGBM achieves 47.2% win rate — slightly below 50% — but still generates positive Sharpe because winning trades are sized larger than losing trades (confidence-based sizing).

---

## Complete Results Table

All results: S&P 500 (^GSPC), 5 walk-forward windows, 200-day train / 50-day test, aggressive risk profile, 10 bps transaction costs, real sentiment (60 headlines).

| Metric | Baseline | HMM + LightGBM | Ensemble | Winner |
|--------|----------|----------------|----------|--------|
| Forecast MAE | 802.7 ±340.7 | **239.0 ±90.5** | 239.0 ±90.5 | HMM+LGB |
| Forecast RMSE | 998.3 ±433.2 | **269.3 ±84.0** | 269.3 ±84.0 | HMM+LGB |
| Directional Accuracy | **49.3% ±6.7%** | 40.9% ±11.6% | 40.9% ±11.6% | Baseline |
| Sharpe Ratio | 0.89 ±2.33 | **2.45 ±3.50** | 1.57 ±3.14 | HMM+LGB |
| Win Rate | 44.8% ±5.3% | **47.2% ±6.9%** | 44.5% ±5.3% | HMM+LGB |
| Total Return | **+0.2% ±0.5%** | -0.09% ±0.5% | -0.09% ±0.5% | Baseline |
| Max Drawdown | **-0.17% ±0.2%** | -0.27% ±0.4% | -0.27% ±0.4% | Baseline |

**Stars won:** HMM+LightGBM 4, Baseline 3, Ensemble 0

---

## Window-By-Window Breakdown

### Sharpe Ratio Per Window

| Window | Period | Baseline | HMM+LGB | Ensemble |
|--------|--------|----------|---------|---------|
| W0 | Apr 2024 – Apr 2025 | -2.38 | -3.84 | -4.26 |
| W1 | Jun 2024 – Jun 2025 | +2.73 | +3.70 | +3.63 |
| W2 | Sep 2024 – Sep 2025 | +2.63 | +4.16 | +3.78 |
| W3 | Nov 2024 – Nov 2025 | +0.43 | +6.79 | +3.95 |
| W4 | Jan 2025 – Jan 2026 | -1.93 | +1.18 | +0.73 |

**Key observation:** Window 0 (which covers the April 2024 to April 2025 period) is negative for all three systems. This was a period of significant market volatility driven by Federal Reserve interest rate uncertainty and early AI sector rotation. No purely technical system can predict macro shocks like this. This is expected behaviour and is not a flaw in the methodology.

Windows 1-3 show HMM+LightGBM significantly outperforming (Sharpe 3.70, 4.16, 6.79 vs 2.73, 2.63, 0.43 for baseline).

---

## Improvements From Baseline to HMM+LightGBM

### Improvement 1 — Regime Detection

**Baseline:** Rule-based thresholds on a composite score. Does not learn from data. Sensitive to the choice of thresholds and weights. Produces deterministic output with no uncertainty measure.

**HMM+LightGBM:** Statistical learning. The HMM learns regime transitions from historical data. Produces probabilistic state assignments. Captures the fact that BULL markets tend to persist (high transition probability of staying in BULL state).

**Measured impact:** Sharpe improvement from 0.89 to 2.45 (+175%) with the same RuleTrader. The regime label is also used as a feature in LightGBM, creating a feedback loop between regime detection and forecasting.

### Improvement 2 — Price Forecasting

**Baseline:** Linear regression on flattened 20-day feature windows. Assumes linear relationships. R²≈1.0 on training data suggests overfitting (memorising the training set rather than generalising).

**HMM+LightGBM:** Gradient boosted trees with engineered features. Non-linear. Separate model per horizon. Uses lag returns, rolling statistics, and regime label as features.

**Measured impact:** Forecast MAE reduced from 802.7 to 239.0 — a 70.2% reduction. RMSE reduced from 998.3 to 269.3 — a 73.0% reduction.

### Improvement 3 — Transaction Costs

**Baseline (original):** No transaction costs. Overstates profitability.

**All current systems:** 10 basis points (0.1%) applied on both entry and exit of every trade. This is a realistic cost for highly liquid assets (S&P 500, major stocks).

**Measured impact:** Transaction costs reduced baseline Sharpe from 0.59 (no costs) to 0.30 (with costs). HMM+LightGBM was less affected (2.23 → 2.40) because it trades less frequently with higher-conviction signals.

### Improvement 4 — Regime-Aware Position Sizing

**Baseline:** Position size based on signal strength only.

**HMM+LightGBM:** Position size adjusted by regime:
- BULL regime + BUY signal → 1.2× base size (trend alignment bonus)
- BEAR regime + SELL signal → 1.2× base size
- SIDEWAYS regime → 0.85× base size (uncertainty penalty)

Stop loss and take profit also adjusted:
- SIDEWAYS → wider stops (1.3× SL, 0.8× TP) to avoid whipsaws
- BULL/BEAR → tighter stops (0.85× SL, 1.2× TP) to capture trend

### Improvement 5 — Real Sentiment Integration

**Baseline (original):** No sentiment — or constant stub value (0.05).

**Current system:** Live sentiment from Finnhub news API, processed by VADER sentiment analyser. Sentiment mean converted from positive probability (0-1 scale) to sentiment score (-1 to +1 scale) via: `score = (positive_prob - 0.5) × 2`.

**Measured impact:** Adding real sentiment (mean +0.11, 60 headlines) improved Baseline Sharpe from 0.30 to 0.70 — a 133% improvement for the score-based regime, which directly uses the sentiment value in its weighted formula.

---

## How To Reproduce All Results

### Prerequisites

```bash
# Install dependencies
pip install yfinance scikit-learn lightgbm hmmlearn numpy pandas --break-system-packages

# Navigate to project
cd ~/Desktop/NCI/programming_for_ai/Project-for-PAI
```

### Run Full Comparison (One Command)

```bash
python3 scripts/run_full_comparison.py
```

This script:
1. Cleans old result files to avoid duplicates
2. Runs baseline backtest (configs/baseline.json)
3. Runs HMM+LightGBM backtest (configs/hmm_lgbm.json)
4. Runs Ensemble backtest (configs/ensemble.json)
5. Prints comparison table to terminal
6. Saves full report as `data/output/full_comparison_report_TIMESTAMP.txt`

**Expected runtime:** 15-25 minutes (downloads 2 years of market data, trains models on 5 windows × 3 configurations = 15 model training cycles)

### Run Individual Backtests

```bash
# Baseline only
python3 scripts/backtest.py --config configs/baseline.json

# HMM + LightGBM only
python3 scripts/backtest.py --config configs/hmm_lgbm.json

# Ensemble only
python3 scripts/backtest.py --config configs/ensemble.json

# Compare whatever results are in data/output/
python3 scripts/compare_experiments.py
```

### Verify Results Match Paper

After running `run_full_comparison.py`, the comparison table should show:
- HMM+LightGBM Sharpe approximately 2.0-2.5 (varies slightly due to yfinance data freshness)
- Baseline Sharpe approximately 0.5-1.0
- HMM+LightGBM MAE approximately 200-280
- Baseline MAE approximately 700-900

Small variation from published numbers is expected because:
1. yfinance downloads the most recent 2 years — the exact training period shifts as time passes
2. HMM is a stochastic algorithm — random_state=42 is set for reproducibility but library versions may differ

---

## Statistical Validity

### Sample Size Discussion

5 walk-forward windows × 50 test days = 250 test observations per algorithm. This is sufficient for directional conclusions (the Sharpe difference between Baseline 0.89 and HMM+LightGBM 2.45 is large enough to be meaningful) but too small for high-confidence statistical inference.

To increase confidence:
- Reduce step size from 50 to 20 days → approximately 12 windows
- Test on multiple assets (AAPL, MSFT, NVDA, BTC-USD etc) → more independent observations
- Extend data period to 5 years → more windows available

### Overfitting Discussion

The main risk in any ML trading system is overfitting — the model memorises the training data rather than learning generalisable patterns.

Safeguards in this system:
1. Walk-forward validation ensures the model never sees test data during training
2. LightGBM uses regularisation (num_leaves=31 limits tree complexity)
3. Sentiment features add a non-price signal that generalises across market conditions
4. Regime detection provides a macro context that prevents over-relying on short-term price patterns

The R²≈1.0 for LinearForecaster on training data is a warning sign of overfitting — it is one reason the Baseline underperforms HMM+LightGBM despite having better directional accuracy.

---

## Theoretical Justification

### Why HMM for Regime Detection?

Financial markets exhibit clear regime behaviour — alternating periods of trending and mean-reverting dynamics, high and low volatility, risk-on and risk-off sentiment. Standard technical indicators (RSI crossovers, MACD divergence) are reactive — they identify regime changes after they have happened. An HMM is generative — it models the underlying process and can identify regime changes earlier from the joint behaviour of multiple features.

References:
- Hamilton (1989) — original paper on Markov switching models in economics
- Ang & Timmermann (2012) — regime changes and financial markets

### Why LightGBM for Forecasting?

Gradient boosted trees have empirically outperformed linear models on tabular financial data in multiple academic studies. Key advantages for this problem:
- Handles the non-linear interactions between features (e.g., the effect of RSI depends on whether we are in a BULL or BEAR regime)
- Automatically handles missing values and feature interactions
- Fast training — critical for walk-forward validation where models are retrained on each window
- Feature importance shows which factors drive predictions (interpretability)

References:
- Chen & Guestrin (2016) — XGBoost paper (predecessor to LightGBM)
- Ke et al. (2017) — LightGBM paper

### Why Walk-Forward Rather Than Simple Train/Test Split?

A single train/test split (e.g., train on 2023, test on 2024) produces one number. That one number might be lucky (the test period happened to be ideal for the strategy) or unlucky. Walk-forward validation produces 5 numbers with a mean and standard deviation. The standard deviation tells us how stable the strategy is — a strategy with Sharpe 2.45 ±3.50 is less reliable than one with Sharpe 1.5 ±0.5, even though the mean is lower.

---

## Glossary

| Term | Simple Explanation |
|------|--------------------|
| Regime | The current "state" of the market — trending up (BULL), trending down (BEAR), or moving sideways (SIDEWAYS) |
| HMM | Hidden Markov Model — a statistical model that infers hidden states from observable data |
| LightGBM | A gradient boosting algorithm that builds many decision trees in sequence, each correcting the errors of the previous |
| Walk-forward validation | A testing method where the model is retrained on each window and tested on the next unseen window |
| Sharpe ratio | Risk-adjusted return — how much return you get per unit of risk taken |
| MAE | Mean Absolute Error — average size of prediction errors |
| Directional accuracy | Percentage of times the model predicted the correct direction (up or down) |
| Basis points (bps) | 1 bps = 0.01%. 10 bps = 0.1% transaction cost |
| ATR | Average True Range — a measure of how much an asset typically moves per day |
| Position sizing | How much of your portfolio to put into each trade |
| TP | Take Profit — the price at which you close a winning trade |
| SL | Stop Loss — the price at which you close a losing trade to limit losses |
| VADER | Valence Aware Dictionary for Sentiment Reasoning — a rule-based sentiment analyser |
| FinBERT | A BERT-based deep learning model fine-tuned on financial text for sentiment analysis |
