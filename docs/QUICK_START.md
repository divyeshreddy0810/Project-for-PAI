# 🚀 QUICK START GUIDE - AI Trading Pipeline

## Installation (2 minutes)

```bash
# Install required packages
pip install pandas numpy requests transformers torch yfinance scikit-learn

# Optional (advanced features)
pip install tensorflow keras
```

## Run Complete System (5 minutes)

### Method 1: Fully Automated
```bash
python3 master_pipeline.py --auto
```
This runs all 5 steps with default settings.

### Method 2: Interactive
```bash
python3 master_pipeline.py --full
```
Choose what to run at each step.

### Method 3: Step-by-Step Manual
```bash
# Step 1: Analyze news sentiment
python3 sentiment_analyzer.py

# Step 2: Technical analysis (choose asset)
python3 technical_indicators.py
# Input: 1  [for S&P 500] or 'all' [for all assets]

# Step 3: Classify market regime
python3 market_regime_model.py

# Step 4: Forecast prices
python3 price_forecaster.py

# Step 5: Generate trading signals
python3 rl_trader.py
# Input: 2  [Moderate risk] or 1-3 [your preference]
```

---

## Check System Status

```bash
python3 master_pipeline.py --check

# Output:
# ✅ Sentiment Analysis: Ready
# ✅ Technical Indicators: Ready
# ✅ Market Regime Model: Ready
# ✅ Price Forecaster: Ready
# ✅ Trading Signal Generator: Ready
```

---

## View System Report

```bash
python3 master_pipeline.py --report

# Displays comprehensive system documentation
```

---

## Sample Output (5-day forecast)

```
==============================================================================
TRADING SIGNAL: ^GSPC (S&P 500)
==============================================================================

🔼 SIGNAL: BUY
   Strength: +0.62/1.0
   Confidence: 85%

💰 PRICE TARGETS:
   Current:  $5834.50
   Predicted (5D): $5900.00
   Expected Return: +1.12%

🎯 RISK MANAGEMENT:
   Stop Loss:  $5543.78 (-5.0%)
   Take Profit: $6417.95 (+10.0%)
   Risk/Reward Ratio: 2.15:1
   Downside Risk:  5.0%

📊 POSITION SIZING:
   Order Size:   10.0% of portfolio
   Order Value:  $10,000.00 (assuming $100k portfolio)

✅ RECOMMENDATION:
   Buy: EV=+0.73%, Confidence=85%, Size=10.0%
```

---

## Output Files Generated

After running the pipeline, check `data/output/`:

```
Latest Sentiment:     latest.json
Latest Regime:        market_regime_20260308_134638.json
Latest Forecast:      price_forecast_20260308_134638.json
Latest Signals:       trading_signals_20260308_134638.json
```

All files are in both **CSV** (for spreadsheets) and **JSON** (for integration).

---

## Asset Coverage (11 total)

```
INDICES      CRYPTO          FOREX        COMMODITIES
─────────────────────────────────────────────────────
S&P 500      Bitcoin         EUR/USD      Gold
NASDAQ       Ethereum        GBP/USD      Crude Oil
Euro Stoxx   Solana
Nikkei 225
```

---

## Risk Profiles

```
Conservative: 5% per trade, 5% stop loss, 60% confidence required
Moderate:     10% per trade, 8% stop loss, 55% confidence required  
Aggressive:   20% per trade, 10% stop loss, 50% confidence required
```

Choose at Step 5 (Trading Signals).

---

## Troubleshooting

### "Module not found"
```bash
pip install pandas numpy requests transformers torch yfinance scikit-learn
```

### "API Error"
- API keys are hardcoded in `sentiment_analyzer.py`
- Check internet connection
- Verify Finnhub/GNews APIs are working

### "No price data"
- yfinance might be blocked
- Try: `pip install --upgrade yfinance`

### "Insufficient data"
- Need at least 20 days of price history
- Check symbol spelling
- Some symbols may not have data available

---

## Next Steps

1. **Paper Trade**: Test signals with play money first
2. **Backtest**: Validate on historical data
3. **Monitor**: Check predictions against actual prices
4. **Refine**: Adjust risk parameters based on results
5. **Deploy**: Only trade live after validation

---

## Files & Documentation

- **README_COMPLETE_SYSTEM.md** — Detailed system guide
- **TECHNICAL_INDICATORS_GUIDE.md** — Code explanations
- **sentiment_analyzer.py** — News sentiment (Step 1)
- **technical_indicators.py** — Technical analysis (Step 2)
- **market_regime_model.py** — Regime classification (Step 3)
- **price_forecaster.py** — Price prediction (Step 4)
- **rl_trader.py** — Trading signals (Step 5)
- **master_pipeline.py** — Orchestrator

---

## Example Trading Workflow

```
Morning: Run sentiment_analyzer.py
         → Check market sentiment overnight

9:30am:  Run technical_indicators.py
         → Analyze technical setup for stocks

10am:    Run market_regime_model.py
         → Classify current market regime

10:30am: Run price_forecaster.py
         → Get price targets for day

11am:    Run rl_trader.py
         → Execute trading decisions based on signals

Throughout day: Monitor trading_signals_*.csv
                Update positions as new data arrives
                Stick to stop loss/take profit levels
```

---

## Key Metrics Cheat Sheet

| Metric | Meaning | Good Signal |
|--------|---------|-------------|
| Sentiment Mean | News tone (-1 to +1) | > 0 (Positive) |
| RSI (14) | Momentum (0-100) | 40-60 (Balanced) |
| MACD | Trend change | Positive cross |
| Signal | Trading recommendation | BUY/STRONG_BUY |
| Confidence | Prediction reliability | > 70% |
| Expected Value | Risk-adjusted return | > +0.5% |

---

## Important ⚠️

✓ **This is educational only** - Not financial advice
✓ **Backtest thoroughly** before live trading
✓ **Start small** with paper trading
✓ **Use proper risk management** always
✓ **Monitor constantly** - Don't automate blindly

---

**Ready to trade? Run:** `python3 master_pipeline.py --auto`

**Questions? Check:** `README_COMPLETE_SYSTEM.md`
