# 🚀 AI-Driven Trading System - Complete Pipeline

A multi-stage AI system that analyzes financial news sentiment, calculates technical indicators, predicts market regimes, forecasts prices, and generates trading signals.

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FINANCIAL NEWS & MARKET DATA                     │
└────┬────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: SENTIMENT ANALYSIS (FinBERT)                               │
│ • Fetch news headlines from Finnhub API                            │
│ • Analyze sentiment using transformer model                        │
│ • Output: Sentiment scores, trends, daily breakdowns               │
└────┬────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: TECHNICAL INDICATORS                                        │
│ • Calculate SMA, EMA, RSI, MACD, Bollinger Bands                   │
│ • Score price trends and momentum                                  │
│ • Output: Technical scores, regime prediction                      │
└────┬────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: MARKET REGIME CLASSIFICATION                               │
│ • Combine sentiment + technical indicators                         │
│ • Classify regime: BULL / BEAR / VOLATILE                          │
│ • Output: Regime probabilities, confidence scores                  │
└────┬────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: PRICE FORECASTING                                          │
│ • Train ML models on historical data                               │
│ • Project prices for 5, 10, 15, 20 days ahead                     │
│ • Output: Price predictions, confidence intervals                  │
└────┬────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: TRADING SIGNAL GENERATION                                  │
│ • Combine all upstream signals                                     │
│ • Generate BUY/SELL/HOLD decisions                                │
│ • Risk management: Position sizing, stop loss, take profit         │
│ • Output: Trading signals, portfolio recommendations               │
└────┬────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  TRADING DECISIONS & PORTFOLIO VALUE                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Analyzed Assets

### Indices (4)
- **S&P 500** (^GSPC) - US Stock Market
- **NASDAQ Composite** (^IXIC) - US Tech Stocks
- **Euro Stoxx 50** (^STOXX50E) - European Stocks
- **Nikkei 225** (^N225) - Japanese Stocks

### Cryptocurrencies (3)
- **Bitcoin** (BTC-USD)
- **Ethereum** (ETH-USD)
- **Solana** (SOL-USD)

### Foreign Exchange (2)
- **EUR/USD** Exchange Rate
- **GBP/USD** Exchange Rate

### Commodities (2)
- **Gold** (GC=F)
- **Crude Oil WTI** (CL=F)

---

## 📦 Installation

### Requirements
```bash
Python 3.8+
pip install pandas numpy requests transformers torch yfinance scikit-learn
```

### Optional (for advanced features)
```bash
pip install tensorflow keras  # For LSTM-based forecasting
pip install plotly            # For visualizations
```

### API Keys Required
1. **Finnhub**: Get key at https://finnhub.io/
2. **GNews**: Get key at https://gnews.io/

API keys are hardcoded in `sentiment_analyzer.py` for this demo.

---

## 🚀 Quick Start

### Option 1: Run Complete Pipeline
```bash
# Interactive mode (choose what to run)
python3 master_pipeline.py --full

# Auto mode (runs everything with defaults)
python3 master_pipeline.py --auto
```

### Option 2: Run Individual Steps
```bash
# Step 1: Sentiment Analysis
python3 sentiment_analyzer.py

# Step 2: Technical Indicators (select symbols)
python3 technical_indicators.py
# Enter: 1  (for S&P 500) or 'all' for all assets

# Step 3: Market Regime Classification
python3 market_regime_model.py

# Step 4: Price Forecasting
python3 price_forecaster.py

# Step 5: Trading Signals (choose risk profile)
python3 rl_trader.py
# Select: 2 (Moderate) or 1 (Conservative) or 3 (Aggressive)
```

### Option 3: Check System Status
```bash
python3 master_pipeline.py --check
python3 master_pipeline.py --report
```

---

## 📊 Output Files

All outputs are saved to `data/output/` with timestamps.

### Sentiment Analysis
```
latest.json                           # Latest sentiment analysis
sentiment_20260308_134638.csv        # CSV summary
sentiment_20260308_134638.json       # Full JSON results
```

**Format:**
```json
{
  "timestamp": "2026-03-08T13:46:38",
  "window": "1mo",
  "assets": [
    {
      "symbol": "^GSPC",
      "name": "S&P 500",
      "overall_mean": 0.082,
      "overall_median": 0.046,
      "overall_std": 0.093,
      "total_headlines": 42
    }
  ]
}
```

### Technical Indicators & Regime
```
regime_prediction_20260308_134638.csv # Regime predictions
```

**CSV Columns:**
```
Asset, Symbol, Days, Headlines, Mean, Median, Std, Signal
S&P 500, ^GSPC, 4, 42, 0.082, 0.046, 0.093, SELL
```

### Market Regime Classification
```
market_regime_20260308_134638.csv    # Regime classifications
market_regime_20260308_134638.json   # Full analysis
```

### Price Forecasts
```
price_forecast_20260308_134638.csv   # Price predictions
price_forecast_20260308_134638.json  # Full forecasts
```

**CSV Columns:**
```
Symbol, Current_Price, Days_Ahead, Target_Date, Predicted_Price, Price_Range, Expected_Return, Confidence
^GSPC, $5834.50, 5, 2026-03-13, $5900.00, $5800.00-$6000.00, +1.12%, 85%
```

### Trading Signals
```
trading_signals_20260308_134638.csv  # Trading recommendations
trading_signals_20260308_134638.json # Signal details
```

**CSV Columns:**
```
Symbol, Signal, Strength, Confidence, Current_Price, Predicted_Price, Expected_Return, Position_Size, Stop_Loss, Take_Profit, Risk_Reward_Ratio, Expected_Value
^GSPC, BUY, +0.62, 85%, $5834.50, $5900.00, +1.12%, 10.0%, $5543.78, $6417.95, 2.15:1, +0.73%
```

---

## 🤖 Risk Profiles

### Conservative
- **Position Size:** 5% per trade
- **Stop Loss:** 5%
- **Take Profit:** 10%
- **Min Confidence:** 60%
- **Max Leverage:** None

**Best for:** Preservation of capital, long-term investors

### Moderate
- **Position Size:** 10% per trade
- **Stop Loss:** 8%
- **Take Profit:** 15%
- **Min Confidence:** 55%
- **Max Leverage:** 2x

**Best for:** Balanced risk/reward, most traders

### Aggressive
- **Position Size:** 20% per trade
- **Stop Loss:** 10%
- **Take Profit:** 25%
- **Min Confidence:** 50%
- **Max Leverage:** 3x

**Best for:** Growth-focused, experienced traders

---

## 📈 Key Metrics Explained

### Sentiment Score (-1 to +1)
- **+1.0** = Extremely positive
- **0.0** = Neutral
- **-1.0** = Extremely negative

### Technical Score (-10 to +10)
- **+10** = Strongest bullish signal
- **0** = Neutral
- **-10** = Strongest bearish signal

### Regime Types
- **🔼 BULL** (score ≥ 3.0) - Uptrend expected
- **↔️ SIDEWAYS** (-3.0 < score < 3.0) - No clear direction
- **🔽 BEAR** (score ≤ -3.0) - Downtrend expected
- **🌪️ VOLATILE** - High price swings expected

### Confidence (0% - 100%)
Higher confidence = more reliable prediction

### Expected Value (EV)
Probability-weighted return accounting for risk:
- **EV = (Win% × Return) - (Loss% × Loss)**
- Positive EV = Favorable risk/reward trade
- Negative EV = Unfavorable, avoid

---

## 🔄 Data Flow Example

```
┌──────────────────────────────────┐
│ Financial News: "S&P drops 2% on│
│ recession fears and job losses"  │
└──────────────┬───────────────────┘
               ▼
        FinBERT Sentiment
        ↓
        Score: -0.35 (Negative)
        
┌──────────────────────────────────┐
│ Price Data: Close=$5840, SMA20=  │
│ $5820, RSI=42, MACD=-0.012       │
└──────────────┬───────────────────┘
               ▼
        Technical Scoring
        ↓
        Price below SMA: -1.0
        RSI weak: -0.5
        MACD bearish: -1.0
        Engineering Score: -2.5
        
        Combined Regime: BEAR
        ↓
        Forecast: $5700-5850 in 5 days
        Probability: 65% downside
        
        Trading Signal: SELL
        confidence: 75%
```

---

## 💡 Usage Examples

### Example 1: Quick Market Check
```bash
# Run just sentiment analysis
python3 sentiment_analyzer.py

# Check what the market thinks about your portfolio
```

### Example 2: Technical Analysis
```bash
# Analyze S&P 500 technical setup
python3 technical_indicators.py
# Enter: 1 (S&P 500 only)
```

### Example 3: Complete Trading Decision
```bash
# Full pipeline for trading decisions
python3 master_pipeline.py --auto

# Get BUY/SELL signals for all 11 assets
```

### Example 4: Conservative Portfolio
```bash
python3 rl_trader.py
# Select: 1 (Conservative)
# Enter: 50000 (portfolio value)
# Get position sizes limited to 5% risk
```

---

## ⚠️ Important Notes

### Disclaimer
This system is for **educational and research purposes** only. It should **NOT** be used for real trading without:
1. Extensive backtesting on historical data
2. Paper trading validation
3. Professional financial advice
4. Risk management protocols

### Limitations
- **Past performance ≠ Future results**
- **Sentiment analysis has inherent bias** from news sources
- **Technical indicators lag price action**
- **Market regimes can change rapidly**
- **Black swan events are unpredictable**

### Best Practices
1. **Verify signals independently** before trading
2. **Use proper risk management** (stop losses, position sizing)
3. **Diversify** across multiple assets
4. **Monitor constantly** - don't set and forget
5. **Paper trade first** to validate strategy

---

## 🧪 Testing the System

### Validate Installation
```bash
python3 master_pipeline.py --check
```

### Run Sentiment Analysis Only
```bash
python3 sentiment_analyzer.py
# Takes 2-3 minutes to download news and analyze sentiment
# Check: data/output/latest.json
```

### Test Technical Analysis
```bash
python3 technical_indicators.py
# Enter: 1 (test with S&P 500)
# Check output and regime classification
```

---

## 🔧 Customization

### Change Analyzed Assets
Edit `sentiment_analyzer.py` - modify `ALL_ASSETS` list:
```python
ALL_ASSETS = [
    {"index": 1, "symbol": "YOUR_SYMBOL", "name": "Name", ...},
]
```

### Adjust Risk Parameters
Edit `rl_trader.py` - modify `RISK_PARAMS` dictionary:
```python
RISK_PARAMS = {
    RiskProfile.CONSERVATIVE: {
        'max_position_size': 0.05,  # Change to 0.10 for 10%
        'stop_loss': 0.05,          # Change to 0.08 for 8%
    }
}
```

### Change Prediction Horizons
Edit `price_forecaster.py`:
```python
PREDICTION_HORIZONS = [5, 10, 15, 20]  # Change to [1, 3, 7, 30]
```

---

## 📚 File Structure

```
Project-for-PAI/
├── sentiment_analyzer.py          # Step 1: News sentiment
├── technical_indicators.py        # Step 2: Technical analysis
├── market_regime_model.py         # Step 3: Regime classification
├── price_forecaster.py            # Step 4: Price prediction
├── rl_trader.py                   # Step 5: Trading signals
├── master_pipeline.py             # Orchestrator
├── data/
│   └── output/
│       ├── latest.json
│       ├── sentiment_*.csv/.json
│       ├── market_regime_*.csv/.json
│       ├── price_forecast_*.csv/.json
│       └── trading_signals_*.csv/.json
├── TECHNICAL_INDICATORS_GUIDE.md  # Detailed code documentation
└── README.md                       # This file
```

---

## 🤝 Support

For issues or questions:
1. Check the `TECHNICAL_INDICATORS_GUIDE.md` for code explanations
2. Review error messages - they often indicate missing packages
3. Ensure all API keys are valid
4. Check internet connection for data fetching

---

## 📝 License & Disclaimer

This project is provided as-is for educational purposes. Users are responsible for validating its predictions and following proper risk management practices.

**Not financial advice. Always consult a professional.**

---

## 🎓 Learning Resources

- **Sentiment Analysis**: FinBERT transformer documentation
- **Technical Indicators**: Investopedia (RSI, MACD, Bollinger Bands)
- **Risk Management**: Position sizing and portfolio theory
- **Machine Learning**: scikit-learn and TensorFlow documentation

---

**Built with ❤️ for AI-driven trading | Last Updated: 2026-03-08**
