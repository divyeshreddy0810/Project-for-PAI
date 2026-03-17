# Global Asset Sentiment Analyzer & Trading System

An AI-driven trading system that combines sentiment analysis, technical indicators, market regime classification, price forecasting, and reinforcement learning-based trading signals.

**Course:** Programming for AI (NCI)  
**Status:** Production Ready

---

## 📁 Project Structure

This repository contains the core logic and scripts to run the trading pipeline. (Note: Output and cache data files are ignored to keep the repository clean).

```text
Project-for-PAI/
├── .gitignore                          # Rules for ignoring output/cache files
├── README.md                           # This file
├── MIGRATION_SUMMARY.md                # Context for recent changes
├── QUICK_START_1M_DATA.md              # Setup guide and performance tips
├── TRAINING_DATA_SCALABILITY.md        # Technical explanation context
├── analyze_dataset_sizes.py            # Utility script for dataset size analysis
│
├── data/
│   ├── cache/
│   │   └── .gitkeep                    # Preserves empty cache directory in Git
│   └── output/
│       └── .gitkeep                    # Preserves empty output directory in Git
│
├── docs/                               # Comprehensive Documentation
│   ├── CHANGELOG.md                    
│   ├── Global_Asset_Sentiment_Analyzer_Technical_Documentation.docx
│   ├── PROJECT_STRUCTURE.md            
│   ├── QUICK_START.md                  
│   ├── QUICK_START_V2.md               
│   ├── SYSTEM_COMPLETION_SUMMARY.md    
│   └── TECHNICAL_INDICATORS_GUIDE.md   
│
├── scripts/
│   └── master_pipeline_v2.py           # Main orchestration script
│
└── src/                                # Core source code
    ├── __init__.py                     
    ├── market_regime_model.py          # Bull/Bear market classification
    ├── price_forecaster.py             # ML-based price prediction
    ├── rl_trader.py                    # Trading signal generation
    ├── sentiment_analyzer.py           # Financial news sentiment analysis (FinBERT)
    ├── technical_indicators.py         # Technical indicator calculations
    └── utils/
        ├── __init__.py                 
        ├── config_manager.py           # Configuration management
        └── sentiment_cache.py          # Sentiment data caching
```

## 🚀 Quick Start & How to Run

### Prerequisites
- Python 3.8+
- Internet connection (for fetching live ticker data and news)

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd Project-for-PAI
pip install -r requirements.txt
```

*(If you do not have a `requirements.txt` file yet, you can generate one locally via `pip freeze > requirements.txt` before pushing to GitHub).*

### 2. Running the Pipeline

The entire system is orchestrated by a single master pipeline script that runs Sentiment Analysis, Technical Indicators, Market Regime Classification, Price Forecasting, and Trading Signal Generation consecutively.

To run the pipeline with default assets and settings:
```bash
python scripts/master_pipeline_v2.py
```

All generated reports, CSVs, and JSON files will be exported automatically to the `data/output/` directory.

---

## 📊 Pipeline Stages

The system executes the following phases in order:

1. **Sentiment Analysis** 📰 (`src/sentiment_analyzer.py`): Analyzes financial news headlines using FinBERT to generate sentiment scores for selected assets.
2. **Technical Indicators** 📈 (`src/technical_indicators.py`): Calculates momentum, trend, and volatility indicators (RSI, MACD, Bollinger Bands, ATR, etc).
3. **Market Regime Classification** 🎯 (`src/market_regime_model.py`): Classifies market conditions (Bull/Bear/Volatile) to adjust strategies.
4. **Price Forecasting** 💰 (`src/price_forecaster.py`): Uses Machine Learning to predict future prices (5-day, 10-day, 15-day, and 20-day horizons).
5. **Trading Signal Generation** 🤖 (`src/rl_trader.py`): Generates buy/sell/hold signals based on multi-factor analysis, position sizing, and risk management.

---

## 📋 Supported Assets

By default, the pipeline can analyze:
*   **Equities:** S&P 500 (^GSPC), NASDAQ (^IXIC), AAPL, MSFT, GOOGL
*   **International Indices:** Euro Stoxx 50 (^STOXX50E), Nikkei 225 (^N225)
*   **Crypto:** Bitcoin (BTC-USD), Ethereum (ETH-USD), Solana (SOL-USD)
*   **Forex:** EUR/USD (EURUSD=X), GBP/USD (GBPUSD=X)
*   **Commodities:** Gold (GC=F), Crude Oil (CL=F)

---

## 🔐 Risk Management

The system includes built-in risk management profiles (Conservative, Moderate, Aggressive) handling:
- **Position Sizing:** Risk-adjusted portfolio allocation.
- **Stop Loss:** Automated loss prevention (5-10% based on profile).
- **Take Profit:** Automated profit-taking levels.
- **Confidence Thresholds:** Minimum system confidence requirements before issuing a `BUY` trade.

---

## 📚 Further Reading

For deep-dives into specific functionality, refer to the `docs/` folder:
- **[Full Context & Architecture](docs/PROJECT_STRUCTURE.md)**
- **[Technical Indicators Reference](docs/TECHNICAL_INDICATORS_GUIDE.md)**
- **[System Completion Summary](docs/SYSTEM_COMPLETION_SUMMARY.md)**
