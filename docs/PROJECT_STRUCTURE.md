# Project Structure Documentation

## Overview

The Global Asset Sentiment Analyzer & Trading System follows a professional Python project structure with clear separation of concerns:

```
Project-for-PAI/
├── src/                           # Main source code package
├── scripts/                       # Executable scripts and pipelines  
├── data/                          # Data storage (cache & output)
├── docs/                          # Documentation
├── .gitignore                     # Git ignore configuration
├── README.md                      # Main project documentation
└── requirements.txt               # Python dependencies
```

## Directory Details

### `src/` - Core Source Code

The main Python package containing all analysis and trading modules:

```
src/
├── __init__.py                    # Package initialization with exports
├── sentiment_analyzer.py          # Financial news sentiment analysis
├── technical_indicators.py        # Technical indicator calculations
├── market_regime_model.py         # Market regime classification
├── price_forecaster.py            # ML-based price prediction
├── rl_trader.py                   # Trading signal generation
└── utils/                         # Utility modules
    ├── __init__.py
    ├── config_manager.py          # Configuration management
    └── sentiment_cache.py         # Sentiment data caching
```

#### Module Responsibilities

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `sentiment_analyzer.py` | News sentiment analysis | Fetch headlines, FinBERT analysis, daily/overall sentiment |
| `technical_indicators.py` | Technical analysis | RSI, MACD, Bollinger Bands, momentum, trend indicators |
| `market_regime_model.py` | Market classification | Bull/Bear/Volatile regime detection |
| `price_forecaster.py` | Price prediction | ML forecasting for 5D/10D/15D/20D horizons |
| `rl_trader.py` | Trading signals | Multi-factor signal generation, position sizing |
| `config_manager.py` | Configuration | Pipeline configuration management |
| `sentiment_cache.py` | Caching | Sentiment data persistence |

### `scripts/` - Pipeline Orchestration

Executable scripts that run the complete trading pipeline:

```
scripts/
├── master_pipeline.py             # Basic pipeline orchestrator
└── master_pipeline_v2.py          # Advanced version with logging
```

- **Used by:** End users running the complete analysis pipeline
- **Function:** Coordinates execution of all src/ modules in sequence
- **Import paths:** Reference `src/` modules using relative imports

### `data/` - Data Storage

Persistent data directory with two subdirectories:

```
data/
├── cache/                         # Cached data for performance
│   ├── .gitkeep                   # Ensures directory tracking
│   └── sentiment_index.json       # Cached sentiment results
└── output/                        # Pipeline outputs
    ├── .gitkeep                   # Ensures directory tracking
    ├── sentiment_*.json/.csv      # Sentiment analysis results
    ├── technical_*.json           # Technical indicator results
    ├── market_regime_*.json/.csv  # Market classification
    ├── price_forecast_*.json/.csv # Price predictions
    └── trading_signals_*.json/.csv# Trading signals
```

**Note:** Output files are timestamped to prevent conflicts (e.g., `sentiment_20260308_134638.csv`)

### `docs/` - Documentation

Complete project documentation:

```
docs/
├── README.md                      # Full system documentation
├── QUICK_START.md                 # Quick start guide
├── QUICK_START_V2.md              # V2 quick start
├── TECHNICAL_INDICATORS_GUIDE.md  # Technical reference
├── SYSTEM_COMPLETION_SUMMARY.md   # Project completion report
├── CHANGELOG.md                   # Version history
└── Global_Asset_Sentiment_Analyzer_Technical_Documentation.docx
```

## Import Structure

### From Scripts
Pipeline scripts use relative imports to access src modules:

```python
# From scripts/master_pipeline_v2.py
from src.utils.config_manager import ConfigManager
from src.sentiment_analyzer import SentimentAnalyzer
```

### From Within src/
Modules within src/ can import from other modules:

```python
# Within src/ modules
from .sentiment_analyzer import SentimentAnalyzer
from .utils.config_manager import ConfigManager
```

### Package Exports
The `src/__init__.py` exports key classes for convenient importing:

```python
from src import (
    SentimentAnalyzer,
    TechnicalIndicators,
    MarketRegimeClassifier,
    PriceForecaster,
    TradeSignalGenerator,
)
```

## Data Flow

```
Input (User selects assets & time window)
    │
    ├──→ scripts/master_pipeline_v2.py
    │
    ├──→ src/sentiment_analyzer.py
    │    └──→ data/output/sentiment_*.json
    │
    ├──→ src/technical_indicators.py
    │    └──→ data/output/technical_*.json
    │
    ├──→ src/market_regime_model.py
    │    └──→ data/output/market_regime_*.json
    │
    ├──→ src/price_forecaster.py
    │    └──→ data/output/price_forecast_*.json
    │
    └──→ src/rl_trader.py
         └──→ data/output/trading_signals_*.json
```

## Version Control

### Git Configuration

**`.gitignore`** excludes:
- Python cache (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `.env`)
- IDE files (`.vscode/`, `.idea/`)
- Generated output data (`data/output/*` except `.gitkeep`)
- Cache files (`data/cache/*` except `.gitkeep`)

**`.gitkeep`** files maintain empty directory structure.

## Dependencies

All required packages are listed in `requirements.txt`:

```
Core Libraries:
- numpy, pandas, scikit-learn
- transformers (FinBERT)
- torch
- yfinance, newsapi
- (See requirements.txt for complete list)
```

Install with:
```bash
pip install -r requirements.txt
```

## Execution Patterns

### Pattern 1: Run Full Pipeline
```bash
python scripts/master_pipeline_v2.py
```
- Orchestrates all 5 stages
- Collects user input once
- Passes config to all modules

### Pattern 2: Run Individual Module
```bash
python src/sentiment_analyzer.py
```
- Can be run standalone
- Useful for testing/debugging
- Generates stage-specific output

### Pattern 3: Import in Custom Script
```python
from src import TradeSignalGenerator
trader = TradeSignalGenerator()
signals = trader.generate_signal(prediction, regime)
```

## Extensibility Guidelines

### Adding a New Module

1. Create `src/new_module.py`
2. Add to `src/__init__.py` exports
3. Update pipeline steps in scripts
4. Create documentation in `docs/`

### Adding a Utility

1. Create `src/utils/new_utility.py`
2. Add to `src/utils/__init__.py`
3. Import in modules using `from .utils.new_utility import ...`

### Adding Documentation

1. Create `docs/NEW_GUIDE.md`
2. Link from main `docs/README.md`
3. Reference in root `README.md`

## Best Practices

✅ **Do:**
- Keep modules focused on single responsibility
- Use type hints in function signatures
- Add docstrings to all classes and functions
- Store configuration in `config_manager.py`
- Cache expensive computations in `data/cache/`
- Log outputs to `data/output/` with timestamps

❌ **Don't:**
- Hard-code paths (use `os.path.join` or `Path`)
- Store binary files in git
- Mix data processing with presentation
- Create global state in modules
- Ignore error handling

## Testing Imports

Verify structure is correct:

```bash
# Test imports from project root
python3 -c "from src import TradeSignalGenerator; print('✅ OK')"

# Test pipeline execution
python3 scripts/master_pipeline_v2.py
```

## Performance Considerations

- **Cache:** Sentiment results cached in `data/cache/` to avoid re-fetching
- **Output:** All results saved to `data/output/` for audit trail
- **Modular:** Each stage can be run independently
- **Scalable:** Directory structure supports adding new assets/indicators

---

**Last Updated:** March 8, 2026  
**Status:** Production Ready
