"""
Global Asset Sentiment Analyzer and Trading System
Main module containing core analysis and trading components.
"""

__version__ = "2.0.0"
__author__ = "Programming for AI Course"

# Import core modules with graceful degradation (silent on error)
try:
    from .sentiment_analyzer import SentimentAnalyzer
except (ImportError, Exception):
    SentimentAnalyzer = None

try:
    from .technical_indicators import TechnicalIndicators
except (ImportError, Exception):
    TechnicalIndicators = None

try:
    from .market_regime_model import MarketRegimeClassifier
except (ImportError, Exception):
    MarketRegimeClassifier = None

try:
    from .price_forecaster import PriceForecaster
except (ImportError, Exception):
    PriceForecaster = None

try:
    from .rl_trader import TradeSignalGenerator
except (ImportError, Exception):
    TradeSignalGenerator = None

__all__ = [
    'SentimentAnalyzer',
    'TechnicalIndicators',
    'MarketRegimeClassifier',
    'PriceForecaster',
    'TradeSignalGenerator',
]
