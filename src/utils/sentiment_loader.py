"""
SentimentLoader
---------------
Reads sentiment_analyzer.py output (data/output/latest.json) and
returns normalised sentiment features per symbol for use in backtest.py.

Scale conversion:
    sentiment_analyzer outputs positive_probability in [0, 1]
    backtest features expect sentiment_mean in [-1, 1]
    Formula: sentiment_mean = (positive_prob - 0.5) * 2

Usage:
    from src.utils.sentiment_loader import load_sentiment

    sent = load_sentiment('^GSPC')
    # Returns: {'sentiment_mean': 0.12, 'sentiment_trend': 0.003,
    #           'headline_count': 47, 'source': 'latest.json'}
"""

import os
import json
import numpy as np
from typing import Dict, Optional

DEFAULT_SENTIMENT_FILE = "data/output/latest.json"


def _positive_prob_to_score(positive_prob: float) -> float:
    """Convert positive probability [0,1] to sentiment score [-1,1]."""
    return float((positive_prob - 0.5) * 2)


def load_sentiment(symbol: str,
                   filepath: str = DEFAULT_SENTIMENT_FILE) -> Dict[str, float]:
    """
    Load and normalise sentiment for a symbol from latest.json.

    Args:
        symbol  : Asset symbol e.g. '^GSPC', 'BTC-USD'
        filepath: Path to sentiment JSON. Default data/output/latest.json.

    Returns:
        Dict with keys:
            sentiment_mean   : float in [-1, 1]
            sentiment_trend  : float (slope of daily sentiment)
            headline_count   : int
            source           : str ('latest.json' or 'stub')
    """
    if not os.path.exists(filepath):
        return _stub_sentiment(symbol, reason='file_not_found')

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"   ⚠️  SentimentLoader: could not read {filepath}: {e}")
        return _stub_sentiment(symbol, reason='read_error')

    # Find matching asset
    assets = data.get('assets', [])
    asset_data = None
    for a in assets:
        if a.get('symbol', '').upper() == symbol.upper():
            asset_data = a
            break

    if asset_data is None:
        print(f"   ⚠️  SentimentLoader: '{symbol}' not found in {filepath} — using stub")
        return _stub_sentiment(symbol, reason='symbol_not_found')

    # overall_mean is positive probability [0,1] → convert to [-1,1]
    overall_mean_raw = asset_data.get('overall_mean', 0.5)
    if overall_mean_raw is None:
        overall_mean_raw = 0.5
    sentiment_mean = _positive_prob_to_score(float(overall_mean_raw))

    # Compute sentiment trend from daily_means
    daily_means = asset_data.get('daily_means', [])
    sentiment_trend = _compute_trend(daily_means)

    headline_count = int(asset_data.get('total_headlines', 0))

    print(f"   ✅ SentimentLoader: {symbol} → "
          f"mean={sentiment_mean:+.3f}  trend={sentiment_trend:+.4f}  "
          f"headlines={headline_count}  "
          f"(raw positive_prob={overall_mean_raw:.3f})")

    return {
        'sentiment_mean':  sentiment_mean,
        'sentiment_trend': sentiment_trend,
        'headline_count':  headline_count,
        'source':          filepath
    }


def _compute_trend(daily_means: list) -> float:
    """Linear regression slope over daily mean sentiment values."""
    if len(daily_means) < 2:
        return 0.0
    scores = [_positive_prob_to_score(d.get('mean_sentiment', 0.5))
              for d in daily_means]
    x = np.arange(len(scores), dtype=float)
    try:
        slope = float(np.polyfit(x, scores, 1)[0])
    except Exception:
        slope = 0.0
    return slope


def _stub_sentiment(symbol: str, reason: str = '') -> Dict[str, float]:
    """Return neutral stub sentiment when real data is unavailable."""
    return {
        'sentiment_mean':  0.05,
        'sentiment_trend': 0.0,
        'headline_count':  0,
        'source':          f'stub ({reason})'
    }


def load_all_sentiments(filepath: str = DEFAULT_SENTIMENT_FILE) -> Dict[str, Dict]:
    """
    Load sentiment for every symbol in the JSON file.

    Returns:
        Dict keyed by symbol: {'^GSPC': {sentiment_mean: ..., ...}, ...}
    """
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception:
        return {}

    result = {}
    for a in data.get('assets', []):
        sym = a.get('symbol', '')
        if sym:
            result[sym] = load_sentiment(sym, filepath)
    return result
