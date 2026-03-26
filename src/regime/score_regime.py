"""
ScoreRegimeDetector
-------------------
Wraps the existing score-based RegimePredictor from technical_indicators.py.
Used as the BASELINE in comparative experiments.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict
from src.regime.base import RegimeDetector


class ScoreRegimeDetector(RegimeDetector):
    """
    Baseline regime detector using weighted composite score
    (price trend + RSI + MACD + sentiment) with fixed thresholds.
    Reproduces the logic from RegimePredictor.score_regime().
    """

    SCORE_WEIGHTS = {
        'price_trend':      1.5,
        'rsi':              1.0,
        'macd':             1.0,
        'sentiment_level':  1.5,
        'sentiment_trend':  1.0,
        'headline_volume':  0.5,
    }

    def __init__(self):
        self.fitted_ = False
        self._last_score = 0.0

    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame) -> None:
        """
        No training required for score-based detector.
        Just marks the detector as ready.

        Args:
            data: DataFrame with columns Close, RSI, MACD, MACD_Signal,
                  SMA_20, SMA_50, SMA_200, sentiment_mean,
                  sentiment_trend, headline_count
        """
        required = ['Close', 'RSI', 'MACD', 'MACD_Signal',
                    'SMA_20', 'sentiment_mean']
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"ScoreRegimeDetector.fit(): missing columns {missing}")
        self.fitted_ = True

    # ------------------------------------------------------------------
    def predict(self, data: pd.DataFrame) -> str:
        """
        Score the latest row and return regime label.

        Returns:
            'bull', 'bear', or 'sideways'
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")

        row = data.iloc[-1]

        def g(col, default=0.0):
            v = row.get(col, default)
            return float(v) if pd.notna(v) else default

        close   = g('Close')
        sma_20  = g('SMA_20',  close)
        sma_50  = g('SMA_50',  close)
        sma_200 = g('SMA_200', close)
        rsi     = g('RSI', 50.0)
        macd    = g('MACD', 0.0)
        signal  = g('MACD_Signal', 0.0)
        sent    = g('sentiment_mean', 0.0)
        s_trend = g('sentiment_trend', 0.0)
        n_head  = g('headline_count', 0.0)

        score = 0.0

        # 1. Price trend
        if close > sma_20 and sma_20 > sma_50:
            t = 3.0
        elif close > sma_50 and sma_50 > sma_200:
            t = 2.0
        elif close > sma_20:
            t = 1.0
        elif close < sma_20 and sma_20 < sma_50:
            t = -3.0
        elif close < sma_50 and sma_50 < sma_200:
            t = -2.0
        elif close < sma_20:
            t = -1.0
        else:
            t = 0.0
        score += t * self.SCORE_WEIGHTS['price_trend']

        # 2. RSI
        if rsi > 70:
            r = -2.0
        elif rsi > 60:
            r = 1.0
        elif rsi > 40:
            r = 0.5
        elif rsi > 30:
            r = -0.5
        else:
            r = 2.0
        score += r * self.SCORE_WEIGHTS['rsi']

        # 3. MACD
        if macd > signal and macd > 0:
            m = 2.0
        elif macd > signal:
            m = 1.0
        elif macd < signal and macd < 0:
            m = -2.0
        else:
            m = -1.0
        score += m * self.SCORE_WEIGHTS['macd']

        # 4. Sentiment level
        if sent > 0.15:
            sl = 2.0
        elif sent > 0.05:
            sl = 1.0
        elif sent > -0.05:
            sl = 0.0
        elif sent > -0.15:
            sl = -1.0
        else:
            sl = -2.0
        score += sl * self.SCORE_WEIGHTS['sentiment_level']

        # 5. Sentiment trend
        if s_trend > 0.05:
            st = 1.5
        elif s_trend > 0:
            st = 0.5
        elif s_trend > -0.05:
            st = 0.0
        else:
            st = -1.0
        score += st * self.SCORE_WEIGHTS['sentiment_trend']

        # 6. Headline volume
        hv = 1.0 if n_head > 50 else 0.5 if n_head > 20 else -0.5
        score += hv * self.SCORE_WEIGHTS['headline_volume']

        self._last_score = score

        if score >= 3.0:
            return 'bull'
        elif score <= -3.0:
            return 'bear'
        else:
            return 'sideways'

    # ------------------------------------------------------------------
    def predict_series(self, data: pd.DataFrame) -> pd.Series:
        """
        Return a regime label for every row (rolling single-row predict).
        Useful for walk-forward windows.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict_series().")
        labels = []
        for i in range(len(data)):
            labels.append(self.predict(data.iloc[:i+1]))
        return pd.Series(labels, index=data.index, name='regime')
