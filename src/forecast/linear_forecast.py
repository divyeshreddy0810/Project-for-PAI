"""
LinearForecaster
----------------
Wraps the existing momentum + linear regression logic from
PriceForecaster (price_forecaster.py). Used as the BASELINE
forecaster in comparative experiments.

Inputs:
    X : (n_samples, n_features) — engineered feature matrix
    y : (n_samples,)            — Close prices (targets)

predict() returns a 1-D array of predicted prices for the
next [5, 10, 15, 20] days, given the last row of X.
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.forecast.base import Forecaster

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


PREDICTION_HORIZONS = [5, 10, 15, 20]


class LinearForecaster(Forecaster):
    """
    Baseline forecaster: linear regression on flattened lookback
    sequences, with momentum extrapolation for multi-step horizons.

    Args:
        lookback   : Number of past bars used as features. Default 20.
        horizons   : List of forward periods to predict. Default [5,10,15,20].
    """

    def __init__(self, lookback: int = 20,
                 horizons: list = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required. Install with: pip install scikit-learn"
            )
        self.lookback  = lookback
        self.horizons  = horizons or PREDICTION_HORIZONS
        self.model_    = LinearRegression()
        self.scaler_   = MinMaxScaler()
        self.fitted_   = False
        self._last_close: float = 0.0
        self._last_trend: float = 0.0
        self._last_vol:   float = 0.01

    # ------------------------------------------------------------------
    @staticmethod
    def _feature_cols() -> list:
        return ['Returns', 'Price_MA5', 'Price_MA20', 'Price_MA50',
                'Volatility', 'RSI', 'MACD', 'Signal',
                'Volume_Trend', 'High_Low_Range', 'Open_Close_Range']

    # ------------------------------------------------------------------
    def _to_sequences(self, X_scaled: np.ndarray,
                      y: np.ndarray):
        """Build (samples, lookback*features) and matching targets."""
        seqs, targets = [], []
        for i in range(len(X_scaled) - self.lookback):
            seqs.append(X_scaled[i:i + self.lookback].flatten())
            targets.append(y[i + self.lookback])
        return np.array(seqs), np.array(targets)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit scaler + linear regression on training data.

        Args:
            X : (n_samples, n_features) float array of engineered features.
            y : (n_samples,) Close prices.
        """
        if len(X) < self.lookback + 1:
            raise ValueError(
                f"LinearForecaster.fit(): need >{self.lookback} rows, "
                f"got {len(X)}"
            )

        X_scaled = self.scaler_.fit_transform(X)
        X_seq, y_seq = self._to_sequences(X_scaled, y)

        self.model_.fit(X_seq, y_seq)
        self.fitted_ = True

        r2 = self.model_.score(X_seq, y_seq)
        print(f"   ✅ LinearForecaster fitted (R²={r2:.3f}, "
              f"n_train={len(X_seq)})")

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict prices for each horizon using momentum extrapolation.

        Args:
            X : (n_samples, n_features) — pass the FULL available window;
                the last `lookback` rows are used as the seed sequence.
                Row order: X[-1] is the most recent observation.
                Column 0 must be 'Returns', last column close prices
                are derived from y passed at fit time — so we use
                momentum from X[:, 0] (Returns column).

        Returns:
            np.ndarray of shape (len(horizons),) — predicted Close prices
            for each horizon in self.horizons.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")

        if len(X) < self.lookback:
            raise ValueError(
                f"LinearForecaster.predict(): need ≥{self.lookback} rows"
            )

        X_scaled = self.scaler_.transform(X)
        seed = X_scaled[-self.lookback:].flatten().reshape(1, -1)
        base_pred = float(self.model_.predict(seed)[0])

        # Momentum extrapolation
        returns_col = X[:, 0]                        # 'Returns' is col 0
        lookback_ret = returns_col[-min(20, len(X)):]
        mean_ret  = float(np.nanmean(lookback_ret))
        volatility = float(np.nanstd(lookback_ret)) or 0.01

        predictions = []
        for h in self.horizons:
            expected_ret = mean_ret * h
            predicted    = base_pred * (1 + expected_ret)
            predictions.append(predicted)

        return np.array(predictions)

    # ------------------------------------------------------------------
    def predict_with_meta(self, X: np.ndarray,
                          last_close: float) -> dict:
        """
        Convenience wrapper — returns a dict keyed by horizon days,
        matching the format expected by backtest.py.

        Args:
            X          : Feature array (same as predict).
            last_close : Actual last known Close price (for context).

        Returns:
            {5: {'predicted_price': ..., 'expected_return': ...}, ...}
        """
        raw = self.predict(X)
        result = {}
        for i, h in enumerate(self.horizons):
            pred_price = float(raw[i])
            exp_ret    = (pred_price - last_close) / last_close if last_close else 0.0
            result[h]  = {
                'predicted_price': pred_price,
                'expected_return': exp_ret,
                'horizon_days':    h
            }
        return result
