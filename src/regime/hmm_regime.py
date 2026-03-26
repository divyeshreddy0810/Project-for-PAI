"""
HMMRegimeDetector
-----------------
Uses a Gaussian HMM (hmmlearn) with diagonal covariance to detect
market regimes from daily features. Used as the ENHANCED system
in comparative experiments.

Features used (all computed from price + sentiment data):
    - daily_return     : Close pct change
    - log_volatility   : Rolling 10-day std of returns (log scale)
    - sentiment_mean   : Daily sentiment score
    - rsi_norm         : RSI normalised to [-1, 1]

States are AUTO-LABELLED after fitting by sorting on mean daily_return:
    lowest  return state → 'bear'
    middle  return state → 'sideways'  (only when n_states=3)
    highest return state → 'bull'
"""

import numpy as np
import pandas as pd
from typing import Any, Optional
from src.regime.base import RegimeDetector

try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False


class HMMRegimeDetector(RegimeDetector):
    """
    Gaussian HMM regime detector with automatic state labelling.

    Args:
        n_states  : Number of hidden states (2 or 3). Default 3.
        n_iter    : EM iterations. Default 200.
        random_state : For reproducibility. Default 42.
    """

    LABEL_MAP_2 = {0: 'bear', 1: 'bull'}
    LABEL_MAP_3 = {0: 'bear', 1: 'sideways', 2: 'bull'}

    def __init__(self, n_states: int = 3, n_iter: int = 1000,
                 random_state: int = 42):
        if not HMMLEARN_AVAILABLE:
            raise ImportError(
                "hmmlearn is required. Install with: pip install hmmlearn"
            )
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model_: Optional[Any] = None
        self.state_to_label_: dict = {}
        self.fitted_ = False

    # ------------------------------------------------------------------
    @staticmethod
    def _build_features(data: pd.DataFrame) -> np.ndarray:
        """
        Build the (n_samples, 4) feature matrix from a DataFrame.

        Required columns: Close, RSI, sentiment_mean
        Optional columns: sentiment_trend (falls back to 0)
        """
        required = ['Close', 'RSI', 'sentiment_mean']
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"HMMRegimeDetector: missing columns {missing}")

        df = data.copy()

        # Daily return
        df['_ret'] = df['Close'].pct_change().fillna(0.0)

        # Log volatility (rolling 10-day std of returns, log-scaled)
        vol = df['_ret'].rolling(10, min_periods=2).std().fillna(
            df['_ret'].std()
        )
        df['_logvol'] = np.log1p(vol.clip(lower=0))

        # RSI normalised to [-1, 1]
        df['_rsi_norm'] = (df['RSI'].fillna(50.0) - 50.0) / 50.0

        # Sentiment (already in roughly [-1, 1])
        df['_sent'] = df['sentiment_mean'].fillna(0.0)

        feat_cols = ['_ret', '_logvol', '_rsi_norm', '_sent']
        X = df[feat_cols].values.astype(float)

        # Replace any remaining NaN / inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the HMM on training data.

        Args:
            data: DataFrame with columns Close, RSI, sentiment_mean.
                  Must have at least 30 rows.
        """
        if len(data) < 30:
            raise ValueError(
                f"HMMRegimeDetector.fit(): need ≥30 rows, got {len(data)}"
            )

        X = self._build_features(data)

        from sklearn.preprocessing import StandardScaler
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X)

        self.model_ = hmmlearn_hmm.GaussianHMM(
            n_components  = self.n_states,
            covariance_type = 'diag',
            n_iter        = self.n_iter,
            tol           = 1e-4,
            random_state  = self.random_state,
            init_params   = 'stmc',
            params        = 'stmc',
            verbose       = False
        )
        self.model_.fit(X)

        # --- Auto-label states by mean daily_return of each state ---
        states = self.model_.predict(X)
        mean_ret = {}
        for s in range(self.n_states):
            mask = states == s
            mean_ret[s] = float(X[mask, 0].mean()) if mask.any() else 0.0

        sorted_states = sorted(mean_ret, key=mean_ret.get)  # low → high

        if self.n_states == 2:
            self.state_to_label_ = {
                sorted_states[0]: 'bear',
                sorted_states[1]: 'bull'
            }
        else:  # 3 states
            self.state_to_label_ = {
                sorted_states[0]: 'bear',
                sorted_states[1]: 'sideways',
                sorted_states[2]: 'bull'
            }

        self.fitted_ = True
        print(f"   ✅ HMM fitted: state→label map = {self.state_to_label_}")

    # ------------------------------------------------------------------
    def predict(self, data: pd.DataFrame) -> str:
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")
        X = self._build_features(data)
        if hasattr(self, 'scaler_'):
            X = self.scaler_.transform(X)
        states = self.model_.predict(X)
        last_state = int(states[-1])
        return self.state_to_label_.get(last_state, 'sideways')

    # ------------------------------------------------------------------
    def predict_series(self, data: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict_series().")
        X = self._build_features(data)
        if hasattr(self, 'scaler_'):
            X = self.scaler_.transform(X)
        states = self.model_.predict(X)
        labels = [self.state_to_label_.get(int(s), 'sideways') for s in states]
        return pd.Series(labels, index=data.index, name='regime')
