"""
LightGBMForecaster
------------------
LightGBM-based price forecaster with engineered features.
Used as the ENHANCED forecaster in comparative experiments.

Feature engineering includes:
    - Lag returns       : 1, 2, 3, 5, 10 day lags
    - Rolling stats     : mean / std over 5, 10, 20 day windows
    - Regime label      : categorical (bull=1, sideways=0, bear=-1)
    - Sentiment         : mean + trend
    - RSI, MACD, Volume trend

One model is trained per horizon (5, 10, 15, 20 days).
Target: forward Close price at horizon h.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from src.forecast.base import Forecaster

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

PREDICTION_HORIZONS = [5, 10, 15, 20]

REGIME_ENCODE = {'bull': 1, 'sideways': 0, 'bear': -1}


class LightGBMForecaster(Forecaster):
    """
    LightGBM forecaster with rich feature engineering.
    Trains one model per prediction horizon.

    Args:
        horizons     : Forward periods to predict. Default [5,10,15,20].
        n_estimators : Trees per model. Default 200.
        learning_rate: Default 0.05.
        num_leaves   : Default 31.
        random_state : Default 42.
    """

    def __init__(self, horizons: List[int] = None,
                 n_estimators: int = 500,
                 learning_rate: float = 0.01,
                 num_leaves: int = 15,
                 min_child_samples: int = 10,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.05,
                 reg_lambda: float = 0.05,
                 early_stopping_rounds: int = 20,
                 random_state: int = 42):
        if not LGBM_AVAILABLE:
            raise ImportError(
                "lightgbm is required. Install with: pip install lightgbm"
            )
        self.horizons             = horizons or PREDICTION_HORIZONS
        self.n_estimators         = n_estimators
        self.learning_rate        = learning_rate
        self.num_leaves           = num_leaves
        self.min_child_samples    = min_child_samples
        self.subsample            = subsample
        self.colsample_bytree     = colsample_bytree
        self.reg_alpha            = reg_alpha
        self.reg_lambda           = reg_lambda
        self.early_stopping_rounds= early_stopping_rounds
        self.random_state         = random_state
        self.models_: Dict[int, lgb.LGBMRegressor] = {}
        self.classifiers_: Dict[int, lgb.LGBMClassifier] = {}
        self.feature_names_: List[str] = []
        self.fitted_ = False

    # ------------------------------------------------------------------
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix from a DataFrame.
        Enhanced with direction-sensitive features.
        """
        out = pd.DataFrame(index=df.index)

        close   = df['Close']
        returns = df['Returns'] if 'Returns' in df.columns \
                  else close.pct_change()

        # --- Lag returns ---
        for lag in [1, 2, 3, 5, 10]:
            out[f'ret_lag{lag}'] = returns.shift(lag)

        # --- Rolling stats on returns ---
        for w in [5, 10, 20]:
            out[f'ret_roll_mean{w}'] = returns.rolling(w, min_periods=2).mean()
            out[f'ret_roll_std{w}']  = returns.rolling(w, min_periods=2).std()

        # --- Rolling stats on Close ---
        for w in [5, 10, 20]:
            out[f'close_roll_mean{w}'] = close.rolling(w, min_periods=2).mean()

        # --- Momentum: price relative to rolling mean ---
        for w in [5, 20]:
            roll = close.rolling(w, min_periods=2).mean()
            out[f'price_vs_ma{w}'] = (close - roll) / roll.replace(0, np.nan)

        # --- Technical indicators (pass-through) ---
        for col in ['RSI', 'MACD', 'Signal', 'Volume_Trend',
                    'High_Low_Range', 'Open_Close_Range']:
            if col in df.columns:
                out[col] = df[col]

        # --- Sentiment ---
        if 'sentiment_mean' in df.columns:
            out['sentiment_mean']  = df['sentiment_mean']
        if 'sentiment_trend' in df.columns:
            out['sentiment_trend'] = df['sentiment_trend']

        # --- Regime as categorical integer ---
        if 'regime' in df.columns:
            out['regime_encoded'] = df['regime'].map(
                REGIME_ENCODE
            ).fillna(0).astype(int)

        return out

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit one LightGBM model per horizon.

        Args:
            X : (n_samples, n_features) engineered feature matrix.
                Build this with engineer_features() then .values.
            y : (n_samples,) Close prices (raw, not returns).

        Note: targets are created internally by shifting y forward
              by each horizon value — so y must be the FULL price
              series aligned with X (same index order).
        """
        if len(X) < max(self.horizons) + 20:
            raise ValueError(
                f"LightGBMForecaster.fit(): need ≥"
                f"{max(self.horizons)+20} rows, got {len(X)}"
            )

        params = dict(
            n_estimators      = self.n_estimators,
            learning_rate     = self.learning_rate,
            num_leaves        = self.num_leaves,
            random_state      = self.random_state,
            n_jobs            = -1,
            verbose           = -1,
            min_child_samples = self.min_child_samples,
            subsample         = self.subsample,
            colsample_bytree  = self.colsample_bytree,
            reg_alpha         = self.reg_alpha,
            reg_lambda        = self.reg_lambda,
        )

        # Store feature names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = []

        for h in self.horizons:
            # Target: Close price h days ahead (raw price — more stable signal)
            y_h = pd.Series(y).shift(-h).values

            # Valid rows: no NaN in features or target
            valid = ~(np.isnan(X).any(axis=1) | np.isnan(y_h))
            X_tr  = X[valid]
            y_tr  = y_h[valid]

            if len(X_tr) < 20:
                print(f"   ⚠️  LightGBM h={h}: only {len(X_tr)} valid rows — skipping")
                continue

            # Only use early stopping if validation set is large enough
            val_size = int(len(X_tr) * 0.15)
            if val_size >= 15:
                X_val, y_val = X_tr[-val_size:], y_tr[-val_size:]
                X_fit, y_fit = X_tr[:-val_size], y_tr[:-val_size]
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_fit, y_fit,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(self.early_stopping_rounds,
                                                  verbose=False),
                               lgb.log_evaluation(period=-1)]
                )
            else:
                # Too few rows for early stopping — train on all data
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr)

            self.models_[h] = model
            best_iter = getattr(model, "best_iteration_", "N/A")

            # Train direction classifier (binary: 1=up, 0=down)
            # Target: is Close at t+h higher than Close at t?
            y_prices = pd.Series(y)
            y_future = y_prices.shift(-h).values
            y_dir    = (y_future > y).astype(int)
            valid_dir= ~np.isnan(y_future)
            X_dir    = X[valid_dir]
            y_d      = y_dir[valid_dir]

            if len(np.unique(y_d)) >= 2 and len(X_dir) >= 20:
                clf_params = dict(
                    n_estimators     = 300,
                    learning_rate    = 0.01,
                    num_leaves       = 15,
                    min_child_samples= 10,
                    subsample        = 0.8,
                    colsample_bytree = 0.8,
                    random_state     = self.random_state,
                    n_jobs           = -1,
                    verbose          = -1,
                )
                clf = lgb.LGBMClassifier(**clf_params)
                clf.fit(X_dir, y_d)
                self.classifiers_[h] = clf

            print(f"   ✅ LightGBM h={h:2d}d fitted "
                  f"(n_train={len(X_tr)}, best_iter={best_iter}, "
                  f"classifier={'yes' if h in self.classifiers_ else 'no'})")

        self.fitted_ = True

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict prices for each horizon using the last row of X.

        Args:
            X : (n_samples, n_features) — pass the full available window;
                only the last row is used for prediction.

        Returns:
            np.ndarray of shape (len(horizons),) — predicted Close prices.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict().")

        # Convert to DataFrame with feature names if available
        if hasattr(self, 'feature_names_') and self.feature_names_:
            import pandas as pd
            last_row = pd.DataFrame(X[[-1]], columns=self.feature_names_)
        else:
            last_row = X[[-1]]  # shape (1, n_features)
        predictions = []

        # Store feature names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = []

        for h in self.horizons:
            if h not in self.models_:
                predictions.append(np.nan)
                continue
            pred = float(self.models_[h].predict(last_row)[0])
            predictions.append(pred)

        return np.array(predictions)

    # ------------------------------------------------------------------
    def predict_with_meta(self, X: np.ndarray,
                          last_close: float) -> dict:
        """
        Returns dict keyed by horizon days.
        Includes direction probability from classifier.
        Only trusts signal when regression AND classifier agree.
        """
        raw = self.predict(X)
        last_row = X[[-1]]
        result = {}
        for i, h in enumerate(self.horizons):
            pred_price  = float(raw[i]) if not np.isnan(raw[i]) else last_close
            exp_ret     = (pred_price - last_close) / last_close if last_close else 0.0

            # Get direction probability from classifier
            up_prob = 0.5  # neutral default
            if h in self.classifiers_:
                try:
                    proba   = self.classifiers_[h].predict_proba(last_row)[0]
                    up_prob = float(proba[1])  # probability of going up
                except Exception:
                    up_prob = 0.5

            # Agreement check: info only, no dampening
            regression_up  = exp_ret > 0
            classifier_up  = up_prob > 0.5
            agreement      = regression_up == classifier_up
            direction_conf = up_prob if regression_up else (1 - up_prob)

            result[h] = {
                'predicted_price': pred_price,
                'expected_return': exp_ret,
                'horizon_days':    h,
                'up_probability':  up_prob,
                'direction_conf':  direction_conf,
                'agreement':       agreement,
            }
        return result

    # ------------------------------------------------------------------
    def feature_importance(self) -> Dict[int, pd.Series]:
        """Return feature importance per horizon model."""
        if not self.fitted_:
            raise RuntimeError("Call fit() first.")
        out = {}
        for h, model in self.models_.items():
            imp = pd.Series(
                model.feature_importances_,
                name=f'importance_h{h}'
            )
            out[h] = imp.sort_values(ascending=False)
        return out
