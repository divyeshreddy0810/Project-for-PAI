"""
RuleTrader
----------
Wraps the existing TradeSignalGenerator from rl_trader.py into the
modular Trader interface. Used by backtest.py for both experiment
configurations (baseline and HMM+LightGBM share the same trader).
"""

import numpy as np
from typing import Dict, Any, Optional
from src.trading.base import Trader


# Risk parameters — mirrors RISK_PARAMS from rl_trader.py
RISK_PARAMS = {
    'conservative': {
        'max_position_size': 0.05,
        'stop_loss':         0.05,
        'take_profit':       0.10,
        'min_confidence':    0.60,
    },
    'moderate': {
        'max_position_size': 0.10,
        'stop_loss':         0.08,
        'take_profit':       0.15,
        'min_confidence':    0.55,
    },
    'aggressive': {
        'max_position_size': 0.20,
        'stop_loss':         0.10,
        'take_profit':       0.25,
        'min_confidence':    0.50,
    },
}


class RuleTrader(Trader):
    """
    Rule-based trading signal generator.

    Args:
        risk_profile   : 'conservative', 'moderate', or 'aggressive'.
        portfolio_value: Starting portfolio value in dollars.
    """

    SIGNAL_THRESHOLDS = {
        'strong_buy':  0.4,
        'buy':         0.1,
        'hold_upper':  0.1,
        'hold_lower': -0.1,
        'sell':       -0.4,
    }

    def __init__(self, risk_profile: str = 'moderate',
                 portfolio_value: float = 100_000.0):
        if risk_profile not in RISK_PARAMS:
            raise ValueError(
                f"risk_profile must be one of {list(RISK_PARAMS.keys())}"
            )
        self.risk_profile   = risk_profile
        self.portfolio_value = portfolio_value
        self.params         = RISK_PARAMS[risk_profile]

    # ------------------------------------------------------------------
    @staticmethod
    def _score_to_label(score: float) -> str:
        """Return signal label string from numeric score."""
        if score >= 0.4:    return 'STRONG_BUY'
        elif score >= 0.1:  return 'BUY'
        elif score >= -0.1: return 'HOLD'
        elif score >= -0.4: return 'SELL'
        else:               return 'STRONG_SELL'

    def generate_signal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading signal from the current market state.

        Args:
            state: Dictionary with keys:
                - current_price    (float)
                - predicted_price  (float)  — 5-day prediction
                - expected_return  (float)  — fractional, e.g. 0.03
                - confidence       (float)  — [0, 1]
                - price_range      (tuple)  — (low, high) predicted range
                - regime           (str)    — 'bull'|'bear'|'sideways'
                - regime_confidence(float)  — optional, default 0.5
                - symbol           (str)    — optional

        Returns:
            Dictionary with signal, position sizing, and risk metrics.
        """
        current       = float(state.get('current_price', 0))
        predicted     = float(state.get('predicted_price', current))
        exp_ret       = float(state.get('expected_return', 0))
        confidence    = float(state.get('confidence', 0.5))
        price_range   = state.get('price_range', (current * 0.95,
                                                   current * 1.05))
        regime        = state.get('regime', 'sideways')
        reg_conf      = float(state.get('regime_confidence', 0.5))

        # ---- Factor 1: Momentum score --------------------------------
        if exp_ret > 0.05:
            momentum_score = 1.0
        elif exp_ret > 0.02:
            momentum_score = 0.5
        elif exp_ret < -0.05:
            momentum_score = -1.0
        elif exp_ret < -0.02:
            momentum_score = -0.5
        else:
            momentum_score = 0.0

        # ---- Factor 2: Regime score ----------------------------------
        if regime == 'bull':
            regime_score = 0.5 * reg_conf
        elif regime == 'bear':
            regime_score = -0.5 * reg_conf
        else:
            regime_score = 0.0

        # ---- Factor 3: Risk/reward -----------------------------------
        lower, upper = float(price_range[0]), float(price_range[1])
        downside = abs((current - lower) / current) if current else 0.01
        upside   = abs((upper - current) / current) if current else 0.01
        rr_ratio = upside / downside if downside > 0 else 1.0

        if rr_ratio > 2.0:
            rr_score = 1.0
        elif rr_ratio > 1.5:
            rr_score = 0.5
        elif rr_ratio < 0.5:
            rr_score = -1.0
        else:
            rr_score = 0.0

        # ---- Factor 4: Confidence score ------------------------------
        conf_score = float(np.clip((confidence - 0.5) * 2, -1, 1))

        # ---- Combine -------------------------------------------------
        total = (momentum_score * 0.25 +
                 regime_score   * 0.25 +
                 rr_score       * 0.20 +
                 conf_score     * 0.30)

        # ---- Signal label --------------------------------------------
        if total >= 0.4:
            signal = 'STRONG_BUY'
        elif total >= 0.1:
            signal = 'BUY'
        elif total >= -0.1:
            signal = 'HOLD'
        elif total >= -0.4:
            signal = 'SELL'
        else:
            signal = 'STRONG_SELL'

        # ---- Regime-aware position sizing ---------------------------
        base = self.params['max_position_size']
        size = base * abs(total)

        # Scale size by regime
        if regime == 'bull' and 'BUY' in self._score_to_label(total):
            size *= 1.2   # Trend alignment: increase size
        elif regime == 'bear' and 'SELL' in self._score_to_label(total):
            size *= 1.2   # Trend alignment: increase size
        elif regime == 'sideways':
            size *= 0.7   # Uncertain regime: reduce size

        if confidence < self.params['min_confidence']:
            size *= 0.5
        if downside > self.params['stop_loss']:
            size *= 0.7
        size = float(np.clip(size, 0, base))

        # ---- Regime-aware stop/take profit ---------------------------
        # Wider stops in sideways (choppy), tighter in trending regimes
        if regime == 'sideways':
            sl_mult = 1.3
            tp_mult = 0.8
        elif regime in ('bull', 'bear'):
            sl_mult = 0.85
            tp_mult = 1.2
        else:
            sl_mult = 1.0
            tp_mult = 1.0

        stop_loss_price   = current * (1 - self.params['stop_loss'] * sl_mult)
        take_profit_price = current * (1 + self.params['take_profit'] * tp_mult)

        win_prob     = (confidence + 0.5) / 2
        expected_val = (win_prob * exp_ret
                        - (1 - win_prob) * self.params['stop_loss'])

        return {
            'symbol':           state.get('symbol', 'UNKNOWN'),
            'current_price':    current,
            'predicted_price':  predicted,
            'expected_return':  exp_ret,
            'signal':           signal,
            'signal_strength':  total,
            'confidence':       confidence,
            'position_size':    size,
            'position_value':   size * self.portfolio_value,
            'stop_loss':        stop_loss_price,
            'take_profit':      take_profit_price,
            'risk_reward_ratio':rr_ratio,
            'downside_risk':    downside,
            'expected_value':   expected_val,
            'momentum_score':   momentum_score,
            'regime_score':     regime_score,
            'rr_score':         rr_score,
            'confidence_score': conf_score,
            'regime':           regime,
        }
