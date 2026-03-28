"""
EnsembleTrader
--------------
Combines two signals:
  - Baseline (score-regime) for DIRECTION  — better directional accuracy
  - LightGBM forecast for MAGNITUDE        — better price level accuracy

Logic:
  1. Get direction vote from score-based regime (+1 / -1 / 0)
  2. Get expected return from LightGBM prediction
  3. Only open a trade when BOTH agree on direction
  4. Size the trade using LightGBM's confidence (magnitude)
  5. Apply regime-aware stop/take-profit from RuleTrader

This hybrid addresses the key weakness of each individual system:
  - Score regime:  good direction, poor price level accuracy
  - LightGBM:      poor direction (41%), excellent price accuracy (MAE 239)
"""

import numpy as np
from typing import Dict, Any
from src.trading.base import Trader

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


def calculate_adx(high, low, close, period=14):
    """ADX trend strength. <20 = choppy, >25 = trending."""
    import pandas as pd
    import numpy as np
    h = pd.Series(high) if not isinstance(high, pd.Series) else high
    l = pd.Series(low)  if not isinstance(low,  pd.Series) else low
    c = pd.Series(close) if not isinstance(close, pd.Series) else close

    tr   = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr  = tr.rolling(period).mean()

    plus_dm  = h.diff()
    minus_dm = (-l.diff())
    plus_dm[plus_dm  < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0

    plus_di  = 100 * (plus_dm.rolling(period).mean()  / atr.replace(0, 1e-8))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, 1e-8))
    dx       = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-8))
    adx      = dx.rolling(period).mean()
    return float(adx.iloc[-1]) if not adx.empty else 20.0


def calculate_ker(close, period=10):
    """
    Kaufman Efficiency Ratio — measures trend efficiency.
    1.0 = perfect trend, 0.0 = pure chop.
    Below 0.30 = choppy, avoid trading.
    """
    import pandas as pd
    c = pd.Series(close) if not isinstance(close, pd.Series) else close
    if len(c) < period + 1:
        return 0.5
    net_change  = abs(float(c.iloc[-1]) - float(c.iloc[-period-1]))
    path_length = float(c.diff().abs().iloc[-period:].sum())
    if path_length == 0:
        return 0.0
    return min(1.0, net_change / path_length)


def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index. >61.8 = choppy, <38.2 = trending.
    """
    import pandas as pd, numpy as np
    h = pd.Series(high)  if not isinstance(high,  pd.Series) else high
    l = pd.Series(low)   if not isinstance(low,   pd.Series) else low
    c = pd.Series(close) if not isinstance(close, pd.Series) else close
    if len(c) < period + 1:
        return 50.0
    tr = pd.concat([h-l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_sum    = tr.rolling(period).sum().iloc[-1]
    price_range= h.rolling(period).max().iloc[-1] - l.rolling(period).min().iloc[-1]
    if price_range == 0 or atr_sum == 0:
        return 50.0
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    return float(chop)


def detect_bb_squeeze(close, period=20, std_mult=2.0, lookback=60):
    """True if BB bandwidth at recent low — avoid trading during squeezes."""
    import pandas as pd
    c    = pd.Series(close) if not isinstance(close, pd.Series) else close
    sma  = c.rolling(period).mean()
    std  = c.rolling(period).std()
    bw   = ((sma + std_mult*std) - (sma - std_mult*std)) / sma.replace(0,1e-8)
    min_bw = bw.rolling(lookback, min_periods=10).min()
    is_squeeze = float(bw.iloc[-1]) <= float(min_bw.iloc[-1]) * 1.05
    return is_squeeze


class EnsembleTrader(Trader):
    """
    Ensemble trader combining score-regime direction with
    LightGBM magnitude.

    Args:
        risk_profile     : 'conservative', 'moderate', or 'aggressive'
        portfolio_value  : Starting portfolio value
        agreement_only   : If True, only trade when both models agree
                           on direction. Default True.
        lgbm_weight      : How much to weight LightGBM return vs
                           score return when sizing. Default 0.7.
    """

    def __init__(self, risk_profile: str = 'moderate',
                 portfolio_value: float = 100_000.0,
                 agreement_only: bool = True,
                 lgbm_weight: float = 0.7):
        if risk_profile not in RISK_PARAMS:
            raise ValueError(f"risk_profile must be one of {list(RISK_PARAMS.keys())}")
        self.risk_profile    = risk_profile
        self.portfolio_value = portfolio_value
        self.agreement_only  = agreement_only
        self.lgbm_weight     = lgbm_weight
        self.params          = RISK_PARAMS[risk_profile]

    # ------------------------------------------------------------------
    def generate_signal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ensemble trading signal.

        Args:
            state: Dictionary with keys:
                - current_price        (float)
                - predicted_price_lgbm (float)  — LightGBM 5d prediction
                - expected_return_lgbm (float)  — LightGBM expected return
                - predicted_price_score(float)  — Score-regime 5d prediction
                - expected_return_score(float)  — Score-regime expected return
                - confidence           (float)  — [0, 1]
                - price_range          (tuple)  — (low, high)
                - regime               (str)    — 'bull'|'bear'|'sideways'
                - regime_confidence    (float)
                - symbol               (str)

        Returns:
            Trading signal dict with ensemble metadata.
        """
        current          = float(state.get('current_price', 0))
        exp_ret_lgbm     = float(state.get('expected_return_lgbm', 0))
        exp_ret_score    = float(state.get('expected_return_score', 0))
        pred_lgbm        = float(state.get('predicted_price_lgbm', current))
        confidence       = float(state.get('confidence', 0.5))
        price_range      = state.get('price_range', (current * 0.95, current * 1.05))
        regime           = state.get('regime', 'sideways')
        reg_conf         = float(state.get('regime_confidence', 0.5))

        # ---- Step 1: Direction votes ---------------------------------
        # Score regime direction (from regime label — more reliable)
        if regime == 'bull':
            score_direction = 1
        elif regime == 'bear':
            score_direction = -1
        else:
            score_direction = 0

        # LightGBM direction (from predicted return)
        if exp_ret_lgbm > 0.005:
            lgbm_direction = 1
        elif exp_ret_lgbm < -0.005:
            lgbm_direction = -1
        else:
            lgbm_direction = 0

        # ---- Step 2: Agreement check --------------------------------
        agreement = (score_direction == lgbm_direction and
                     score_direction != 0)

        if self.agreement_only and not agreement:
            return self._neutral_signal(state, reason='no_agreement')

        # ---- Step 3: Blended expected return -------------------------
        blended_return = (
            self.lgbm_weight       * exp_ret_lgbm +
            (1 - self.lgbm_weight) * exp_ret_score
        )

        # ---- Step 4: Signal strength ---------------------------------
        # Momentum component
        if abs(blended_return) > 0.05:
            momentum_score = np.sign(blended_return) * 1.0
        elif abs(blended_return) > 0.02:
            momentum_score = np.sign(blended_return) * 0.5
        else:
            momentum_score = 0.0

        # Regime component
        regime_score = score_direction * 0.5 * reg_conf

        # Confidence component
        conf_score = float(np.clip((confidence - 0.5) * 2, -1, 1))

        # Risk/reward component
        lower, upper = float(price_range[0]), float(price_range[1])
        downside = abs((current - lower) / current) if current else 0.01
        upside   = abs((upper - current) / current) if current else 0.01
        rr_ratio = upside / downside if downside > 0 else 1.0
        rr_score = 1.0 if rr_ratio > 2.0 else 0.5 if rr_ratio > 1.5 else -1.0 if rr_ratio < 0.5 else 0.0

        total = (momentum_score * 0.25 +
                 regime_score   * 0.30 +   # Higher weight — regime is our direction anchor
                 rr_score       * 0.15 +
                 conf_score     * 0.30)

        # ---- Step 5: Signal label ------------------------------------
        if total >= 0.4:    signal = 'STRONG_BUY'
        elif total >= 0.1:  signal = 'BUY'
        elif total >= -0.1: signal = 'HOLD'
        elif total >= -0.4: signal = 'SELL'
        else:               signal = 'STRONG_SELL'

        # ---- Step 6: Regime-aware position sizing --------------------
        base = self.params['max_position_size']
        size = base * abs(total)

        if regime == 'bull' and 'BUY' in signal:
            size *= 1.2
        elif regime == 'bear' and 'SELL' in signal:
            size *= 1.2
        elif regime == 'sideways':
            size *= 0.6   # Extra caution in sideways — only agreement trades allowed

        if confidence < self.params['min_confidence']:
            size *= 0.5
        size = float(np.clip(size, 0, base))

        # ---- Step 7: Regime-aware stops ------------------------------
        if regime == 'sideways':
            sl_mult, tp_mult = 1.3, 0.8
        else:
            sl_mult, tp_mult = 0.85, 1.2

        stop_loss_price   = current * (1 - self.params['stop_loss'] * sl_mult)
        take_profit_price = current * (1 + self.params['take_profit'] * tp_mult)

        win_prob     = (confidence + 0.5) / 2
        expected_val = (win_prob * blended_return
                        - (1 - win_prob) * self.params['stop_loss'])

        return {
            'symbol':               state.get('symbol', 'UNKNOWN'),
            'current_price':        current,
            'predicted_price':      pred_lgbm,
            'expected_return':      blended_return,
            'expected_return_lgbm': exp_ret_lgbm,
            'expected_return_score':exp_ret_score,
            'signal':               signal,
            'signal_strength':      total,
            'confidence':           confidence,
            'position_size':        size,
            'position_value':       size * self.portfolio_value,
            'stop_loss':            stop_loss_price,
            'take_profit':          take_profit_price,
            'risk_reward_ratio':    rr_ratio,
            'downside_risk':        downside,
            'expected_value':       expected_val,
            'momentum_score':       momentum_score,
            'regime_score':         regime_score,
            'rr_score':             rr_score,
            'confidence_score':     conf_score,
            'regime':               regime,
            'agreement':            agreement,
            'score_direction':      score_direction,
            'lgbm_direction':       lgbm_direction,
        }

    # ------------------------------------------------------------------
    def _neutral_signal(self, state: Dict[str, Any],
                        reason: str = '') -> Dict[str, Any]:
        return {
            'symbol':        state.get('symbol', 'UNKNOWN'),
            'current_price': float(state.get('current_price', 0)),
            'signal':        'HOLD',
            'signal_strength': 0.0,
            'confidence':    0.0,
            'position_size': 0.0,
            'position_value':0.0,
            'expected_return':0.0,
            'reason':        reason,
            'agreement':     False,
        }
