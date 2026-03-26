"""
Evaluation Metrics
------------------
All metric functions used to compare experiment configurations.

Functions:
    forecast_mae          — Mean Absolute Error
    forecast_rmse         — Root Mean Squared Error
    directional_accuracy  — % of correct direction predictions
    total_return          — Cumulative return over the test period
    sharpe_ratio          — Annualised Sharpe (daily returns)
    max_drawdown          — Maximum peak-to-trough drawdown
    win_rate              — % of profitable trades
    summarise             — Returns all metrics as a single dict
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional


# ------------------------------------------------------------------
def forecast_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error between predicted and actual prices."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


# ------------------------------------------------------------------
def forecast_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error between predicted and actual prices."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


# ------------------------------------------------------------------
def directional_accuracy(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_prev: np.ndarray) -> float:
    """
    Fraction of predictions where the predicted direction
    (up/down vs previous close) matches the actual direction.

    Args:
        y_true : Actual future prices.
        y_pred : Predicted future prices.
        y_prev : Previous close prices (baseline for direction).

    Returns:
        Float in [0, 1].
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_prev = np.array(y_prev, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_prev))
    if mask.sum() == 0:
        return np.nan

    actual_dir    = np.sign(y_true[mask] - y_prev[mask])
    predicted_dir = np.sign(y_pred[mask] - y_prev[mask])
    return float(np.mean(actual_dir == predicted_dir))


# ------------------------------------------------------------------
def total_return(equity_curve: np.ndarray) -> float:
    """
    Total cumulative return from an equity curve.

    Args:
        equity_curve: Array of portfolio values over time.

    Returns:
        Float, e.g. 0.18 means +18%.
    """
    curve = np.array(equity_curve, dtype=float)
    curve = curve[~np.isnan(curve)]
    if len(curve) < 2 or curve[0] == 0:
        return np.nan
    return float((curve[-1] - curve[0]) / curve[0])


# ------------------------------------------------------------------
def sharpe_ratio(daily_returns: np.ndarray,
                 risk_free_rate: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """
    Annualised Sharpe ratio from daily returns.

    Args:
        daily_returns    : Array of daily P&L returns (fractional).
        risk_free_rate   : Annual risk-free rate. Default 0.0.
        periods_per_year : Trading days per year. Default 252.

    Returns:
        Float Sharpe ratio.
    """
    rets = np.array(daily_returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    if len(rets) < 2:
        return np.nan

    daily_rf  = risk_free_rate / periods_per_year
    excess    = rets - daily_rf
    mean_exc  = np.mean(excess)
    std_exc   = np.std(excess, ddof=1)

    if std_exc == 0:
        return np.nan

    return float((mean_exc / std_exc) * np.sqrt(periods_per_year))


# ------------------------------------------------------------------
def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown of an equity curve.

    Returns:
        Float, e.g. -0.12 means -12% drawdown.
    """
    curve = np.array(equity_curve, dtype=float)
    curve = curve[~np.isnan(curve)]
    if len(curve) < 2:
        return np.nan

    peak    = np.maximum.accumulate(curve)
    # Avoid division by zero
    drawdown = np.where(peak > 0, (curve - peak) / peak, 0.0)
    return float(np.min(drawdown))


# ------------------------------------------------------------------
def win_rate(trade_returns: np.ndarray) -> float:
    """
    Fraction of trades with positive return.

    Args:
        trade_returns: Array of per-trade P&L (fractional).

    Returns:
        Float in [0, 1].
    """
    rets = np.array(trade_returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return np.nan
    return float(np.mean(rets > 0))


# ------------------------------------------------------------------
def summarise(y_true:        np.ndarray,
              y_pred:        np.ndarray,
              y_prev:        np.ndarray,
              equity_curve:  np.ndarray,
              daily_returns: np.ndarray,
              trade_returns: np.ndarray,
              label:         str = '') -> Dict[str, Any]:
    """
    Compute all metrics and return as a single dictionary.

    Args:
        y_true        : Actual future prices.
        y_pred        : Predicted prices.
        y_prev        : Previous close prices.
        equity_curve  : Portfolio value over time.
        daily_returns : Daily P&L returns.
        trade_returns : Per-trade P&L returns.
        label         : Experiment name (for display).

    Returns:
        Dict with keys matching experiment_goal.metrics in project JSON.
    """
    return {
        'label':               label,
        'forecast_mae':        forecast_mae(y_true, y_pred),
        'forecast_rmse':       forecast_rmse(y_true, y_pred),
        'directional_accuracy':directional_accuracy(y_true, y_pred, y_prev),
        'total_return':        total_return(equity_curve),
        'sharpe_ratio':        sharpe_ratio(daily_returns),
        'max_drawdown':        max_drawdown(equity_curve),
        'win_rate':            win_rate(trade_returns),
    }
