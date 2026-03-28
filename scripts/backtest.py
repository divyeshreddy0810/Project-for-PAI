"""
Walk-Forward Backtester
-----------------------
Loads a config file, loops over expanding walk-forward windows,
instantiates the specified regime detector + forecaster + trader,
runs simulation on each test window, and stores results.

Usage:
    python scripts/backtest.py --config configs/baseline.json
    python scripts/backtest.py --config configs/hmm_lgbm.json

Output:
    data/output/results_<config_name>_<timestamp>.json
    data/output/results_<config_name>_<timestamp>.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

# Make src importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
from src.evaluation.metrics import summarise

OUTPUT_DIR = "data/output"


# ==================================================================
# DATA HELPERS
# ==================================================================

def fetch_data(symbol: str, period: str = '2y') -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance and flatten columns."""
    print(f"   📥 Fetching {period} of daily data for {symbol}...")
    df = yf.download(symbol, period=period, interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.dropna(subset=['Close'], inplace=True)
    print(f"   ✅ {len(df)} rows fetched")
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Returns, MAs, RSI, MACD, Volatility, Volume_Trend."""
    df = df.copy()
    close = df['Close']

    df['Returns']         = close.pct_change()
    df['Price_MA5']       = close.rolling(5,  min_periods=2).mean()
    df['Price_MA20']      = close.rolling(20, min_periods=5).mean()
    df['Price_MA50']      = close.rolling(50, min_periods=10).mean()
    df['SMA_20']          = df['Price_MA20']
    df['SMA_50']          = df['Price_MA50']
    df['SMA_200']         = close.rolling(200, min_periods=30).mean()
    df['Volatility']      = df['Returns'].rolling(14, min_periods=3).std()

    # RSI
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'].fillna(50, inplace=True)

    # MACD
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    df['MACD']  = ema12 - ema26
    df['Signal']= df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Signal'] = df['Signal']

    # Volume trend
    vol_ma = df['Volume'].rolling(20, min_periods=5).mean()
    df['Volume_Trend'] = (df['Volume'] / vol_ma.replace(0, np.nan)).fillna(1.0)

    df['High_Low_Range']   = (df['High'] - df['Low']) / close.replace(0, np.nan)
    df['Open_Close_Range'] = (close - df['Open']) / df['Open'].replace(0, np.nan)

    return df


def add_cross_asset_features(df: pd.DataFrame,
                              symbol: str) -> pd.DataFrame:
    """
    Adds VIX9D/VIX stress ratio + BTC + Gold cross-asset momentum.
    Falls back gracefully if any ticker fails.
    """
    df = df.copy()
    start = df.index[0].strftime('%Y-%m-%d')
    end   = df.index[-1].strftime('%Y-%m-%d')

    # VIX9D / VIX stress ratio
    try:
        vix9  = yf.download('^VIX9D', start=start, end=end,
                             progress=False)
        vix   = yf.download('^VIX',   start=start, end=end,
                             progress=False)
        if isinstance(vix9.columns, pd.MultiIndex):
            vix9.columns = vix9.columns.droplevel(1)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns  = vix.columns.droplevel(1)
        vix9_close = vix9['Close'].reindex(df.index, method='ffill')
        vix_close  = vix['Close'].reindex(df.index,  method='ffill')
        df['vix_stress_ratio'] = (vix9_close / vix_close.replace(0, np.nan)
                                  ).fillna(1.0)
        df['vix_level']        = vix_close.fillna(20.0)
    except Exception:
        df['vix_stress_ratio'] = 1.0
        df['vix_level']        = 20.0

    # BTC + Gold cross-asset momentum
    try:
        btc  = yf.download('BTC-USD', start=start, end=end,
                            progress=False)
        gold = yf.download('GC=F',    start=start, end=end,
                            progress=False)
        if isinstance(btc.columns,  pd.MultiIndex):
            btc.columns  = btc.columns.droplevel(1)
        if isinstance(gold.columns, pd.MultiIndex):
            gold.columns = gold.columns.droplevel(1)
        btc_c  = btc['Close'].reindex(df.index,  method='ffill')
        gold_c = gold['Close'].reindex(df.index, method='ffill')
        btc_ma20  = btc_c.rolling(20,  min_periods=5).mean()
        gold_ma20 = gold_c.rolling(20, min_periods=5).mean()
        df['btc_above_ma20']  = (btc_c  > btc_ma20 ).astype(float).fillna(0)
        df['gold_above_ma20'] = (gold_c > gold_ma20).astype(float).fillna(0)
        # Risk composite: 0=both below, 1=mixed, 2=both above
        df['risk_composite']  = df['btc_above_ma20'] + df['gold_above_ma20']
        # BTC 5-day momentum
        df['btc_mom5'] = btc_c.pct_change(5).reindex(
            df.index, method='ffill').fillna(0)
    except Exception:
        df['btc_above_ma20']  = 0.5
        df['gold_above_ma20'] = 0.5
        df['risk_composite']  = 1.0
        df['btc_mom5']        = 0.0

    return df


def add_cross_asset_features(df: pd.DataFrame,
                              symbol: str) -> pd.DataFrame:
    """
    Adds VIX9D/VIX stress ratio + BTC + Gold cross-asset momentum.
    Falls back gracefully if any ticker fails.
    """
    df = df.copy()
    start = df.index[0].strftime('%Y-%m-%d')
    end   = df.index[-1].strftime('%Y-%m-%d')

    # VIX9D / VIX stress ratio
    try:
        vix9  = yf.download('^VIX9D', start=start, end=end,
                             progress=False)
        vix   = yf.download('^VIX',   start=start, end=end,
                             progress=False)
        if isinstance(vix9.columns, pd.MultiIndex):
            vix9.columns = vix9.columns.droplevel(1)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns  = vix.columns.droplevel(1)
        vix9_close = vix9['Close'].reindex(df.index, method='ffill')
        vix_close  = vix['Close'].reindex(df.index,  method='ffill')
        df['vix_stress_ratio'] = (vix9_close / vix_close.replace(0, np.nan)
                                  ).fillna(1.0)
        df['vix_level']        = vix_close.fillna(20.0)
    except Exception:
        df['vix_stress_ratio'] = 1.0
        df['vix_level']        = 20.0

    # BTC + Gold cross-asset momentum
    try:
        btc  = yf.download('BTC-USD', start=start, end=end,
                            progress=False)
        gold = yf.download('GC=F',    start=start, end=end,
                            progress=False)
        if isinstance(btc.columns,  pd.MultiIndex):
            btc.columns  = btc.columns.droplevel(1)
        if isinstance(gold.columns, pd.MultiIndex):
            gold.columns = gold.columns.droplevel(1)
        btc_c  = btc['Close'].reindex(df.index,  method='ffill')
        gold_c = gold['Close'].reindex(df.index, method='ffill')
        btc_ma20  = btc_c.rolling(20,  min_periods=5).mean()
        gold_ma20 = gold_c.rolling(20, min_periods=5).mean()
        df['btc_above_ma20']  = (btc_c  > btc_ma20 ).astype(float).fillna(0)
        df['gold_above_ma20'] = (gold_c > gold_ma20).astype(float).fillna(0)
        # Risk composite: 0=both below, 1=mixed, 2=both above
        df['risk_composite']  = df['btc_above_ma20'] + df['gold_above_ma20']
        # BTC 5-day momentum
        df['btc_mom5'] = btc_c.pct_change(5).reindex(
            df.index, method='ffill').fillna(0)
    except Exception:
        df['btc_above_ma20']  = 0.5
        df['gold_above_ma20'] = 0.5
        df['risk_composite']  = 1.0
        df['btc_mom5']        = 0.0

    return df


def add_sentiment_stub(df: pd.DataFrame,
                       sentiment_mean: float = 0.05,
                       sentiment_trend: float = 0.0) -> pd.DataFrame:
    """Adds constant sentiment columns as a stub (fallback only)."""
    df = df.copy()
    df['sentiment_mean']  = sentiment_mean
    df['sentiment_trend'] = sentiment_trend
    df['headline_count']  = 30
    return df


def add_real_sentiment(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Loads real sentiment from sentiment_analyzer.py output (latest.json).
    Falls back to stub if file not found or symbol not present.
    """
    try:
        from src.utils.sentiment_loader import load_sentiment
        sent = load_sentiment(symbol)
        df = df.copy()
        df['sentiment_mean']  = sent['sentiment_mean']
        df['sentiment_trend'] = sent['sentiment_trend']
        df['headline_count']  = float(sent['headline_count'])
        return df
    except Exception as e:
        print(f"   ⚠️  Real sentiment load failed: {e} — using stub")
        return add_sentiment_stub(df)


# ==================================================================
# COMPONENT FACTORY
# ==================================================================

def build_regime_detector(name: str, cfg: dict):
    if name == 'score':
        from src.regime.score_regime import ScoreRegimeDetector
        return ScoreRegimeDetector()
    elif name == 'hmm':
        from src.regime.hmm_regime import HMMRegimeDetector
        n_states = cfg.get('hmm_states', 3)
        return HMMRegimeDetector(n_states=n_states)
    else:
        raise ValueError(f"Unknown regime detector: {name}")


def build_forecaster(name: str, cfg: dict):
    horizons = cfg.get('horizons', [5, 10, 15, 20])
    if name == 'linear':
        from src.forecast.linear_forecast import LinearForecaster
        return LinearForecaster(horizons=horizons)
    elif name == 'lgbm':
        from src.forecast.lgbm_forecast import LightGBMForecaster
        return LightGBMForecaster(horizons=horizons)
    else:
        raise ValueError(f"Unknown forecaster: {name}")


def build_trader(cfg: dict):
    trader_type = cfg.get('trader', 'rule')
    if trader_type == 'ensemble':
        from src.trading.ensemble_trader import EnsembleTrader
        return EnsembleTrader(
            risk_profile    = cfg.get('risk_profile', 'moderate'),
            portfolio_value = cfg.get('portfolio_value', 100_000.0),
            agreement_only  = cfg.get('agreement_only', True),
            lgbm_weight     = cfg.get('lgbm_weight', 0.7)
        )
    from src.trading.rule_trader import RuleTrader
    return RuleTrader(
        risk_profile    = cfg.get('risk_profile', 'moderate'),
        portfolio_value = cfg.get('portfolio_value', 100_000.0)
    )


# ==================================================================
# FEATURE MATRIX BUILDERS
# ==================================================================

SCORE_FEATURE_COLS = [
    'Close', 'RSI', 'MACD', 'MACD_Signal',
    'SMA_20', 'SMA_50', 'SMA_200',
    'sentiment_mean', 'sentiment_trend', 'headline_count'
]

LINEAR_FEATURE_COLS = [
    'Returns', 'Price_MA5', 'Price_MA20', 'Price_MA50',
    'Volatility', 'RSI', 'MACD', 'Signal',
    'Volume_Trend', 'High_Low_Range', 'Open_Close_Range'
]


def get_score_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SCORE_FEATURE_COLS if c in df.columns]
    return df[cols].copy()


def get_linear_features(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in LINEAR_FEATURE_COLS if c in df.columns]
    return df[cols].fillna(0).values


def get_lgbm_features(df: pd.DataFrame) -> np.ndarray:
    from src.forecast.lgbm_forecast import LightGBMForecaster
    feat_df = LightGBMForecaster.engineer_features(df)
    return feat_df.fillna(0).values


# ==================================================================
# SINGLE WALK-FORWARD WINDOW
# ==================================================================

def run_window(train_df: pd.DataFrame,
               test_df:  pd.DataFrame,
               regime_name: str,
               forecast_name: str,
               cfg: dict,
               window_idx: int,
               cost_bps: float = 10.0) -> Dict[str, Any]:
    """
    Fit components on train_df, simulate trading on test_df.
    Returns a dict of per-window results.
    """
    print(f"\n   Window {window_idx}: "
          f"train={len(train_df)}d  test={len(test_df)}d  "
          f"({train_df.index[0].date()} → {test_df.index[-1].date()})")

    # ---- 1. Regime detector ----------------------------------------
    regime_det = build_regime_detector(regime_name, cfg)
    score_train = get_score_features(train_df)
    regime_det.fit(score_train)

    # Label entire dataset (train+test) for feature injection
    full_df   = pd.concat([train_df, test_df])
    score_full = get_score_features(full_df)
    regime_labels = regime_det.predict_series(score_full)
    full_df = full_df.copy()
    full_df['regime'] = regime_labels.values

    train_with_regime = full_df.iloc[:len(train_df)]
    test_with_regime  = full_df.iloc[len(train_df):]

    # ---- 2. Forecaster ---------------------------------------------
    forecaster = build_forecaster(forecast_name, cfg)
    horizons   = cfg.get('horizons', [5, 10, 15, 20])

    if forecast_name == 'linear':
        X_train = get_linear_features(train_with_regime)
        y_train = train_with_regime['Close'].values
    else:
        X_train = get_lgbm_features(train_with_regime)
        y_train = train_with_regime['Close'].values

    forecaster.fit(X_train, y_train)

    # ---- 3. Trader -------------------------------------------------
    trader = build_trader(cfg)

    # ---- 4. Simulate test period -----------------------------------
    y_true_list, y_pred_list, y_prev_list = [], [], []
    equity_curve   = [cfg.get('portfolio_value', 100_000.0)]
    daily_returns  = []
    trade_returns  = []

    primary_horizon = horizons[0]   # Use 5-day as primary trading signal

    for i in range(len(test_with_regime)):
        row        = test_with_regime.iloc[i]
        current_px = float(row['Close'])
        regime_lbl = str(row.get('regime', 'sideways'))

        # Build feature window: all train data + test rows up to i
        window_df = pd.concat([
            train_with_regime,
            test_with_regime.iloc[:i+1]
        ])

        if forecast_name == 'linear':
            X_window = get_linear_features(window_df)
        else:
            X_window = get_lgbm_features(window_df)

        # Predict
        try:
            pred_meta = forecaster.predict_with_meta(X_window, current_px)
        except Exception as e:
            pred_meta = {h: {'predicted_price': current_px,
                             'expected_return': 0.0} for h in horizons}

        pred_5d    = pred_meta[primary_horizon]['predicted_price']
        exp_ret    = pred_meta[primary_horizon]['expected_return']
        confidence = max(0.3, 1 - float(row.get('Volatility', 0.02)) * 2)

        # Price range estimate
        vol        = float(row.get('Volatility', 0.01)) or 0.01
        std_move   = current_px * vol * np.sqrt(primary_horizon / 252)
        price_range = (pred_5d - std_move, pred_5d + std_move)

        # HMM confidence: posterior probability of most likely state
        try:
            if hasattr(regime_det, 'model_') and hasattr(regime_det, 'scaler_'):
                X_conf = regime_det._build_features(window_df)
                X_conf = regime_det.scaler_.transform(X_conf)
                probs  = regime_det.model_.predict_proba(X_conf)
                hmm_confidence = float(np.max(probs[-1]))
            else:
                hmm_confidence = 0.6
        except Exception:
            hmm_confidence = 0.6

        # Score-regime direction estimate using ScoreRegimeDetector
        # This gives us the baseline direction signal for agreement filtering
        lookback_slice = window_df['Returns'].iloc[-20:] if len(window_df) >= 20 else window_df['Returns']
        mean_ret_score = float(lookback_slice.mean()) if len(lookback_slice) > 0 else 0.0
        pred_score     = current_px * (1 + mean_ret_score * 5)
        exp_ret_score  = mean_ret_score * 5



        # Trading signal — clean, no agreement filtering
        state = {
            'symbol':               cfg.get('symbol', 'ASSET'),
            'current_price':        current_px,
            'predicted_price':      pred_5d,
            'predicted_price_lgbm': pred_5d,
            'predicted_price_score':pred_score,
            'expected_return':      exp_ret,
            'expected_return_lgbm': exp_ret,
            'expected_return_score':exp_ret_score,
            'confidence':           confidence,
            'price_range':          price_range,
            'regime':               regime_lbl,
            'regime_confidence':    0.6,
        }
        signal = trader.generate_signal(state)

        # Simulate P&L: use next-day return as proxy
        if i + 1 < len(test_with_regime):
            next_px      = float(test_with_regime.iloc[i+1]['Close'])
            actual_ret   = (next_px - current_px) / current_px
        else:
            actual_ret   = 0.0

        # Position-adjusted return
        position = signal['position_size']
        if 'BUY' in signal['signal']:
            trade_ret = actual_ret * position
        elif 'SELL' in signal['signal']:
            trade_ret = -actual_ret * position
        else:
            trade_ret = 0.0

        prev_equity = equity_curve[-1]
        new_equity  = prev_equity * (1 + trade_ret)
        equity_curve.append(new_equity)
        daily_returns.append(trade_ret)

        # ATR-based volatility targeting: scale position by volatility
        atr_val = float(row.get('Volatility', 0.01)) or 0.01
        target_risk  = 0.01
        atr_scale    = float(np.clip(target_risk / (atr_val + 1e-8), 0.3, 2.0))
        max_pos      = 0.20  # aggressive cap
        position     = float(np.clip(
            position * (0.5 + 0.5 * atr_scale), 0, max_pos
        ))

        # Apply transaction cost (both sides: entry + exit = 2x)
        if signal['signal'] != 'HOLD':
            cost = (cost_bps / 10000.0) * 2 * position
            trade_ret -= cost
        if signal['signal'] != 'HOLD':
            trade_returns.append(trade_ret)

        # Forecast accuracy tracking (5-day horizon)
        future_idx = i + primary_horizon
        if future_idx < len(test_with_regime):
            y_true_list.append(
                float(test_with_regime.iloc[future_idx]['Close'])
            )
            y_pred_list.append(pred_5d)
            y_prev_list.append(current_px)

    # ---- 5. Metrics ------------------------------------------------
    metrics = summarise(
        y_true        = np.array(y_true_list),
        y_pred        = np.array(y_pred_list),
        y_prev        = np.array(y_prev_list),
        equity_curve  = np.array(equity_curve),
        daily_returns = np.array(daily_returns),
        trade_returns = np.array(trade_returns),
        label         = f"window_{window_idx}"
    )
    metrics['window']      = window_idx
    metrics['train_size']  = len(train_df)
    metrics['test_size']   = len(test_df)
    metrics['train_start'] = str(train_df.index[0].date())
    metrics['test_end']    = str(test_df.index[-1].date())

    print(f"   📊 Sharpe={metrics['sharpe_ratio']:.2f}  "
          f"Return={metrics['total_return']:.1%}  "
          f"MAE={metrics['forecast_mae']:.2f}  "
          f"DirAcc={metrics['directional_accuracy']:.1%}")

    return metrics


# ==================================================================
# MAIN BACKTEST LOOP
# ==================================================================

def run_backtest(config_path: str):
    """Load config and run full walk-forward backtest."""

    with open(config_path) as f:
        cfg = json.load(f)

    exp_name      = cfg.get('name', 'experiment')
    symbol        = cfg.get('symbol', '^GSPC')
    regime_name   = cfg.get('regime', 'score')
    forecast_name = cfg.get('forecaster', 'linear')
    train_days    = cfg.get('train_window_days', 200)
    test_days     = cfg.get('test_window_days',  50)
    step_days     = cfg.get('step_days',         50)
    data_period   = cfg.get('data_period',       '2y')
    cost_bps      = cfg.get('cost_bps',          10.0)

    print(f"\n{'='*70}")
    print(f"BACKTEST: {exp_name}")
    print(f"  Symbol    : {symbol}")
    print(f"  Regime    : {regime_name}")
    print(f"  Forecaster: {forecast_name}")
    print(f"  Windows   : train={train_days}d / test={test_days}d / step={step_days}d")
    print(f"{'='*70}")

    # Fetch and prepare data
    df = fetch_data(symbol, period=data_period)
    df = add_technical_features(df)
    use_real_sentiment = cfg.get('use_real_sentiment', True)
    if use_real_sentiment:
        print("   📰 Loading real sentiment from latest.json...")
        df = add_real_sentiment(df, symbol)
    else:
        df = add_sentiment_stub(
            df,
            sentiment_mean  = cfg.get('sentiment_mean',  0.05),
            sentiment_trend = cfg.get('sentiment_trend', 0.0)
        )
    df.dropna(subset=['Returns', 'RSI', 'MACD'], inplace=True)

    # Add cross-asset features (VIX stress ratio + BTC/Gold momentum)
    print("   📡 Fetching cross-asset features (VIX, BTC, Gold)...")
    df = add_cross_asset_features(df, symbol)

    n = len(df)
    if n < train_days + test_days:
        print(f"❌ Not enough data: {n} rows < {train_days + test_days} required")
        sys.exit(1)

    # Walk-forward windows
    all_results = []
    window_idx  = 0
    start       = 0

    while start + train_days + test_days <= n:
        train_df = df.iloc[start : start + train_days]
        test_df  = df.iloc[start + train_days : start + train_days + test_days]

        result = run_window(
            train_df, test_df,
            regime_name, forecast_name,
            cfg, window_idx,
            cost_bps=cost_bps
        )
        all_results.append(result)

        start      += step_days
        window_idx += 1

    if not all_results:
        print("❌ No windows completed.")
        sys.exit(1)

    # Aggregate across windows
    metrics_keys = ['forecast_mae', 'forecast_rmse', 'directional_accuracy',
                    'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']

    aggregated = {'label': exp_name, 'n_windows': len(all_results)}
    for k in metrics_keys:
        vals = [r[k] for r in all_results if r.get(k) is not None
                and not np.isnan(r[k])]
        aggregated[k + '_mean'] = float(np.mean(vals))  if vals else np.nan
        aggregated[k + '_std']  = float(np.std(vals))   if vals else np.nan

    print(f"\n{'='*70}")
    print(f"AGGREGATED RESULTS — {exp_name}")
    print(f"{'='*70}")
    for k in metrics_keys:
        m = aggregated.get(k + '_mean', np.nan)
        s = aggregated.get(k + '_std',  np.nan)
        print(f"  {k:25s}: {m:+.4f}  (±{s:.4f})")

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = exp_name.replace(' ', '_').replace('+', 'plus')

    json_path = os.path.join(OUTPUT_DIR,
                             f"results_{safe_name}_{ts}.json")
    with open(json_path, 'w') as f:
        json.dump({'config': cfg,
                   'aggregated': aggregated,
                   'windows': all_results}, f, indent=2, default=str)
    print(f"\n✅ Results saved → {json_path}")

    csv_path = os.path.join(OUTPUT_DIR,
                            f"results_{safe_name}_{ts}.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"✅ CSV saved      → {csv_path}")

    return aggregated


# ==================================================================
# ENTRY POINT
# ==================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Walk-forward backtester for AI trading pipeline'
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to experiment config JSON (e.g. configs/baseline.json)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)

    run_backtest(args.config)
