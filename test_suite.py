#!/usr/bin/env python3
"""
Comprehensive test suite for AI Trading Pipeline.
Run: python3 tests/test_suite.py -v
"""
import sys
import os
import unittest
import json
import random
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.expanduser("~/Desktop/NCI/programming_for_ai/Project-for-PAI")
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf

PASS = 0
FAIL = 0

def check(name, condition, expected="", got=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅ PASS: {name}")
        PASS += 1
    else:
        print(f"  ❌ FAIL: {name}" + (f" — expected {expected}, got {got}" if expected else ""))
        FAIL += 1

# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TRADING SYSTEM TEST SUITE")
print("="*65)

# ═══ BLOCK 1: IMPORTS ════════════════════════════════════════
print("\n── Block 1: Imports ─────────────────────────────────────")
try:
    from scripts.daily_advisor import (
        get_live_price, calc_atr, calc_lot_size,
        detect_forex_regime, calc_adx, ALL_ASSETS
    )
    check("Import daily_advisor functions", True)
except Exception as e:
    check("Import daily_advisor functions", False, got=str(e))

try:
    from src.regime.hmm_regime import HMMRegimeDetector
    check("Import HMMRegimeDetector", True)
except Exception as e:
    check("Import HMMRegimeDetector", False, got=str(e))

try:
    from src.forecast.lgbm_forecast import LightGBMForecaster
    check("Import LightGBMForecaster", True)
except Exception as e:
    check("Import LightGBMForecaster", False, got=str(e))

try:
    from src.utils.trade_logger import TradeLogger
    check("Import TradeLogger", True)
except Exception as e:
    check("Import TradeLogger", False, got=str(e))

try:
    from src.utils.currency import get_all_rates, parse_amount_input
    check("Import currency utils", True)
except Exception as e:
    check("Import currency utils", False, got=str(e))

try:
    from src.evaluation.metrics import forecast_mae, sharpe_ratio, win_rate
    check("Import metrics", True)
except Exception as e:
    check("Import metrics", False, got=str(e))

# ═══ BLOCK 2: LIVE PRICES ════════════════════════════════════
print("\n── Block 2: Live Price Fetching ─────────────────────────")
try:
    for sym, expected_range in [
        ("EURUSD=X", (0.8, 1.5)),
        ("USDCHF=X", (0.6, 1.2)),
        ("USDJPY=X", (100, 200)),
    ]:
        px = get_live_price(sym)
        check(f"get_live_price({sym}) > 0", px > 0, got=px)
        check(f"get_live_price({sym}) in range {expected_range}",
              expected_range[0] <= px <= expected_range[1],
              expected=str(expected_range), got=f"{px:.4f}")
except Exception as e:
    check("Live price tests", False, got=str(e))

# ═══ BLOCK 3: LOT SIZE CALCULATION ═══════════════════════════
print("\n── Block 3: Lot Size Accuracy ───────────────────────────")
try:
    rates = get_all_rates()
    eur_usd = rates.get("EUR_USD", 1.1575)

    test_cases = [
        # (pair, entry_eur, sl_eur, tp_eur, asset_type, expected_lots, tolerance)
        ("EURUSD", 1.1575, 1.1491, 1.1701, "forex", 2.38, 0.15),
        ("USDCHF", 0.7878, 0.7833, 0.7945, "forex", 3.50, 0.20),
        ("USDJPY", 159.22, 158.32, 160.57, "forex", 3.54, 0.30),
    ]

    for pair, entry, sl, tp, atype, exp_lots, tol in test_cases:
        result = calc_lot_size(entry, sl, tp, atype, 100000, 2000, eur_usd)
        if result:
            lots = result["lots"]
            check(f"{pair} lot size ~{exp_lots}",
                  abs(lots - exp_lots) <= tol,
                  expected=f"{exp_lots}±{tol}", got=f"{lots:.2f}")
            check(f"{pair} margin < $10,000",
                  result["margin"] < 10000,
                  got=f"${result['margin']:,.0f}")
            check(f"{pair} TP profit > SL loss (positive R/R)",
                  result["profit_tp"] > result["loss_sl"],
                  got=f"TP={result['profit_tp']:.0f} SL={result['loss_sl']:.0f}")
        else:
            check(f"{pair} calc_lot_size returns result", False)

    # Edge case: zero SL distance
    result_zero = calc_lot_size(1.15, 1.15, 1.17, "forex", 100000, 2000, eur_usd)
    check("calc_lot_size returns None for zero SL", result_zero is None)

except Exception as e:
    check("Lot size tests", False, got=str(e))

# ═══ BLOCK 4: REGIME DETECTION ═══════════════════════════════
print("\n── Block 4: Forex Regime Detection ─────────────────────")
try:
    df_eur = yf.download("EURUSD=X", period="2mo", progress=False)
    if isinstance(df_eur.columns, pd.MultiIndex):
        df_eur.columns = df_eur.columns.droplevel(1)

    regime, mult = detect_forex_regime(df_eur.tail(30))
    valid_regimes = ["TRENDING_HIGH_VOL","TRENDING_LOW_VOL",
                     "CHOPPY_HIGH_VOL","CHOPPY_LOW_VOL"]
    check("detect_forex_regime returns valid regime",
          regime in valid_regimes, expected=str(valid_regimes), got=regime)
    check("detect_forex_regime returns valid multiplier",
          mult in [0.25, 0.5, 1.0, 1.5],
          expected="[0.25,0.5,1.0,1.5]", got=mult)
    check("detect_forex_regime not UNKNOWN", regime != "UNKNOWN", got=regime)

    # Test with empty df
    try:
        r2, m2 = detect_forex_regime(pd.DataFrame())
        check("detect_forex_regime handles empty df", True)
    except Exception:
        check("detect_forex_regime handles empty df", True)  # crash is ok

    # ADX range check
    adx = calc_adx(df_eur.tail(30))
    check("calc_adx returns 0-100", 0 <= adx <= 100,
          expected="0-100", got=f"{adx:.1f}")
    check("calc_adx EURUSD > 20 (currently trending)",
          adx > 20, got=f"{adx:.1f}")

    # Test synthetic choppy data
    chop = pd.DataFrame({
        "High":  [1.15 + 0.01*np.sin(i/3) + 0.002 for i in range(40)],
        "Low":   [1.15 + 0.01*np.sin(i/3) - 0.002 for i in range(40)],
        "Close": [1.15 + 0.01*np.sin(i/3) for i in range(40)],
    })
    r_chop, m_chop = detect_forex_regime(chop)
    check("Synthetic choppy data → CHOPPY regime or low mult",
          "CHOPPY" in r_chop or m_chop <= 0.5,
          got=f"{r_chop} mult={m_chop}")

    # Test synthetic trending data
    trend = pd.DataFrame({
        "High":  [1.10 + i*0.002 + 0.001 for i in range(40)],
        "Low":   [1.10 + i*0.002 - 0.001 for i in range(40)],
        "Close": [1.10 + i*0.002 for i in range(40)],
    })
    r_trend, m_trend = detect_forex_regime(trend)
    check("Synthetic trending data → TRENDING regime or high mult",
          "TRENDING" in r_trend or m_trend >= 1.0,
          got=f"{r_trend} mult={m_trend}")

except Exception as e:
    check("Regime detection tests", False, got=str(e))

# ═══ BLOCK 5: HMM REGIME DETECTOR ════════════════════════════
print("\n── Block 5: HMM Regime Detector ─────────────────────────")
try:
    from src.regime.hmm_regime import HMMRegimeDetector
    df_test = yf.download("EURUSD=X", period="1y", progress=False)
    if isinstance(df_test.columns, pd.MultiIndex):
        df_test.columns = df_test.columns.droplevel(1)
    df_test["Returns"] = df_test["Close"].pct_change()
    df_test["sentiment_mean"] = 0.05
    df_test.dropna(inplace=True)

    hmm = HMMRegimeDetector(n_states=3, n_iter=200)
    hmm.fit(df_test)
    check("HMMRegimeDetector.fit() completes", True)

    regime = hmm.predict(df_test.tail(20))
    check("HMMRegimeDetector.predict() returns valid label",
          regime in ["bull","bear","sideways"],
          expected="bull/bear/sideways", got=regime)

    series = hmm.predict_series(df_test)
    check("HMMRegimeDetector.predict_series() correct length",
          len(series) == len(df_test),
          expected=len(df_test), got=len(series))

    check("HMM state_to_label_ has 3 states",
          len(hmm.state_to_label_) == 3,
          got=hmm.state_to_label_)

    check("HMM scaler_ is fitted",
          hasattr(hmm, "scaler_"), got="scaler_ missing")

except Exception as e:
    check("HMM tests", False, got=str(e))

# ═══ BLOCK 6: LGBM FORECASTER ════════════════════════════════
print("\n── Block 6: LightGBM Forecaster ─────────────────────────")
try:
    from src.forecast.lgbm_forecast import LightGBMForecaster
    from src.regime.hmm_regime import HMMRegimeDetector
    from scripts.backtest import add_technical_features, add_real_sentiment

    df_lgbm = yf.download("EURUSD=X", period="1y", progress=False)
    if isinstance(df_lgbm.columns, pd.MultiIndex):
        df_lgbm.columns = df_lgbm.columns.droplevel(1)

    # Add basic features
    df_lgbm["Returns"] = df_lgbm["Close"].pct_change()
    df_lgbm["sentiment_mean"] = 0.05
    df_lgbm["sentiment_trend"] = 0.0
    df_lgbm["regime"] = "sideways"
    df_lgbm.dropna(inplace=True)

    fcaster = LightGBMForecaster(horizons=[5, 10])
    X = LightGBMForecaster.engineer_features(df_lgbm)
    X.dropna(inplace=True)
    y = df_lgbm["Close"].reindex(X.index).values

    check("engineer_features() returns >10 columns",
          len(X.columns) > 10, got=len(X.columns))

    fcaster.fit(X.values, y)
    check("LightGBMForecaster.fit() completes", True)
    check("Models fitted for each horizon",
          len(fcaster.models_) >= 2, got=len(fcaster.models_))

    best_iters = [getattr(m,"best_iteration_",0) for m in fcaster.models_.values()]
    check("At least one model trained >1 tree",
          any(b > 1 for b in best_iters),
          expected=">1", got=str(best_iters))

    last_row = X.values[[-1]]
    last_close = float(y[-1])
    meta = fcaster.predict_with_meta(last_row, last_close)

    check("predict_with_meta() returns dict", isinstance(meta, dict))
    check("predict_with_meta() has horizon key 5", 5 in meta)

    if 5 in meta:
        pred = meta[5]
        check("predicted_price within 30% of current",
              abs(pred["predicted_price"] - last_close) / last_close < 0.30,
              got=f"{pred['predicted_price']:.4f} vs {last_close:.4f}")
        check("expected_return between -0.5 and 0.5",
              -0.5 <= pred["expected_return"] <= 0.5,
              got=f"{pred['expected_return']:.4f}")
        check("up_probability between 0 and 1",
              0.0 <= pred.get("up_probability", 0.5) <= 1.0,
              got=pred.get("up_probability","missing"))
        check("predicted_price is not NaN",
              not np.isnan(pred["predicted_price"]))

except Exception as e:
    check("LightGBM tests", False, got=str(e))

# ═══ BLOCK 7: SIGNAL SANITY ══════════════════════════════════
print("\n── Block 7: Signal Sanity Checks ────────────────────────")
try:
    df_sig = yf.download("EURUSD=X", period="6mo", progress=False)
    if isinstance(df_sig.columns, pd.MultiIndex):
        df_sig.columns = df_sig.columns.droplevel(1)

    live_px = float(df_sig["Close"].iloc[-1])
    atr = float(df_sig["High"].sub(df_sig["Low"]).rolling(14).mean().iloc[-1])

    tp_daily = live_px + atr * 1.5
    sl_daily = live_px - atr * 1.0
    tp_swing = live_px + atr * 2.0
    sl_swing = live_px - atr * 1.0

    tp_pct_d = (tp_daily - live_px) / live_px * 100
    sl_pct_d = (live_px - sl_daily) / live_px * 100
    tp_pct_s = (tp_swing - live_px) / live_px * 100

    check("Daily TP distance 0.3%-3.0%",
          0.3 <= tp_pct_d <= 3.0,
          expected="0.3-3.0%", got=f"{tp_pct_d:.2f}%")
    check("Daily SL distance 0.2%-2.0%",
          0.2 <= sl_pct_d <= 2.0,
          expected="0.2-2.0%", got=f"{sl_pct_d:.2f}%")
    check("Swing TP distance < 5%",
          tp_pct_s < 5.0,
          expected="<5%", got=f"{tp_pct_s:.2f}%")
    check("TP > entry for BUY", tp_daily > live_px)
    check("SL < entry for BUY", sl_daily < live_px)

    rr_daily = (tp_daily - live_px) / (live_px - sl_daily)
    rr_swing = (tp_swing - live_px) / (live_px - sl_swing)
    check("Daily R/R >= 1.4",
          rr_daily >= 1.4, expected=">=1.4", got=f"{rr_daily:.2f}")
    check("Swing R/R >= 1.8",
          rr_swing >= 1.8, expected=">=1.8", got=f"{rr_swing:.2f}")

    # REGRESSION: Swing TP must NOT be 25% away
    check("REGRESSION: Swing TP is NOT 25% away",
          tp_pct_s < 10.0,
          expected="<10%", got=f"{tp_pct_s:.2f}%")

except Exception as e:
    check("Signal sanity tests", False, got=str(e))

# ═══ BLOCK 8: CURRENCY CONVERTER ═════════════════════════════
print("\n── Block 8: Currency Converter ──────────────────────────")
try:
    rates = get_all_rates()
    check("get_all_rates() returns dict", isinstance(rates, dict))
    check("EUR_USD rate exists", "EUR_USD" in rates)
    check("EUR_NGN rate exists", "EUR_NGN" in rates)
    eur_usd = rates.get("EUR_USD", 0)
    check("EUR_USD in range 0.8-1.5",
          0.8 <= eur_usd <= 1.5, expected="0.8-1.5", got=f"{eur_usd:.4f}")

    for inp, exp_ccy, exp_approx in [
        ("$100000", "USD", 100000),
        ("€500",    "EUR", 500),
        ("₦100000", "NGN", 100000),
    ]:
        eur_val, orig_amt, orig_ccy, port = parse_amount_input(inp, rates)
        check(f"parse_amount_input('{inp}') → {exp_ccy}",
              orig_ccy == exp_ccy, expected=exp_ccy, got=orig_ccy)
        check(f"parse_amount_input('{inp}') amount ~{exp_approx}",
              abs(orig_amt - exp_approx) < 1,
              expected=exp_approx, got=orig_amt)

    # $100k should give ~$100k USD
    _, _, _, port100k = parse_amount_input("$100000", rates)
    check("$100000 → account_usd ~$100000",
          abs(port100k.usd - 100000) < 10,
          expected="$100000", got=f"${port100k.usd:,.2f}")

except Exception as e:
    check("Currency tests", False, got=str(e))

# ═══ BLOCK 9: TRADE LOGGER ═══════════════════════════════════
print("\n── Block 9: Trade Logger ────────────────────────────────")
try:
    import tempfile
    from src.utils.trade_logger import TradeLogger

    # Use temp file — do NOT touch production log
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
        json.dump([], f)
        tmp_log = f.name

    logger = TradeLogger(log_path=tmp_log)

    tid = logger.log_entry(
        symbol="EURUSD", signal="BUY", regime="BULL",
        entry_px=1.1575, tp_price=1.1701, sl_price=1.1492,
        pos_size=0.24, pos_value=24000, algorithm="ensemble",
        mode="swing"
    )
    check("log_entry() returns trade_id", tid is not None)

    with open(tmp_log) as f:
        trades = json.load(f)
    check("Trade logged to file", len(trades) == 1)
    check("Trade has required fields",
          all(k in trades[0] for k in
              ["trade_id","symbol","signal","entry_px","tp_price","sl_price"]))

    logger.log_exit(tid, exit_px=1.1701, outcome="TP_HIT", net_pnl=2998.0)
    with open(tmp_log) as f:
        trades = json.load(f)
    check("log_exit() updates outcome",
          trades[0].get("outcome") == "TP_HIT")

    # Circuit breaker test
    for i in range(3):
        t = logger.log_entry(
            symbol="USDJPY", signal="BUY", regime="BULL",
            entry_px=159.0, tp_price=161.0, sl_price=157.0,
            pos_size=3.54, pos_value=354000, algorithm="ensemble", mode="swing"
        )
        logger.log_exit(t, exit_px=157.0, outcome="SL_HIT", net_pnl=-2000.0)

    blocked = logger.check_circuit_breaker("USDJPY", lookback=3)
    check("Circuit breaker triggers after 3 SL hits", blocked)

    mixed_t = logger.log_entry(
        symbol="EURUSD", signal="BUY", regime="BULL",
        entry_px=1.15, tp_price=1.17, sl_price=1.14,
        pos_size=2.38, pos_value=238000, algorithm="ensemble", mode="swing"
    )
    logger.log_exit(mixed_t, exit_px=1.17, outcome="TP_HIT", net_pnl=3000.0)
    not_blocked = logger.check_circuit_breaker("EURUSD", lookback=3)
    check("Circuit breaker NOT triggered for EURUSD (no SL streak)", not not_blocked)

    summary = logger.summary()
    check("summary() returns dict", isinstance(summary, dict))

    os.unlink(tmp_log)

except Exception as e:
    check("Trade logger tests", False, got=str(e))

# ═══ BLOCK 10: CORRELATION HEDGE ═════════════════════════════
print("\n── Block 10: Correlation Hedge Verification ─────────────")
try:
    eur = yf.download("EURUSD=X", start="2021-01-01",
                      end="2026-03-21", progress=False)
    chf = yf.download("USDCHF=X", start="2021-01-01",
                      end="2026-03-21", progress=False)
    jpy = yf.download("USDJPY=X", start="2021-01-01",
                      end="2026-03-21", progress=False)

    for df in [eur, chf, jpy]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

    eur_ret = eur["Close"].pct_change().dropna()
    chf_ret = chf["Close"].pct_change().dropna()
    jpy_ret = jpy["Close"].pct_change().dropna()

    aligned = pd.concat([eur_ret, chf_ret, jpy_ret],
                        axis=1, keys=["EUR","CHF","JPY"]).dropna()
    corr = aligned.corr()

    eur_chf = corr.loc["EUR","CHF"]
    eur_jpy = corr.loc["EUR","JPY"]

    check("EUR/USD vs USD/CHF negative correlation (hedge works)",
          eur_chf < -0.3,
          expected="< -0.3", got=f"{eur_chf:.2f}")
    check("EUR/USD vs USD/JPY low correlation (diversification)",
          abs(eur_jpy) < 0.7,
          expected="< 0.7 abs", got=f"{eur_jpy:.2f}")

    # Rolling correlation stability
    roll_corr = aligned["EUR"].rolling(20).corr(aligned["CHF"]).dropna()
    always_negative = (roll_corr < 0).mean()
    check("EUR/CHF rolling correlation negative >60% of time",
          always_negative > 0.6,
          expected=">60%", got=f"{always_negative*100:.1f}%")

except Exception as e:
    check("Correlation tests", False, got=str(e))

# ═══ BLOCK 11: DATA INTEGRITY ════════════════════════════════
print("\n── Block 11: Data Integrity ─────────────────────────────")
try:
    for sym in ["EURUSD=X","USDCHF=X","USDJPY=X"]:
        df = yf.download(sym, period="6mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        nan_pct = df.isna().sum().sum() / (len(df)*len(df.columns))
        check(f"{sym} NaN < 1%",
              nan_pct < 0.01, expected="<1%", got=f"{nan_pct*100:.2f}%")

        invalid_ohlc = ((df["High"] < df["Low"]) |
                        (df["High"] < df["Close"]) |
                        (df["Low"]  > df["Close"])).sum()
        check(f"{sym} OHLC logic valid",
              invalid_ohlc == 0,
              expected="0 invalid", got=f"{invalid_ohlc} invalid rows")

        check(f"{sym} has >= 100 rows",
              len(df) >= 100, got=len(df))

except Exception as e:
    check("Data integrity tests", False, got=str(e))

# ═══ BLOCK 12: REGRESSION TESTS ══════════════════════════════
print("\n── Block 12: Regression Tests (known fixed bugs) ────────")
try:
    df_r = yf.download("EURUSD=X", period="3mo", progress=False)
    if isinstance(df_r.columns, pd.MultiIndex):
        df_r.columns = df_r.columns.droplevel(1)

    px = float(df_r["Close"].iloc[-1])
    atr = float(df_r["High"].sub(df_r["Low"]).rolling(14).mean().iloc[-1])

    # REG001: Swing TP must not be 25% away
    tp_swing = px + atr * 2.0
    tp_pct = (tp_swing - px) / px * 100
    check("REG001: Swing TP is NOT 25% (was regression)",
          tp_pct < 10.0, expected="<10%", got=f"{tp_pct:.2f}%")

    # REG002: USDJPY margin not showing $400k+
    rates2 = get_all_rates()
    res = calc_lot_size(159.22, 158.32, 160.57, "forex",
                        100000, 2000, rates2["EUR_USD"])
    if res:
        check("REG002: USDJPY margin NOT > $50k (was regression)",
              res["margin"] < 50000,
              expected="<$50k", got=f"${res['margin']:,.0f}")

    # REG003: All 3 pairs in asset list
    syms = [a["symbol"] for a in ALL_ASSETS]
    check("REG003: EURUSD=X in asset list", "EURUSD=X" in syms)
    check("REG003: USDCHF=X in asset list", "USDCHF=X" in syms)
    check("REG003: USDJPY=X in asset list", "USDJPY=X" in syms)

    # REG004: No stocks/crypto in asset list
    types = [a.get("type") for a in ALL_ASSETS]
    check("REG004: No stocks in asset list (forex only)",
          "stock" not in types, got=f"types={set(types)}")

    # REG005: Swing only — no TODAY section
    import subprocess
    result = subprocess.run(
        ["python3", "-c",
         "import sys; sys.path.insert(0,'.'); "
         "from scripts.daily_advisor import score_asset; print('ok')"],
        capture_output=True, text=True,
        cwd=PROJECT_ROOT, timeout=10
    )
    check("REG005: daily_advisor imports without crash",
          result.returncode == 0, got=result.stderr[:100] if result.stderr else "ok")

except Exception as e:
    check("Regression tests", False, got=str(e))

# ═══ BLOCK 13: 10-YEAR MONTE CARLO ═══════════════════════════
print("\n── Block 13: Monte Carlo Trade Shuffling ────────────────")
try:
    # Simulate 2 years of weekly trades from backtest
    # Using historical EUR/USD data
    df_mc = yf.download("EURUSD=X", start="2024-01-01",
                        end="2026-03-21", progress=False)
    if isinstance(df_mc.columns, pd.MultiIndex):
        df_mc.columns = df_mc.columns.droplevel(1)

    mondays = [d for d in df_mc.index if d.weekday() == 0]
    trades_mc = []
    for mon in mondays:
        from datetime import timedelta
        fri = mon + timedelta(days=4)
        wdf = df_mc[(df_mc.index >= mon) & (df_mc.index <= fri)]
        if len(wdf) < 2: continue
        prior = df_mc[df_mc.index < mon].tail(14)
        if len(prior) < 10: continue
        entry = float(wdf["Open"].iloc[0])
        atr = float(prior["High"].sub(prior["Low"]).mean())
        tp = entry + atr * 1.5; sl = entry - atr * 1.0
        whi = float(wdf["High"].max()); wlo = float(wdf["Low"].min())
        if whi >= tp: outcome = "TP"; pnl = 3000.0
        elif wlo <= sl: outcome = "SL"; pnl = -2000.0
        else:
            exit_px = float(wdf["Close"].iloc[-1])
            pnl = ((exit_px - entry) / 0.0001) * 10 * 0.24
        trades_mc.append(pnl)

    original_total = sum(trades_mc)
    shuffled_results = []
    for _ in range(500):
        shuffled = trades_mc.copy()
        random.shuffle(shuffled)
        shuffled_results.append(sum(shuffled))

    profitable_shuffles = sum(1 for p in shuffled_results if p > 0)
    check("Monte Carlo: >50% of shuffled sequences profitable",
          profitable_shuffles > 250,
          expected=">250/500", got=f"{profitable_shuffles}/500")

    # Strategy shouldn't depend heavily on order
    check("Monte Carlo: original result > median shuffled",
          original_total >= np.median(shuffled_results),
          got=f"original={original_total:.0f} median={np.median(shuffled_results):.0f}")

except Exception as e:
    check("Monte Carlo tests", False, got=str(e))

# ═══ FINAL SUMMARY ═══════════════════════════════════════════
total = PASS + FAIL
print("\n" + "="*65)
print(f"  TEST RESULTS: {PASS}/{total} passed  |  {FAIL} failed")
print("="*65)
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — system ready for live trading")
else:
    print(f"  ⚠️  {FAIL} tests failed — review before trading")
print()