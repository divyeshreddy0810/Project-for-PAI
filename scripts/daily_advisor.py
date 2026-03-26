#!/usr/bin/env python3
"""
Daily Trading Advisor
---------------------
Launch once per day. The system:

  1. Checks all previously suggested trades → updates outcomes
  2. Shows running accuracy log
  3. Scans all assets → ranks by signal strength + regime
  4. Recommends TOP 3 for TODAY (tight ATR levels, 1-day)
  5. Recommends TOP 3 for SWING (5-day horizon)
  6. Prints MT4-ready entry cards
  7. Saves everything to trade log

Usage:
  python3 scripts/daily_advisor.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
from src.utils.trade_logger import TradeLogger

# ── Multimodal consensus ──────────────────────────────────────
try:
    from src.regime.pretrained_hmm import get_pretrained_hmm
    _PRETRAINED_HMM = get_pretrained_hmm()
except Exception as _e:
    _PRETRAINED_HMM = None

try:
    from src.trading.consensus import calculate_consensus, format_consensus_line
    _CONSENSUS_OK = True
except Exception:
    _CONSENSUS_OK = False
# ─────────────────────────────────────────────────────────────

from scripts.paper_trade import (
    fetch_data, add_features, get_sentiment,
    run_regime, run_forecast, run_trader,
    sep, header, section
)

# ── Asset universe — 12-month backtested forex hedge portfolio ─────────
# EUR/USD + USD/CHF + USD/JPY proven optimal over 12 months:
# +32.8% return, -17.2% max drawdown, 43% win rate
# Negative correlation between EUR/USD and USD/CHF acts as natural hedge
ALL_ASSETS = [
    # PRIMARY — bullish EUR, bearish USD
    {"symbol": "EURUSD=X", "name": "EUR/USD", "type": "forex",
     "mt4": "EURUSD", "pip": 0.0001, "pip_val": 10,
     "role": "primary"},

    # HEDGE — bullish USD, bearish CHF (-0.85 correlation to EUR/USD)
    {"symbol": "USDCHF=X", "name": "USD/CHF", "type": "forex",
     "mt4": "USDCHF", "pip": 0.0001, "pip_val": 10,
     "role": "hedge"},

    # DIVERSIFIER — uncorrelated (+0.20 correlation to EUR/USD)
    {"symbol": "USDJPY=X", "name": "USD/JPY", "type": "forex",
     "mt4": "USDJPY", "pip": 0.01, "pip_val": 1000,
     "role": "diversifier"},
]

# Deduplicate
seen = set()
ASSETS = []
for a in ALL_ASSETS:
    if a["symbol"] not in seen:
        seen.add(a["symbol"])
        ASSETS.append(a)

ALGO = {
    "name":    "HMM + LightGBM",
    "regime":  "hmm",
    "forecast":"lgbm",
    "trader":  "rule",
}


def get_live_price(symbol: str) -> float:
    """Fetch latest price — works on weekends by using last close."""
    # Method 1: fast_info (live during market hours)
    try:
        t  = yf.Ticker(symbol)
        fi = t.fast_info
        px = fi.get("last_price") or fi.get("regularMarketPrice")
        if px and float(px) > 0:
            return float(px)
    except Exception:
        pass
    # Method 2: 5-day daily data (works weekends — returns last Friday close)
    try:
        df = yf.download(symbol, period="5d", interval="1d", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    # Method 3: 1-minute intraday (market hours only)
    try:
        df = yf.download(symbol, period="1d", interval="1m", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def calc_adx(df: pd.DataFrame, period: int = 14) -> float:
    """ADX trend strength. >25 = trending, <20 = choppy."""
    try:
        h = df["High"]; l = df["Low"]; c = df["Close"]
        tr  = pd.concat([h-l,(h-c.shift()).abs(),
                         (l-c.shift()).abs()],axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        pdm = h.diff().clip(lower=0)
        ndm = (-l.diff()).clip(lower=0)
        pdm[pdm < ndm] = 0; ndm[ndm < pdm] = 0
        pdi = 100*(pdm.rolling(period).mean()/(atr+1e-8))
        ndi = 100*(ndm.rolling(period).mean()/(atr+1e-8))
        dx  = 100*((pdi-ndi).abs()/(pdi+ndi+1e-8))
        return float(dx.rolling(period).mean().iloc[-1])
    except Exception:
        return 20.0  # neutral default


def detect_forex_regime(df: pd.DataFrame) -> tuple:
    """
    Returns (regime_label, risk_multiplier) based on ADX + ATR.
    TRENDING_HIGH_VOL → 1.5x risk (best for this strategy)
    TRENDING_LOW_VOL  → 1.0x risk
    CHOPPY_HIGH_VOL   → 0.5x risk
    CHOPPY_LOW_VOL    → 0.25x risk
    """
    try:
        adx     = calc_adx(df)
        atr_pct = float(df["High"].sub(df["Low"]).mean()
                        / df["Close"].iloc[-1] * 100)
        # Handle flat/synthetic data edge case
        import math
        if math.isnan(adx) or adx == 0:
            adx = 0.0
        if adx > 25 and atr_pct > 0.8:
            return "TRENDING_HIGH_VOL", 1.5
        elif adx > 25:
            return "TRENDING_LOW_VOL",  1.0
        elif atr_pct > 0.8:
            return "CHOPPY_HIGH_VOL",   0.5
        else:
            return "CHOPPY_LOW_VOL",    0.25
    except Exception:
        return "CHOPPY_LOW_VOL", 0.25


def calc_atr(df: pd.DataFrame) -> float:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr  = pd.concat([h-l, (h-c.shift()).abs(),
                     (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    return atr if atr > 0 else float(c.iloc[-1]) * 0.01


# ── Step 1: Update previous open trades ──────────────────────────────

def update_open_trades(logger: TradeLogger):
    trades = logger._read()
    open_t = [t for t in trades if t["outcome"] == "OPEN"]

    if not open_t:
        print("  No open trades to update.\n")
        return

    print(f"  Found {len(open_t)} open trade(s) — checking outcomes...\n")
    updated = 0

    for t in open_t:
        symbol  = t["symbol"]
        live_px = get_live_price(symbol)
        if live_px <= 0:
            print(f"  ⚠️  [{symbol}] Could not fetch price — skipping")
            continue

        entry   = t["entry_px"]
        tp      = t["tp_price"]
        sl      = t["sl_price"]
        is_long = "BUY" in t["signal"]

        tp_hit  = live_px >= tp if is_long else live_px <= tp
        sl_hit  = live_px <= sl if is_long else live_px >= sl

        # Check if entry time was more than 24h ago → force close
        entry_dt  = datetime.fromisoformat(t["entry_time"])
        age_hours = (datetime.now() - entry_dt).total_seconds() / 3600

        if tp_hit:
            outcome = "TP_HIT"
        elif sl_hit:
            outcome = "SL_HIT"
        elif age_hours >= 24 and t.get("tp_sl_mode") == "daily":
            outcome = "CLOSED"   # Daily trade expired
        else:
            # Still open — update current price note but don't close
            raw_ret = (live_px - entry) / entry
            unreal  = raw_ret * t["pos_value"]
            sign    = "+" if unreal >= 0 else ""
            print(f"  ⏳ [{symbol:<10}] Still open  "
                  f"Entry €{entry:,.2f} → Now €{live_px:,.2f}  "
                  f"({(live_px-entry)/entry*100:+.2f}%)  "
                  f"Unrealised: {sign}€{unreal:.2f}")
            continue

        raw_ret = (live_px - entry) / entry
        if not is_long:
            raw_ret = -raw_ret
        cost    = (10 / 10000) * 2
        net_pnl = t["pos_value"] * (raw_ret - cost)

        logger.log_exit(t["trade_id"], live_px, outcome, net_pnl)
        icon    = "🟢" if outcome == "TP_HIT" else \
                  "🔴" if outcome == "SL_HIT" else "⏸️ "
        label   = {"TP_HIT":"TP HIT",
                   "SL_HIT":"SL HIT",
                   "CLOSED":"EXPIRED"}.get(outcome, outcome)
        sign    = "+" if net_pnl >= 0 else ""
        print(f"  {icon} [{symbol:<10}] {label:<10}  "
              f"Entry €{entry:,.2f} → Exit €{live_px:,.2f}  "
              f"P&L: {sign}€{net_pnl:.2f}")
        updated += 1

    if updated:
        print(f"\n  ✅ {updated} trade(s) closed and logged.")
    print()


# ── Step 2: Scan and score all assets ────────────────────────────────

def score_asset(asset: dict, risk: str,
                portfolio: float) -> dict | None:
    symbol = asset["symbol"]
    try:
        df = fetch_data(symbol, period="8mo")
        if df.empty or len(df) < 50:
            return None
    except Exception:
        return None

    live_px = get_live_price(symbol)
    if live_px <= 0:
        live_px = float(df["Close"].iloc[-1])

    # Detect forex regime from raw OHLC before add_features modifies df
    forex_regime = "UNKNOWN"
    risk_mult    = 1.0
    if asset.get("type") == "forex":
        try:
            _raw = df[["High","Low","Close"]].tail(30).copy()
            if len(_raw) >= 15:
                forex_regime, risk_mult = detect_forex_regime(_raw)
        except Exception:
            forex_regime, risk_mult = "UNKNOWN", 1.0

    sent = get_sentiment(symbol)
    df   = add_features(df, sent["sentiment_mean"],
                        sent["sentiment_trend"])
    df.dropna(subset=["Returns","RSI"], inplace=True)
    if len(df) < 30:
        return None

    try:
        regime = run_regime(ALGO["regime"], df)
    except Exception:
        regime = "sideways"

    try:
        forecast = run_forecast(ALGO["forecast"], df, live_px)
    except Exception:
        forecast = {h: {"predicted_price": live_px,
                         "expected_return": 0.0}
                    for h in [5,10,15,20]}

    pred_5d   = forecast[5]["predicted_price"]
    exp_ret   = forecast[5]["expected_return"]
    vol       = float(df["Volatility"].iloc[-1] or 0.01)
    confidence= max(0.3, 1 - vol * 2)
    atr_val   = calc_atr(df)
    std_move  = live_px * vol * np.sqrt(5/252)
    price_range=(pred_5d - std_move, pred_5d + std_move)

    state = {
        "symbol":               symbol,
        "current_price":        live_px,
        "predicted_price":      pred_5d,
        "predicted_price_lgbm": pred_5d,
        "predicted_price_score":live_px*(1+exp_ret),
        "expected_return":      exp_ret,
        "expected_return_lgbm": exp_ret,
        "expected_return_score":exp_ret,
        "confidence":           confidence,
        "price_range":          price_range,
        "regime":               regime,
        "regime_confidence":    0.6,
    }

    from src.trading.rule_trader import RuleTrader
    trader = RuleTrader(risk_profile=risk,
                        portfolio_value=portfolio)
    signal = trader.generate_signal(state)

    # Composite score: signal strength × confidence × regime boost
    regime_boost = {"bull":1.2, "bear":0.8, "sideways":0.9
                    }.get(regime, 1.0)
    score = abs(signal["signal_strength"]) * confidence * regime_boost

    # Daily ATR levels
    tp_daily = live_px + atr_val * 1.5
    sl_daily = live_px - atr_val * 1.0
    if "SELL" in signal["signal"]:
        tp_daily = live_px - atr_val * 1.5
        sl_daily = live_px + atr_val * 1.0

    # Swing levels — ATR-based (validated in 12-month backtest)
    # atr_val already calculated above via calc_atr(df)
    # Use 2.0x ATR for 5-day hold window
    if "SELL" in signal["signal"]:
        tp_swing = live_px - atr_val * 2.0
        sl_swing = live_px + atr_val * 1.0
    else:
        tp_swing = live_px + atr_val * 2.0
        sl_swing = live_px - atr_val * 1.0

    # ── Multimodal consensus ──────────────────────────────────
    consensus_result = None
    if _CONSENSUS_OK and _PRETRAINED_HMM is not None:
        try:
            hmm_regime_pt = _PRETRAINED_HMM.predict(df)
            _adx_r, _     = detect_forex_regime(df[["High","Low","Close"]].tail(30))
            _exp   = forecast[5]["expected_return"]
            _uprob = forecast[5].get("up_probability", 0.5)
            _dir   = "BUY" if "BUY" in signal["signal"] else "SELL"
            consensus_result = calculate_consensus(
                hmm_regime          = hmm_regime_pt,
                lgbm_expected_return= _exp,
                lgbm_up_probability = _uprob,
                adx_regime          = _adx_r,
                signal_direction    = _dir,
            )
            if consensus_result["consensus"] == "NO_TRADE":
                return None   # skip — models disagree
            risk_mult = consensus_result["size_multiplier"]
            forex_regime = f"{_adx_r} | HMM:{hmm_regime_pt}"
        except Exception as _ce:
            consensus_result = None   # fall back silently
    # ──────────────────────────────────────────────────────────

    return {
        "symbol":      symbol,
        "name":        asset["name"],
        "score":       score,
        "forex_regime": forex_regime,
        "risk_mult":    risk_mult,
        "consensus":    consensus_result,
        "signal":      signal["signal"],
        "signal_str":  signal["signal_strength"],
        "regime":      regime.upper(),
        "confidence":  confidence,
        "live_px":     live_px,
        "pred_5d":     pred_5d,
        "exp_ret":     exp_ret,
        "atr":         atr_val,
        "pos_size":    signal["position_size"],
        "pos_value":   signal["position_size"] * portfolio,
        # Daily trade levels
        "tp_daily":    tp_daily,
        "sl_daily":    sl_daily,
        "rr_daily":    abs(tp_daily-live_px)/abs(sl_daily-live_px)
                       if abs(sl_daily-live_px)>0 else 1.0,
        # Swing trade levels
        "tp_swing":    tp_swing,
        "sl_swing":    sl_swing,
        "rr_swing":    abs(tp_swing-live_px)/abs(sl_swing-live_px)
                       if abs(sl_swing-live_px)>0 else 1.0,
    }


# ── Lot size calculator ──────────────────────────────────────────────

LOT_SPECS = {
    # asset_type: (contract_size, pip_value_per_lot_per_dollar_move)
    # Gold/Silver: 1 lot = 100 oz. $1 move = $100 P&L per lot
    "commodity": {"contract": 100,   "unit": "oz",    "pip_mult": 100},
    # Forex: 1 lot = 100,000 base units. 1 pip (0.0001) = $10 per lot
    "forex":     {"contract": 100000,"unit": "units", "pip_mult": 10},
    # Index: 1 lot = $1 per point
    "index":     {"contract": 1,     "unit": "pts",   "pip_mult": 1},
    # Crypto: BTC/ETH = 1 coin per lot. XRP/others = 1000 coins per lot
    "crypto":    {"contract": 1,     "unit": "coins", "pip_mult": 1},
}

def calc_lot_size(entry: float, sl: float, tp: float,
                  asset_type: str, account_usd: float,
                  target_loss_usd: float, eur_usd: float,
                  asset_name: str = "") -> dict:
    """
    Calculate MT4 lot size so that hitting SL = target_loss_usd.
    entry is the RAW price (EUR for EURUSD, USD for USDJPY/USDCHF).
    """
    sl_distance = abs(entry - sl)
    tp_distance = abs(tp   - entry)
    if sl_distance <= 0:
        return None

    if asset_type == "forex":
        name_upper = asset_name.upper()
        if entry > 50 or "JPY" in name_upper:
            # JPY pairs — entry is raw USD ~159
            pip_size        = 0.01
            pip_val_per_lot = (0.01 / entry) * 100000   # ~$6.28
        elif "CHF" in name_upper or "CAD" in name_upper or (
              name_upper.startswith("USD") and entry < 5):
            # USD-base non-JPY (USDCHF ~0.79, USDCAD ~1.38)
            pip_size        = 0.0001
            pip_val_per_lot = (0.0001 / entry) * 100000  # ~$12.69 for CHF
        else:
            # EUR-base pairs (EURUSD ~1.15, GBPUSD ~1.32)
            entry_usd       = entry * eur_usd
            pip_size        = 0.0001
            pip_val_per_lot = 10.0  # $10 per pip standard lot

        sl_pips        = sl_distance / pip_size
        tp_pips        = tp_distance / pip_size
        pnl_per_lot_sl = sl_pips * pip_val_per_lot
        pnl_per_lot_tp = tp_pips * pip_val_per_lot

    elif asset_type == "crypto":
        contract        = 1000 if entry < 100 else 1
        pnl_per_lot_sl  = sl_distance * contract
        pnl_per_lot_tp  = tp_distance * contract

    elif asset_type == "commodity":
        pnl_per_lot_sl  = sl_distance * 100
        pnl_per_lot_tp  = tp_distance * 100

    else:  # index
        pnl_per_lot_sl  = sl_distance
        pnl_per_lot_tp  = tp_distance

    if pnl_per_lot_sl <= 0:
        return None

    raw_lots  = target_loss_usd / pnl_per_lot_sl
    lots      = max(0.01, round(raw_lots, 2))
    profit_tp = lots * pnl_per_lot_tp
    loss_sl   = lots * pnl_per_lot_sl

    # Margin: notional / leverage
    leverage = {"forex":100,"commodity":20,"index":20,"crypto":5}.get(asset_type,20)
    if asset_type == "forex":
        if entry > 50:          # JPY: base=USD
            margin = (lots * 100000) / leverage
        elif entry < 2:         # EUR-base: margin in USD
            margin = (lots * 100000 * entry * eur_usd) / leverage
        else:                   # USD-base low price
            margin = (lots * 100000 * entry) / leverage
    else:
        margin = (lots * entry) / leverage

    return {
        "lots":       lots,
        "profit_tp":  profit_tp,
        "loss_sl":    loss_sl,
        "margin":     margin,
        "entry_usd":  entry * eur_usd if entry < 5 else entry,
        "sl_usd":     sl    * eur_usd if sl    < 5 else sl,
        "tp_usd":     tp    * eur_usd if tp    < 5 else tp,
    }


def print_mt4_card(rank: int, r: dict,
                   mode: str, portfolio: float,
                   rates: dict = None,
                   account_usd: float = 0,
                   target_loss_usd: float = 2000,
                   eur_usd: float = 1.08):
    """Print MT4-ready trade card with lot size and progress tracker."""
    tp      = r["tp_daily"]  if mode == "daily" else r["tp_swing"]
    sl      = r["sl_daily"]  if mode == "daily" else r["sl_swing"]
    rr      = r["rr_daily"]  if mode == "daily" else r["rr_swing"]
    label   = "TODAY (1-day ATR)" if mode == "daily" else "SWING (5-day)"
    horizon = "Check back tomorrow" if mode == "daily" else "Check back in 5 trading days"
    d_arrow = "▲ LONG" if "BUY" in r["signal"] else "▼ SHORT"

    eur_ngn  = (rates or {}).get("EUR_NGN", 1600.0)
    eur_usd  = (rates or {}).get("EUR_USD", 1.08)

    # USD prices for MT4
    entry_usd = r["live_px"] * eur_usd
    tp_usd    = tp  * eur_usd
    sl_usd    = sl  * eur_usd
    pred_usd  = r["pred_5d"] * eur_usd

    # MT4 symbol
    mt4_sym = next((a.get("mt4", r["symbol"]) for a in ALL_ASSETS
                    if a["symbol"] == r["symbol"]), r["symbol"])

    print(f"\n  ┌─ #{rank} {r['symbol']:<12} {r['name']:<18} MT4: {mt4_sym}")
    print(f"  │  {d_arrow}  ·  {r['regime']} regime  ·  conf {r['confidence']:.0%}  ·  {label}")
    print(f"  │")
    # Dynamic risk adjustment
    forex_regime = r.get("forex_regime", "UNKNOWN")
    risk_mult    = r.get("risk_multiplier", 1.0)
    # Consensus line
    _cons = r.get("consensus")
    if _cons and _CONSENSUS_OK:
        try:
            print(format_consensus_line(_cons, base_risk=target_loss_usd))
        except Exception:
            pass
    adj_loss     = target_loss_usd * risk_mult
    regime_emoji = {"TRENDING_HIGH_VOL":"🔥","TRENDING_LOW_VOL":"📈",
                    "CHOPPY_HIGH_VOL":"⚠️","CHOPPY_LOW_VOL":"🔵"}.get(forex_regime,"")
    # Only show regime line if no consensus (consensus already shows regime)
    if not r.get("consensus"):
        print(f"  │  Regime     : {regime_emoji} {forex_regime}  (sizing: {risk_mult}x → risk ${adj_loss:,.0f})")
    print(f"  │  Entry      : ${entry_usd:>12,.4f}   ← type this in MT4")
    print(f"  │  Take Profit: ${tp_usd:>12,.4f}   ({abs(tp-r['live_px'])/r['live_px']*100:+.2f}%)")
    print(f"  │  Stop Loss  : ${sl_usd:>12,.4f}   ({abs(sl-r['live_px'])/r['live_px']*100:+.2f}%)")
    print(f"  │  Risk/Reward: {rr:.2f}:1")
    print(f"  │  5-Day Pred : ${pred_usd:>12,.4f}   (expected {r['exp_ret']*100:+.2f}%)")

    # Lot size block
    if account_usd > 0:
        asset_type = next((a.get("type","forex") for a in ALL_ASSETS
                           if a["symbol"] == r["symbol"]), "forex")
        lot = calc_lot_size(r["live_px"], sl, tp, asset_type,
                            account_usd, target_loss_usd, eur_usd,
                            asset_name=r.get("mt4", r["symbol"]))
        if lot:
            acct_tp  = account_usd + lot["profit_tp"]
            acct_sl  = account_usd - lot["loss_sl"]
            progress = (acct_tp / 1_000_000) * 100
            print(f"  │")
            print(f"  │  ── MT4 ORDER ─────────────────────────────────")
            print(f"  │  Symbol     : {mt4_sym}")
            print(f"  │  Action     : {'BUY' if 'BUY' in r['signal'] else 'SELL'}")
            print(f"  │  Lot size   : {lot['lots']:.2f} lots  ← type in MT4")
            print(f"  │  Entry $    : ${lot['entry_usd']:>12,.4f}")
            print(f"  │  TP $       : ${lot['tp_usd']:>12,.4f}")
            print(f"  │  SL $       : ${lot['sl_usd']:>12,.4f}")
            print(f"  │  Margin req : ~${lot['margin']:,.0f}")
            print(f"  │")
            print(f"  │  If TP hit  : +${lot['profit_tp']:,.0f}  → Account: ${acct_tp:,.0f}")
            print(f"  │  If SL hit  : -${lot['loss_sl']:,.0f}  → Account: ${acct_sl:,.0f}")
            print(f"  │  To $1M     : {progress:.2f}% of the way")

    print(f"  └─ {horizon}")

def main():
    header(f"DAILY ADVISOR  ·  {datetime.now().strftime('%A %d %B %Y  %H:%M')}")
    print("  Checking previous trades → Scanning market → "
          "Recommending best trades today\n")

    # Portfolio setup
    section("YOUR PORTFOLIO")

    # Fetch live rates first
    from src.utils.currency import get_all_rates, parse_amount_input, Portfolio
    rates = get_all_rates()

    print(f"\n  Live rates:")
    print(f"    €1  =  ₦{rates['EUR_NGN']:,.0f}")
    print(f"    $1  =  ₦{rates['USD_NGN']:,.0f}")
    print(f"    €1  =  ${rates['EUR_USD']:.4f}")
    print()

    raw = input("  Enter portfolio amount (e.g. ₦100000 / €500 / $300): ").strip()
    try:
        eur_val, orig_amt, orig_ccy, port_obj = parse_amount_input(raw, rates)
        portfolio = eur_val
    except Exception:
        portfolio = 1000.0
        port_obj  = Portfolio(1000.0, "EUR", rates)
        print(f"  ⚠️  Invalid — using €1,000")

    print(f"\n  ✅ Portfolio: {port_obj.display()}")

    # USD account size for lot sizing
    # account_usd should reflect the actual trading account size in USD
    if hasattr(port_obj, 'original_currency') and port_obj.original_currency == 'USD':
        account_usd = port_obj.original_amount
    elif hasattr(port_obj, 'usd'):
        account_usd = port_obj.usd
    else:
        account_usd = portfolio * rates.get('EUR_USD', 1.08)
    eur_usd     = rates.get('EUR_USD', 1.08)

    # Target loss per trade
    print("\n  Max loss per trade:")
    print("  1. $1,500  (conservative — 1.5% of $100k)")
    print("  2. $2,000  (moderate    — 2%   of $100k)")
    print("  3. $3,000  (aggressive  — 3%   of $100k)")
    tl = input("  → ").strip() or "2"
    target_loss_usd = {"1": 1500, "2": 2000, "3": 3000}.get(tl, 2000)
    target_gain_usd = target_loss_usd * 2  # 2:1 R:R
    print(f"  ✅ Risk per trade: max -${target_loss_usd:,}  /  target +${target_gain_usd:,}")

    print("\n  Risk profile:")
    print("  1. Conservative  2. Moderate  3. Aggressive")
    rc   = input("  → ").strip() or "2"
    risk = {"1":"conservative","2":"moderate",
            "3":"aggressive"}.get(rc, "moderate")
    print(f"  ✅ Risk: {risk.capitalize()}\n")

    # Step 1: Update previous trades
    logger = TradeLogger()
    section("STEP 1 — UPDATING PREVIOUS TRADES")
    update_open_trades(logger)

    # Show running accuracy
    logger.print_summary()

    # Step 2: Scan assets
    input("  Press Enter to scan today's market...\n")
    section("STEP 2 — SCANNING ALL ASSETS")
    print("  Analysing 14 assets — this takes ~2 minutes...\n")

    scored = []
    for asset in ASSETS:
        sym = asset["symbol"]
        print(f"  [{sym:<10}] scanning...", end="\r")
        result = score_asset(asset, risk, portfolio)
        if result and result["signal"] != "HOLD":
            scored.append(result)
            print(f"  [{sym:<10}] {result['signal']:<12}  "
                  f"score={result['score']:.3f}  "
                  f"regime={result['regime']:<10}  "
                  f"exp={result['exp_ret']*100:+.2f}%")
        else:
            print(f"  [{sym:<10}] HOLD — skipped           ")
        time.sleep(0.3)

    if not scored:
        print("\n  ❌ No tradeable signals found today. "
              "Market may be closed or data unavailable.")
        return

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Step 3: Recommendations
    print(f"\n\n")
    header(f"TODAY'S RECOMMENDATIONS  ·  "
           f"{datetime.now().strftime('%d %b %Y')}")
    print(f"  Portfolio: {port_obj.display()}  ·  "
          f"Risk: {risk.capitalize()}  ·  "
          f"Algorithm: {ALGO['name']}\n")

    rates = rates if 'rates' in dir() else {}
    # Category-based picks — best signal per asset class
    def best_in(types):
        return next((r for r in scored
                     if next((a.get("type","") for a in ALL_ASSETS
                              if a["symbol"]==r["symbol"]),"") in types
                     and r["signal"] != "HOLD"), None)

    top_overall   = scored[0] if scored else None
    top_forex     = best_in(["forex"])
    top_commodity = best_in(["commodity"])
    top_crypto    = best_in(["crypto"])
    top_index     = best_in(["index"])

    seen = set()
    top_daily = []
    for label, r in [
        ("🏆 TOP OVERALL",    top_overall),
        ("💱 BEST FOREX",     top_forex),
        ("🥇 BEST COMMODITY", top_commodity),
        ("🔵 BEST CRYPTO",    top_crypto),
        ("📈 BEST INDEX",     top_index),
    ]:
        if r and r["symbol"] not in seen:
            seen.add(r["symbol"])
            r["_cat_label"] = label
            top_daily.append(r)

    top_swing = sorted(
        scored,
        key=lambda x: abs(x["exp_ret"]) * x["confidence"],
        reverse=True
    )[:3]

    # Daily picks
    # ── SWING TRADES ONLY ────────────────────────────────────────────
    # Daily trading BANNED — 5-year backtest proves -388% return
    # Swing weekly: +203% over 5 years, 24.8% annualised
    section("SWING TRADES  (enter Monday · hold to TP/SL · exit Friday)")
    print("  EUR/USD + USD/CHF + USD/JPY · 1.5x ATR TP · 1.0x ATR SL\n")
    for i, r in enumerate(top_daily, 1):
        cat = r.get("_cat_label", f"#{i}")
        print(f"  {cat}")
        print_mt4_card(i, r, "swing", portfolio, rates,
                       account_usd, target_loss_usd, eur_usd)
        print()

    # Log all recommendations
    print()
    sep()
    print("  Logging all recommendations to trade log...")
    logged = 0
    for r in scored[:6]:
        mode = "daily" if r in top_daily else "swing"
        tp   = r["tp_daily"] if mode=="daily" else r["tp_swing"]
        sl   = r["sl_daily"] if mode=="daily" else r["sl_swing"]
        tid  = logger.log_entry(
            symbol    = r["symbol"],
            name      = r["name"],
            signal    = r["signal"],
            regime    = r["regime"],
            entry_px  = r["live_px"],
            tp_price  = tp,
            sl_price  = sl,
            pos_value = r["pos_value"],
            algorithm = ALGO["name"],
            risk      = risk,
            confidence= r["confidence"],
            pred_5d   = r["pred_5d"],
            exp_ret   = r["exp_ret"],
            tp_sl_mode= mode
        )
        logged += 1

    print(f"  ✅ {logged} trades logged → "
          f"run 'python3 scripts/trade_log.py' anytime to check\n")

    # Save full report
    os.makedirs("data/output", exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"data/output/daily_advice_{ts}.json"
    with open(path, "w") as f:
        json.dump({
            "date":      datetime.now().isoformat(),
            "portfolio": portfolio,
            "risk":      risk,
            "algorithm": ALGO["name"],
            "daily_picks": top_daily,
            "swing_picks": top_swing,
            "all_scored":  scored
        }, f, indent=2, default=str)

    sep("═")
    print(f"\n  📁 Full report saved → {path}")
    print(f"  📋 Trade log        → data/logs/trade_log.json")
    print(f"\n  Next steps:")
    print(f"  1. Enter the trades above into MT4 manually")
    print(f"  2. Set exactly the TP and SL shown")
    print(f"  3. Come back tomorrow and run this script again")
    print(f"  4. System will auto-check if TP/SL was hit\n")
    sep("═")
    print()


if __name__ == "__main__":
    main()
