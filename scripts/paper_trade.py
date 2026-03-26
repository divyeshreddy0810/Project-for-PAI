#!/usr/bin/env python3
"""
Paper Trader — Daily Simulation
--------------------------------
Simulates "if you traded today" using your live pipeline.

Flow:
  1. Select assets
  2. Enter portfolio value
  3. Select algorithm
  4. Fetches live prices + latest sentiment
  5. Runs regime detection + forecasting + signal generation
  6. Prints per-asset report: entry, TP, SL, position, simulated P&L

Usage:
  python3 scripts/paper_trade.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf

# ── Asset catalogue ─────────────────────────────────────────────────
ASSETS = [
    # Indices
    {"idx": 1,  "symbol": "^GSPC",    "name": "S&P 500"},
    {"idx": 2,  "symbol": "^IXIC",    "name": "NASDAQ"},
    {"idx": 3,  "symbol": "^DJI",     "name": "Dow Jones"},
    # Stocks
    {"idx": 4,  "symbol": "AAPL",     "name": "Apple"},
    {"idx": 5,  "symbol": "MSFT",     "name": "Microsoft"},
    {"idx": 6,  "symbol": "NVDA",     "name": "NVIDIA"},
    {"idx": 7,  "symbol": "TSLA",     "name": "Tesla"},
    {"idx": 8,  "symbol": "AMZN",     "name": "Amazon"},
    {"idx": 9,  "symbol": "META",     "name": "Meta"},
    {"idx": 10, "symbol": "GOOGL",    "name": "Alphabet"},
    {"idx": 11, "symbol": "JPM",      "name": "JP Morgan"},
    # Crypto
    {"idx": 12, "symbol": "BTC-USD",  "name": "Bitcoin"},
    {"idx": 13, "symbol": "ETH-USD",  "name": "Ethereum"},
    {"idx": 14, "symbol": "SOL-USD",  "name": "Solana"},
    {"idx": 15, "symbol": "BNB-USD",  "name": "BNB"},
    {"idx": 16, "symbol": "XRP-USD",  "name": "XRP"},
    # Forex Major
    {"idx": 17, "symbol": "EURUSD=X", "name": "EUR/USD"},
    {"idx": 18, "symbol": "GBPUSD=X", "name": "GBP/USD"},
    {"idx": 19, "symbol": "USDJPY=X", "name": "USD/JPY"},
    {"idx": 20, "symbol": "AUDUSD=X", "name": "AUD/USD"},
    {"idx": 21, "symbol": "USDCAD=X", "name": "USD/CAD"},
    # African Forex
    {"idx": 22, "symbol": "USDNGN=X", "name": "USD/NGN"},
    {"idx": 23, "symbol": "EURNGN=X", "name": "EUR/NGN"},
    {"idx": 24, "symbol": "USDZAR=X", "name": "USD/ZAR"},
    {"idx": 25, "symbol": "USDKES=X", "name": "USD/KES"},
    # Commodities
    {"idx": 26, "symbol": "GC=F",     "name": "Gold"},
    {"idx": 27, "symbol": "SI=F",     "name": "Silver"},
    {"idx": 28, "symbol": "CL=F",     "name": "Crude Oil"},
    {"idx": 29, "symbol": "NG=F",     "name": "Natural Gas"},
    {"idx": 30, "symbol": "HG=F",     "name": "Copper"},
]

ALGORITHMS = {
    "1": {
        "name":    "Baseline",
        "regime":  "score",
        "forecast":"linear",
        "trader":  "rule",
        "desc":    "Score-based regime + linear regression. Fast, interpretable."
    },
    "2": {
        "name":    "HMM + LightGBM",
        "regime":  "hmm",
        "forecast":"lgbm",
        "trader":  "rule",
        "desc":    "HMM regime detection + LightGBM forecaster. Best Sharpe (2.45)."
    },
    "3": {
        "name":    "Ensemble",
        "regime":  "hmm",
        "forecast":"lgbm",
        "trader":  "ensemble",
        "desc":    "HMM + LightGBM + ensemble signal blending. Balanced approach."
    },
}

RISK_PARAMS = {
    'conservative': {'max_position_size': 0.05, 'stop_loss': 0.05, 'take_profit': 0.10},
    'moderate':     {'max_position_size': 0.10, 'stop_loss': 0.08, 'take_profit': 0.15},
    'aggressive':   {'max_position_size': 0.20, 'stop_loss': 0.10, 'take_profit': 0.25},
}


# ── Helpers ──────────────────────────────────────────────────────────

def sep(char="─", n=70):
    print(char * n)

def header(title):
    sep("═")
    print(f"  {title}")
    sep("═")

def section(title):
    print(f"\n{title}")
    sep()


def fetch_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    return df


def add_features(df: pd.DataFrame, sentiment_mean: float,
                 sentiment_trend: float) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    df["Returns"]       = close.pct_change()
    df["Price_MA5"]     = close.rolling(5,  min_periods=2).mean()
    df["Price_MA20"]    = close.rolling(20, min_periods=5).mean()
    df["Price_MA50"]    = close.rolling(50, min_periods=10).mean()
    df["SMA_20"]        = df["Price_MA20"]
    df["SMA_50"]        = df["Price_MA50"]
    df["SMA_200"]       = close.rolling(200, min_periods=30).mean()
    df["Volatility"]    = df["Returns"].rolling(14, min_periods=3).std()

    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"]           = (100 - (100 / (1 + gain / loss.replace(0, np.nan)))).fillna(50)

    ema12           = close.ewm(span=12, adjust=False).mean()
    ema26           = close.ewm(span=26, adjust=False).mean()
    df["MACD"]      = ema12 - ema26
    df["Signal"]    = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Signal"] = df["Signal"]

    vol_ma              = df["Volume"].rolling(20, min_periods=5).mean()
    df["Volume_Trend"]  = (df["Volume"] / vol_ma.replace(0, np.nan)).fillna(1.0)
    df["High_Low_Range"]    = (df["High"] - df["Low"]) / close.replace(0, np.nan)
    df["Open_Close_Range"]  = (close - df["Open"]) / df["Open"].replace(0, np.nan)

    df["sentiment_mean"]  = sentiment_mean
    df["sentiment_trend"] = sentiment_trend
    df["headline_count"]  = 30.0
    return df


def get_sentiment(symbol: str) -> dict:
    try:
        from src.utils.sentiment_loader import load_sentiment
        return load_sentiment(symbol)
    except Exception:
        return {"sentiment_mean": 0.05, "sentiment_trend": 0.0,
                "headline_count": 0, "source": "stub"}


def run_regime(regime_name: str, df: pd.DataFrame) -> str:
    if regime_name == "score":
        from src.regime.score_regime import ScoreRegimeDetector
        det = ScoreRegimeDetector()
        det.fit(df)
        return det.predict(df)
    else:
        from src.regime.hmm_regime import HMMRegimeDetector
        det = HMMRegimeDetector(n_states=3)
        if len(df) < 30:
            return "sideways"
        det.fit(df)
        return det.predict(df)


def run_forecast(forecast_name: str, df: pd.DataFrame,
                 current_px: float) -> dict:
    horizons = [5, 10, 15, 20]
    if forecast_name == "linear":
        from src.forecast.linear_forecast import LinearForecaster
        LINEAR_COLS = ["Returns","Price_MA5","Price_MA20","Price_MA50",
                       "Volatility","RSI","MACD","Signal",
                       "Volume_Trend","High_Low_Range","Open_Close_Range"]
        cols  = [c for c in LINEAR_COLS if c in df.columns]
        X     = df[cols].fillna(0).values
        y     = df["Close"].values
        fc    = LinearForecaster(horizons=horizons)
        if len(X) < fc.lookback + 1:
            return {h: {"predicted_price": current_px, "expected_return": 0.0} for h in horizons}
        fc.fit(X, y)
        return fc.predict_with_meta(X, current_px)
    else:
        from src.forecast.lgbm_forecast import LightGBMForecaster
        fc    = LightGBMForecaster(horizons=horizons)
        feat  = LightGBMForecaster.engineer_features(df).fillna(0)
        X     = feat.values
        y     = df["Close"].values
        if len(X) < max(horizons) + 20:
            return {h: {"predicted_price": current_px, "expected_return": 0.0} for h in horizons}
        fc.fit(X, y)
        return fc.predict_with_meta(X, current_px)


def run_trader(trader_name: str, state: dict,
               risk_profile: str, portfolio: float) -> dict:
    if trader_name == "ensemble":
        from src.trading.ensemble_trader import EnsembleTrader
        t = EnsembleTrader(risk_profile=risk_profile,
                           portfolio_value=portfolio,
                           agreement_only=False)
    else:
        from src.trading.rule_trader import RuleTrader
        t = RuleTrader(risk_profile=risk_profile,
                       portfolio_value=portfolio)
    return t.generate_signal(state)


# ── Selection helpers ─────────────────────────────────────────────────

def pick_assets() -> list:
    section("AVAILABLE ASSETS")
    for a in ASSETS:
        print(f"  {a['idx']:>2}. {a['symbol']:<12} {a['name']}")
    print("\n  Enter numbers separated by commas, or 'all'")
    raw = input("  → ").strip()
    if raw.lower() == "all":
        return ASSETS
    chosen = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            match = [a for a in ASSETS if a["idx"] == idx]
            if match:
                chosen.append(match[0])
    if not chosen:
        print("  ⚠️  No valid assets — defaulting to S&P 500")
        return [ASSETS[0]]
    return chosen


def pick_portfolio() -> float:
    section("PORTFOLIO VALUE")
    from src.utils.currency import get_all_rates, parse_amount_input
    rates = get_all_rates()
    print(f"  €1 = ₦{rates['EUR_NGN']:,.0f}  |  $1 = ₦{rates['USD_NGN']:,.0f}  |  €1 = ${rates['EUR_USD']:.2f}")
    raw = input("  Enter amount (e.g. ₦100000 / €500 / $300): ").strip()
    try:
        eur_val, orig_amt, orig_ccy, port_obj = parse_amount_input(raw, rates)
        print(f"  ✅ {port_obj.display()}")
        return eur_val
    except Exception:
        print("  ⚠️  Invalid — defaulting to €1,000")
        return 1000.0


def pick_algorithm() -> dict:
    section("SELECT ALGORITHM")
    for k, v in ALGORITHMS.items():
        print(f"  {k}. {v['name']}")
        print(f"     {v['desc']}")
        print()
    raw = input("  Enter choice (1/2/3, default=2): ").strip() or "2"
    return ALGORITHMS.get(raw, ALGORITHMS["2"])


def pick_risk() -> str:
    section("RISK PROFILE")
    print("  1. Conservative  (5%  max position, 5%  SL, 10% TP)")
    print("  2. Moderate      (10% max position, 8%  SL, 15% TP)")
    print("  3. Aggressive    (20% max position, 10% SL, 25% TP)")
    raw = input("\n  Enter choice (1/2/3, default=2): ").strip() or "2"
    return {"1": "conservative", "2": "moderate", "3": "aggressive"}.get(raw, "moderate")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    header("PAPER TRADER  ·  DAILY SIMULATION")
    print("  Simulates 'if you traded today' using your live pipeline.")
    print("  Uses real-time prices + latest sentiment data.\n")

    assets    = pick_assets()
    portfolio = pick_portfolio()
    algo      = pick_algorithm()
    risk      = pick_risk()
    rp        = RISK_PARAMS[risk]

    print(f"\n  ✅ Portfolio: €{portfolio:,.2f}")
    print(f"  ✅ Algorithm: {algo['name']}")
    print(f"  ✅ Risk:      {risk.capitalize()}")
    print(f"  ✅ Assets:    {', '.join(a['symbol'] for a in assets)}")

    section("RUNNING ANALYSIS  (this may take ~60 seconds)")

    results = []

    for asset in assets:
        symbol = asset["symbol"]
        name   = asset["name"]
        print(f"\n  [{symbol}] Fetching data...")

        # 1. Price data
        try:
            df = fetch_data(symbol, period="8mo")
            if df.empty or len(df) < 50:
                print(f"  ⚠️  [{symbol}] Not enough data — skipping")
                continue
        except Exception as e:
            print(f"  ⚠️  [{symbol}] Data fetch failed: {e}")
            continue

        current_px  = float(df["Close"].iloc[-1])
        open_px     = float(df["Open"].iloc[-1])
        prev_close  = float(df["Close"].iloc[-2]) if len(df) > 1 else current_px
        day_chg_pct = (current_px - prev_close) / prev_close * 100

        # 2. Sentiment
        sent = get_sentiment(symbol)
        print(f"  [{symbol}] Sentiment: {sent['sentiment_mean']:+.3f}  "
              f"({sent['headline_count']} headlines)")

        # 3. Features
        df = add_features(df, sent["sentiment_mean"], sent["sentiment_trend"])
        df.dropna(subset=["Returns", "RSI"], inplace=True)

        # 4. Regime
        print(f"  [{symbol}] Running regime detection ({algo['regime']})...")
        try:
            regime = run_regime(algo["regime"], df)
        except Exception as e:
            print(f"  ⚠️  [{symbol}] Regime failed: {e} — defaulting to sideways")
            regime = "sideways"

        # 5. Forecast
        print(f"  [{symbol}] Running forecast ({algo['forecast']})...")
        try:
            forecast = run_forecast(algo["forecast"], df, current_px)
        except Exception as e:
            print(f"  ⚠️  [{symbol}] Forecast failed: {e}")
            forecast = {h: {"predicted_price": current_px,
                            "expected_return": 0.0} for h in [5,10,15,20]}

        pred_5d    = forecast[5]["predicted_price"]
        exp_ret    = forecast[5]["expected_return"]
        confidence = max(0.3, 1 - float(df["Volatility"].iloc[-1] or 0.02) * 2)
        vol        = float(df["Volatility"].iloc[-1] or 0.01)
        std_move   = current_px * vol * np.sqrt(5 / 252)
        price_range= (pred_5d - std_move, pred_5d + std_move)

        # 6. Signal
        state = {
            "symbol":               symbol,
            "current_price":        current_px,
            "predicted_price":      pred_5d,
            "predicted_price_lgbm": pred_5d,
            "predicted_price_score":current_px * (1 + exp_ret),
            "expected_return":      exp_ret,
            "expected_return_lgbm": exp_ret,
            "expected_return_score":exp_ret,
            "confidence":           confidence,
            "price_range":          price_range,
            "regime":               regime,
            "regime_confidence":    0.6,
        }
        signal = run_trader(algo["trader"], state, risk, portfolio)

        # 7. Paper P&L simulation
        # Entry = today's open, current = live price
        entry_px    = open_px
        paper_ret   = (current_px - entry_px) / entry_px if entry_px else 0.0
        pos_size    = signal["position_size"]
        pos_value   = pos_size * portfolio
        raw_pnl     = pos_value * paper_ret
        cost        = pos_value * (10 / 10000) * 2   # 10 bps both sides
        net_pnl     = raw_pnl - cost

        # TP / SL status
        tp_price    = signal.get("take_profit", current_px * 1.15)
        sl_price    = signal.get("stop_loss",   current_px * 0.92)
        in_trade    = signal["signal"] != "HOLD"
        tp_hit      = in_trade and (current_px >= tp_price if "BUY" in signal["signal"] else current_px <= tp_price)
        sl_hit      = in_trade and (current_px <= sl_price if "BUY" in signal["signal"] else current_px >= sl_price)
        if not in_trade:
            tp_sl_status = "⏸️  NO TRADE"
        elif tp_hit:
            tp_sl_status = "🟢 TP HIT"
        elif sl_hit:
            tp_sl_status = "🔴 SL HIT"
        else:
            tp_sl_status = "⏳ IN PLAY"

        results.append({
            "symbol":       symbol,
            "name":         name,
            "current_px":   current_px,
            "open_px":      open_px,
            "day_chg_pct":  day_chg_pct,
            "regime":       regime.upper(),
            "signal":       signal["signal"],
            "confidence":   signal["confidence"],
            "pos_size_pct": pos_size * 100,
            "pos_value":    pos_value,
            "tp_price":     tp_price,
            "sl_price":     sl_price,
            "pred_5d":      pred_5d,
            "exp_ret":      exp_ret * 100,
            "paper_ret":    paper_ret * 100,
            "net_pnl":      net_pnl,
            "tp_sl_status": tp_sl_status,
        })

    if not results:
        print("\n❌ No results generated.")
        return

    # ── Report ────────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n\n")
    header(f"PAPER TRADING REPORT  ·  {now}")
    print(f"  Algorithm : {algo['name']}")
    print(f"  Portfolio : €{portfolio:,.2f}")
    print(f"  Risk      : {risk.capitalize()}")
    print(f"  Timeframe : Daily (today's open → current price)")

    total_pnl       = sum(r["net_pnl"] for r in results)
    total_deployed  = sum(r["pos_value"] for r in results)
    total_deployed_pct = total_deployed / portfolio * 100

    for r in results:
        section(f"{r['symbol']}  —  {r['name']}")
        print(f"  Live Price   : €{r['current_px']:>12,.2f}   "
              f"(Day: {r['day_chg_pct']:+.2f}%)")
        print(f"  Today's Open : €{r['open_px']:>12,.2f}")
        print(f"  Regime       : {r['regime']}")
        print(f"  Signal       : {r['signal']:<12}  "
              f"(confidence: {r['confidence']:.0%})")
        print(f"  5-Day Pred   : €{r['pred_5d']:>12,.2f}   "
              f"(expected: {r['exp_ret']:+.2f}%)")
        print()
        print(f"  Position Size: {r['pos_size_pct']:.1f}% of portfolio")
        print(f"  Position €   : €{r['pos_value']:>10,.2f}")
        print(f"  Take Profit  : €{r['tp_price']:>12,.2f}")
        print(f"  Stop Loss    : €{r['sl_price']:>12,.2f}")
        print(f"  Status       : {r['tp_sl_status']}")
        print()
        pnl_sign = "+" if r["net_pnl"] >= 0 else ""
        print(f"  Paper Return : {r['paper_ret']:+.3f}%  "
              f"(open→now)")
        print(f"  Net P&L      : {pnl_sign}€{r['net_pnl']:,.2f}  "
              f"(after 10 bps costs)")

    sep("═")
    print(f"\n  PORTFOLIO SUMMARY")
    sep()
    print(f"  Total deployed : €{total_deployed:,.2f}  "
          f"({total_deployed_pct:.1f}% of portfolio)")
    pnl_sign = "+" if total_pnl >= 0 else ""
    print(f"  Total net P&L  : {pnl_sign}€{total_pnl:,.2f}")
    print(f"  Portfolio now  : €{portfolio + total_pnl:,.2f}")
    pct_chg = total_pnl / portfolio * 100
    print(f"  Change         : {pct_chg:+.3f}%")
    sep("═")
    print()

    # Save report
    os.makedirs("data/output", exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"data/output/paper_trade_{ts}.json"
    with open(path, "w") as f:
        json.dump({
            "timestamp":  now,
            "algorithm":  algo["name"],
            "portfolio":  portfolio,
            "risk":       risk,
            "results":    results,
            "summary": {
                "total_pnl":      total_pnl,
                "total_deployed": total_deployed,
                "portfolio_after":portfolio + total_pnl
            }
        }, f, indent=2, default=str)
    print(f"  ✅ Report saved → {path}\n")


if __name__ == "__main__":
    main()
