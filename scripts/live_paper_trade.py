#!/usr/bin/env python3
"""
Live Paper Trader
-----------------
Runs full analysis once at start, then monitors live prices
every 60 seconds. Auto-closes positions when TP or SL is hit.
Runs until all positions are closed or you press Ctrl+C.

Usage:
  python3 scripts/live_paper_trade.py
  python3 scripts/live_paper_trade.py --interval 30   # check every 30s
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf

# ── reuse helpers from paper_trade.py ────────────────────────────────
from scripts.paper_trade import (
    ASSETS, ALGORITHMS, RISK_PARAMS,
    pick_assets, pick_portfolio, pick_algorithm, pick_risk,
    fetch_data, add_features, get_sentiment,
    run_regime, run_forecast, run_trader,
    sep, header, section
)


# ── Live price fetch ─────────────────────────────────────────────────

def get_live_price(symbol: str) -> float:
    """Fetch latest traded price using yfinance fast_info."""
    try:
        ticker = yf.Ticker(symbol)
        price  = ticker.fast_info.get("last_price") or \
                 ticker.fast_info.get("regularMarketPrice")
        if price and float(price) > 0:
            return float(price)
    except Exception:
        pass
    # Fallback: download 1-day 1-minute bar
    try:
        df = yf.download(symbol, period="1d", interval="1m", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def get_live_prices(symbols: list) -> dict:
    """Fetch live prices for multiple symbols."""
    prices = {}
    for sym in symbols:
        px = get_live_price(sym)
        prices[sym] = px
    return prices


# ── Position tracker ─────────────────────────────────────────────────

class Position:
    def __init__(self, symbol, name, entry_px, signal,
                 tp_price, sl_price, pos_value, pos_size_pct,
                 regime, confidence, pred_5d, exp_ret):
        self.symbol       = symbol
        self.name         = name
        self.entry_px     = entry_px
        self.signal       = signal        # BUY / SELL / STRONG_BUY etc
        self.tp_price     = tp_price
        self.sl_price     = sl_price
        self.pos_value    = pos_value
        self.pos_size_pct = pos_size_pct
        self.regime       = regime
        self.confidence   = confidence
        self.pred_5d      = pred_5d
        self.exp_ret      = exp_ret
        self.is_long      = "BUY" in signal
        self.is_hold      = signal == "HOLD"

        # Runtime state
        self.status       = "OPEN"   # OPEN / TP_HIT / SL_HIT / CLOSED
        self.exit_px      = None
        self.exit_time    = None
        self.pnl          = 0.0
        self.current_px   = entry_px
        self.peak_px      = entry_px
        self.trough_px    = entry_px

    def update(self, live_px: float, cost_bps: float = 10.0):
        """Check live price against TP/SL. Returns True if position closed."""
        if self.status != "OPEN":
            return False
        if self.is_hold or live_px <= 0:
            return False

        self.current_px = live_px
        self.peak_px    = max(self.peak_px,   live_px)
        self.trough_px  = min(self.trough_px, live_px)

        tp_hit = live_px >= self.tp_price if self.is_long else live_px <= self.tp_price
        sl_hit = live_px <= self.sl_price if self.is_long else live_px >= self.sl_price

        if tp_hit:
            self._close("TP_HIT", live_px, cost_bps)
            return True
        if sl_hit:
            self._close("SL_HIT", live_px, cost_bps)
            return True

        return False

    def _close(self, reason: str, exit_px: float, cost_bps: float):
        self.status    = reason
        self.exit_px   = exit_px
        self.exit_time = datetime.now()
        raw_ret        = (exit_px - self.entry_px) / self.entry_px
        if not self.is_long:
            raw_ret = -raw_ret
        cost           = (cost_bps / 10000.0) * 2
        self.pnl       = self.pos_value * (raw_ret - cost)

    def current_pnl(self, cost_bps: float = 10.0) -> float:
        if self.status != "OPEN":
            return self.pnl
        if self.is_hold or self.current_px <= 0:
            return 0.0
        raw_ret = (self.current_px - self.entry_px) / self.entry_px
        if not self.is_long:
            raw_ret = -raw_ret
        cost = (cost_bps / 10000.0) * 2
        return self.pos_value * (raw_ret - cost)

    def status_icon(self) -> str:
        return {
            "OPEN":   "⏳",
            "TP_HIT": "🟢",
            "SL_HIT": "🔴",
            "CLOSED": "⏸️ "
        }.get(self.status, "?")

    def unrealised_pct(self) -> float:
        if self.current_px <= 0 or self.entry_px <= 0:
            return 0.0
        return (self.current_px - self.entry_px) / self.entry_px * 100


# ── Status printer ───────────────────────────────────────────────────

def print_status(positions: list, portfolio: float,
                 algo_name: str, risk: str,
                 check_num: int, next_check: str):
    os.system("clear")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header(f"LIVE PAPER TRADER  ·  {now}")
    print(f"  Algorithm : {algo_name}   Risk: {risk.capitalize()}")
    print(f"  Portfolio : €{portfolio:,.2f}")
    print(f"  Check #{check_num:<4}  Next update: {next_check}")

    open_pos   = [p for p in positions if p.status == "OPEN" and not p.is_hold]
    closed_pos = [p for p in positions if p.status in ("TP_HIT","SL_HIT")]
    hold_pos   = [p for p in positions if p.is_hold]

    # Open positions
    if open_pos:
        section(f"OPEN POSITIONS ({len(open_pos)})")
        for p in open_pos:
            unreal = p.unrealised_pct()
            dist_tp = (p.tp_price - p.current_px) / p.current_px * 100
            dist_sl = (p.current_px - p.sl_price) / p.current_px * 100
            print(f"  {p.symbol:<12} {p.name:<18} {p.regime:<10} {p.signal}")
            print(f"    Entry: €{p.entry_px:>12,.2f}  →  Now: €{p.current_px:>12,.2f}  "
                  f"({unreal:+.2f}%)")
            print(f"    TP:    €{p.tp_price:>12,.2f}  ({dist_tp:+.2f}% away)   "
                  f"SL: €{p.sl_price:>12,.2f}  ({dist_sl:+.2f}% away)")
            pnl = p.current_pnl()
            sign = "+" if pnl >= 0 else ""
            print(f"    Size:  €{p.pos_value:>8,.2f}  ({p.pos_size_pct:.1f}%)   "
                  f"Unrealised P&L: {sign}€{pnl:.2f}")
            print()

    # Closed positions
    if closed_pos:
        section(f"CLOSED POSITIONS ({len(closed_pos)})")
        for p in closed_pos:
            icon   = p.status_icon()
            label  = "TP HIT" if p.status == "TP_HIT" else "SL HIT"
            sign   = "+" if p.pnl >= 0 else ""
            held   = str(p.exit_time - datetime(
                         p.exit_time.year, p.exit_time.month, p.exit_time.day,
                         0, 0, 0)).split(".")[0] if p.exit_time else "?"
            print(f"  {icon} {p.symbol:<12} {label:<8}  "
                  f"Entry €{p.entry_px:,.2f} → Exit €{p.exit_px:,.2f}   "
                  f"P&L: {sign}€{p.pnl:.2f}")

    # HOLD positions
    if hold_pos:
        section(f"NO TRADE (HOLD signal)")
        for p in hold_pos:
            print(f"  ⏸️   {p.symbol:<12} {p.name:<18} Regime: {p.regime}  "
                  f"Now: €{p.current_px:,.2f}")

    # Summary
    total_pnl    = sum(p.current_pnl() for p in positions)
    realised_pnl = sum(p.pnl for p in closed_pos)
    unreal_pnl   = sum(p.current_pnl() for p in open_pos)
    deployed     = sum(p.pos_value for p in open_pos)

    sep("═")
    print(f"\n  LIVE SUMMARY")
    sep()
    print(f"  Deployed     : €{deployed:,.2f}  "
          f"({deployed/portfolio*100:.1f}% of portfolio)")
    sign = "+" if unreal_pnl >= 0 else ""
    print(f"  Unrealised   : {sign}€{unreal_pnl:.2f}")
    sign = "+" if realised_pnl >= 0 else ""
    print(f"  Realised     : {sign}€{realised_pnl:.2f}")
    sign = "+" if total_pnl >= 0 else ""
    print(f"  Total P&L    : {sign}€{total_pnl:.2f}  "
          f"({total_pnl/portfolio*100:+.3f}%)")
    print(f"  Portfolio    : €{portfolio + total_pnl:,.2f}")
    sep("═")

    remaining = len(open_pos)
    if remaining == 0 and len(closed_pos) > 0:
        print("\n  ✅ All positions closed. Printing final report...\n")
    else:
        print(f"\n  {remaining} position(s) still open  ·  Ctrl+C to stop early\n")


# ── Final report ─────────────────────────────────────────────────────

def print_final_report(positions: list, portfolio: float,
                       algo_name: str, risk: str,
                       start_time: datetime):
    elapsed  = datetime.now() - start_time
    hours    = int(elapsed.total_seconds() // 3600)
    minutes  = int((elapsed.total_seconds() % 3600) // 60)
    now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header(f"FINAL PAPER TRADING REPORT  ·  {now}")
    print(f"  Algorithm : {algo_name}")
    print(f"  Portfolio : €{portfolio:,.2f}")
    print(f"  Risk      : {risk.capitalize()}")
    print(f"  Session   : {hours}h {minutes}m")

    total_pnl = 0.0
    for p in positions:
        section(f"{p.symbol}  —  {p.name}")
        print(f"  Signal    : {p.signal}  (confidence: {p.confidence:.0%})")
        print(f"  Regime    : {p.regime}")
        print(f"  5-Day Pred: €{p.pred_5d:,.2f}  (expected: {p.exp_ret*100:+.2f}%)")

        if p.is_hold:
            print(f"  Result    : ⏸️  NO TRADE (HOLD signal)")
            print(f"  Entry     : €{p.entry_px:,.2f}")
            print(f"  Final     : €{p.current_px:,.2f}")
            print(f"  P&L       : €0.00 (not in position)")
        else:
            icon  = p.status_icon()
            label = {"TP_HIT":"✅ TAKE PROFIT HIT",
                     "SL_HIT":"❌ STOP LOSS HIT",
                     "OPEN":  "⏳ STILL OPEN"}.get(p.status, p.status)

            print(f"  Result    : {icon} {label}")
            print(f"  Entry     : €{p.entry_px:,.2f}")

            exit_px = p.exit_px or p.current_px
            print(f"  Exit      : €{exit_px:,.2f}")
            print(f"  TP level  : €{p.tp_price:,.2f}")
            print(f"  SL level  : €{p.sl_price:,.2f}")
            print(f"  Position  : €{p.pos_value:,.2f}  ({p.pos_size_pct:.1f}%)")

            pnl  = p.pnl if p.status != "OPEN" else p.current_pnl()
            sign = "+" if pnl >= 0 else ""
            ret  = (exit_px - p.entry_px) / p.entry_px * 100
            print(f"  Return    : {ret:+.3f}%")
            print(f"  Net P&L   : {sign}€{pnl:.2f}  (after 10 bps costs)")
            total_pnl += pnl

    sep("═")
    print(f"\n  PORTFOLIO FINAL SUMMARY")
    sep()

    tp_count  = sum(1 for p in positions if p.status == "TP_HIT")
    sl_count  = sum(1 for p in positions if p.status == "SL_HIT")
    open_count= sum(1 for p in positions if p.status == "OPEN"
                    and not p.is_hold)
    hold_count= sum(1 for p in positions if p.is_hold)

    print(f"  ✅ TP hit     : {tp_count}")
    print(f"  ❌ SL hit     : {sl_count}")
    print(f"  ⏳ Still open : {open_count}")
    print(f"  ⏸️  No trade   : {hold_count}")
    sign = "+" if total_pnl >= 0 else ""
    print(f"\n  Total net P&L : {sign}€{total_pnl:.2f}")
    print(f"  Portfolio end : €{portfolio + total_pnl:,.2f}")
    print(f"  Change        : {total_pnl/portfolio*100:+.3f}%")
    sep("═")
    print()

    # Save
    os.makedirs("data/output", exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"data/output/live_trade_{ts}.json"
    with open(path, "w") as f:
        json.dump({
            "timestamp":   now,
            "algorithm":   algo_name,
            "portfolio":   portfolio,
            "risk":        risk,
            "session_mins":int(elapsed.total_seconds()//60),
            "summary": {
                "tp_hit":        tp_count,
                "sl_hit":        sl_count,
                "still_open":    open_count,
                "no_trade":      hold_count,
                "total_pnl":     total_pnl,
                "portfolio_end": portfolio + total_pnl
            },
            "positions": [
                {
                    "symbol":     p.symbol,
                    "signal":     p.signal,
                    "regime":     p.regime,
                    "entry_px":   p.entry_px,
                    "exit_px":    p.exit_px,
                    "tp_price":   p.tp_price,
                    "sl_price":   p.sl_price,
                    "status":     p.status,
                    "pnl":        p.pnl if p.status != "OPEN"
                                  else p.current_pnl(),
                    "pos_value":  p.pos_value,
                } for p in positions
            ]
        }, f, indent=2, default=str)
    print(f"  ✅ Report saved → {path}\n")


# ── TP/SL mode picker ────────────────────────────────────────────────

def pick_tpsl_mode() -> str:
    section("TP/SL MODE")
    print("  1. Standard  — risk profile levels (5-day horizon)")
    print("     TP=15-25% away  SL=8-10% away")
    print("     Best for: multi-day swing trades")
    print()
    print("  2. Daily ATR — tight levels based on today's volatility")
    print("     TP=1.5x ATR away  SL=1.0x ATR away")
    print("     Best for: intraday simulation, closes same day")
    raw = input("\n  Enter choice (1/2, default=2): ").strip() or "2"
    return "daily" if raw == "2" else "standard"


def calc_atr_levels(df: pd.DataFrame, entry_px: float,
                    is_long: bool, atr_mult_tp: float = 1.5,
                    atr_mult_sl: float = 1.0):
    """Calculate TP/SL based on 14-day ATR."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    tr1   = high - low
    tr2   = (high - close.shift()).abs()
    tr3   = (low  - close.shift()).abs()
    atr   = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    atr_val = float(atr.iloc[-1]) if not atr.empty else entry_px * 0.01
    if is_long:
        tp = entry_px + atr_val * atr_mult_tp
        sl = entry_px - atr_val * atr_mult_sl
    else:
        tp = entry_px - atr_val * atr_mult_tp
        sl = entry_px + atr_val * atr_mult_sl
    return tp, sl


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60,
                        help="Price check interval in seconds (default 60)")
    args = parser.parse_args()

    header("LIVE PAPER TRADER  ·  SETUP")
    print("  Monitors real market prices and auto-closes when TP/SL hit.")
    print(f"  Price checks every {args.interval} seconds.\n")

    assets    = pick_assets()
    portfolio = pick_portfolio()
    algo      = pick_algorithm()
    risk      = pick_risk()
    tpsl_mode = pick_tpsl_mode()
    rp        = RISK_PARAMS[risk]

    print(f"\n  ✅ Running initial analysis...")
    start_time = datetime.now()

    from src.utils.trade_logger import TradeLogger
    logger    = TradeLogger()
    positions = []
    trade_ids = {}

    for asset in assets:
        symbol = asset["symbol"]
        name   = asset["name"]
        print(f"  [{symbol}] Analysing...")

        try:
            df = fetch_data(symbol, period="8mo")
            if df.empty or len(df) < 50:
                print(f"  ⚠️  [{symbol}] Not enough data — skipping")
                continue
        except Exception as e:
            print(f"  ⚠️  [{symbol}] Data failed: {e}")
            continue

        # Entry price = live price right now
        live_px    = get_live_price(symbol)
        entry_px   = live_px if live_px > 0 else float(df["Close"].iloc[-1])

        sent = get_sentiment(symbol)
        df   = add_features(df, sent["sentiment_mean"], sent["sentiment_trend"])
        df.dropna(subset=["Returns", "RSI"], inplace=True)

        try:
            regime = run_regime(algo["regime"], df)
        except Exception:
            regime = "sideways"

        try:
            forecast   = run_forecast(algo["forecast"], df, entry_px)
        except Exception:
            forecast   = {h: {"predicted_price": entry_px,
                               "expected_return": 0.0} for h in [5,10,15,20]}

        pred_5d    = forecast[5]["predicted_price"]
        exp_ret    = forecast[5]["expected_return"]
        confidence = max(0.3, 1 - float(df["Volatility"].iloc[-1] or 0.02) * 2)
        vol        = float(df["Volatility"].iloc[-1] or 0.01)
        std_move   = entry_px * vol * np.sqrt(5 / 252)
        price_range= (pred_5d - std_move, pred_5d + std_move)

        state = {
            "symbol":               symbol,
            "current_price":        entry_px,
            "predicted_price":      pred_5d,
            "predicted_price_lgbm": pred_5d,
            "predicted_price_score":entry_px * (1 + exp_ret),
            "expected_return":      exp_ret,
            "expected_return_lgbm": exp_ret,
            "expected_return_score":exp_ret,
            "confidence":           confidence,
            "price_range":          price_range,
            "regime":               regime,
            "regime_confidence":    0.6,
        }
        signal = run_trader(algo["trader"], state, risk, portfolio)

        is_long   = "BUY" in signal["signal"]
        if tpsl_mode == "daily":
            tp_px, sl_px = calc_atr_levels(df, entry_px, is_long)
        else:
            tp_px = signal.get("take_profit", entry_px * 1.15)
            sl_px = signal.get("stop_loss",   entry_px * 0.92)

        pos = Position(
            symbol       = symbol,
            name         = name,
            entry_px     = entry_px,
            signal       = signal["signal"],
            tp_price     = tp_px,
            sl_price     = sl_px,
            pos_value    = signal["position_size"] * portfolio,
            pos_size_pct = signal["position_size"] * 100,
            regime       = regime.upper(),
            confidence   = signal["confidence"],
            pred_5d      = pred_5d,
            exp_ret      = exp_ret,
        )
        positions.append(pos)

        # Log entry
        if not pos.is_hold:
            tid = logger.log_entry(
                symbol=symbol, name=name,
                signal=signal["signal"], regime=regime.upper(),
                entry_px=entry_px, tp_price=tp_px, sl_price=sl_px,
                pos_value=pos.pos_value, algorithm=algo["name"],
                risk=risk, confidence=signal["confidence"],
                pred_5d=pred_5d, exp_ret=exp_ret,
                tp_sl_mode=tpsl_mode
            )
            trade_ids[symbol] = tid

        print(f"  [{symbol}] {signal['signal']:<12} "
              f"Entry: €{entry_px:,.2f}  "
              f"TP: €{pos.tp_price:,.2f}  "
              f"SL: €{pos.sl_price:,.2f}  "
              f"({'ATR daily' if tpsl_mode=='daily' else 'standard'})")

    if not positions:
        print("\n❌ No positions created.")
        return

    # Separate tradeable from hold
    active = [p for p in positions if not p.is_hold]
    print(f"\n  ✅ {len(active)} active positions  |  "
          f"{len(positions)-len(active)} HOLD (no trade)")
    print(f"  ⏱️  Monitoring every {args.interval}s  ·  Ctrl+C to stop\n")
    time.sleep(2)

    check_num = 0
    try:
        while True:
            open_active = [p for p in active if p.status == "OPEN"]
            if not open_active:
                break

            # Fetch live prices
            symbols     = [p.symbol for p in open_active]
            live_prices = get_live_prices(symbols)

            # Update HOLD positions too (for display)
            hold_pos = [p for p in positions if p.is_hold]
            for p in hold_pos:
                px = get_live_price(p.symbol)
                if px > 0:
                    p.current_px = px

            # Check TP/SL
            for p in open_active:
                px = live_prices.get(p.symbol, 0)
                if px > 0:
                    was_open = p.status == "OPEN"
                    p.update(px)
                    if was_open and p.status in ("TP_HIT","SL_HIT"):
                        tid = trade_ids.get(p.symbol)
                        if tid:
                            logger.log_exit(tid, p.exit_px,
                                            p.status, p.pnl)

            check_num += 1
            next_time  = (datetime.now() +
                          timedelta(seconds=args.interval)
                         ).strftime("%H:%M:%S")

            print_status(positions, portfolio, algo["name"],
                         risk, check_num, next_time)

            # Check if all done
            still_open = [p for p in active if p.status == "OPEN"]
            if not still_open:
                print("  ✅ All positions resolved.\n")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n  ⚠️  Stopped by user.\n")

    print_final_report(positions, portfolio, algo["name"],
                       risk, start_time)
    logger.print_summary()


if __name__ == "__main__":
    main()
