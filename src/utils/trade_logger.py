"""
TradeLogger
-----------
Persistent log of every signal the system generates.
Tracks: entry, TP, SL, outcome, actual P&L, accuracy over time.
Saves to data/logs/trade_log.json — append only, never overwrites.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

LOG_DIR  = "data/logs"
LOG_FILE = "data/logs/trade_log.json"


class TradeLogger:

    def __init__(self, log_file: str = LOG_FILE):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        if not os.path.exists(log_file):
            self._write([])

    def _read(self) -> List[Dict]:
        try:
            with open(self.log_file) as f:
                return json.load(f)
        except Exception:
            return []

    def _write(self, trades: List[Dict]):
        with open(self.log_file, "w") as f:
            json.dump(trades, f, indent=2, default=str)

    # ------------------------------------------------------------------
    def check_circuit_breaker(self, symbol: str,
                              lookback: int = 3) -> bool:
        """
        Returns True if trading is BLOCKED for this symbol.
        Triggered when last N trades all hit SL.
        """
        recent = [t for t in self.trades
                  if t.get("symbol") == symbol
                  and t.get("outcome") in ("SL_HIT", "OPEN")]
        last_n = [t for t in recent
                  if t.get("outcome") == "SL_HIT"][-lookback:]
        if len(last_n) >= lookback:
            print(f"  ⛔ Circuit breaker: {symbol} — "
                  f"{lookback} consecutive losses. Skipping.")
            return True
        return False

    def log_entry(self, symbol: str, name: str,
                  signal: str, regime: str,
                  entry_px: float, tp_price: float,
                  sl_price: float, pos_value: float,
                  algorithm: str, risk: str,
                  confidence: float, pred_5d: float,
                  exp_ret: float, tp_sl_mode: str) -> str:
        """Log a new trade entry. Returns trade_id."""
        trades   = self._read()
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        trade = {
            "trade_id":    trade_id,
            "symbol":      symbol,
            "name":        name,
            "algorithm":   algorithm,
            "risk":        risk,
            "tp_sl_mode":  tp_sl_mode,
            "signal":      signal,
            "regime":      regime,
            "confidence":  round(confidence, 4),
            "entry_px":    entry_px,
            "tp_price":    tp_price,
            "sl_price":    sl_price,
            "pos_value":   pos_value,
            "pred_5d":     pred_5d,
            "exp_ret":     round(exp_ret, 4),
            "entry_time":  datetime.now().isoformat(),
            "exit_time":   None,
            "exit_px":     None,
            "outcome":     "OPEN",   # OPEN / TP_HIT / SL_HIT / CLOSED
            "actual_ret":  None,
            "net_pnl":     None,
            "correct_dir": None,     # True if predicted direction was right
        }

        trades.append(trade)
        self._write(trades)
        return trade_id

    # ------------------------------------------------------------------
    def log_exit(self, trade_id: str, exit_px: float,
                 outcome: str, net_pnl: float):
        """Update a trade with its outcome."""
        trades = self._read()
        for t in trades:
            if t["trade_id"] == trade_id:
                t["exit_time"]  = datetime.now().isoformat()
                t["exit_px"]    = exit_px
                t["outcome"]    = outcome
                t["net_pnl"]    = round(net_pnl, 4)
                t["actual_ret"] = round(
                    (exit_px - t["entry_px"]) / t["entry_px"], 4
                ) if t["entry_px"] else 0.0
                # Correct direction: predicted up and went up, or predicted down and went down
                predicted_up    = t["exp_ret"] > 0
                actual_up       = t["actual_ret"] > 0
                t["correct_dir"]= predicted_up == actual_up
                break
        self._write(trades)

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Return accuracy and P&L summary across all closed trades."""
        trades  = self._read()
        closed  = [t for t in trades if t["outcome"] in
                   ("TP_HIT","SL_HIT","CLOSED")]

        if not closed:
            return {"total_trades": 0, "message": "No closed trades yet"}

        tp_count  = sum(1 for t in closed if t["outcome"] == "TP_HIT")
        sl_count  = sum(1 for t in closed if t["outcome"] == "SL_HIT")
        total_pnl = sum(t["net_pnl"] or 0 for t in closed)
        dir_acc   = sum(1 for t in closed
                        if t["correct_dir"]) / len(closed)

        by_algo: Dict[str, list] = {}
        for t in closed:
            by_algo.setdefault(t["algorithm"], []).append(t)

        algo_stats = {}
        for algo, ts in by_algo.items():
            pnls = [t["net_pnl"] or 0 for t in ts]
            algo_stats[algo] = {
                "trades":    len(ts),
                "tp_rate":   sum(1 for t in ts
                                 if t["outcome"]=="TP_HIT") / len(ts),
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl":   round(float(np.mean(pnls)), 2),
            }

        return {
            "total_trades":     len(closed),
            "open_trades":      len(trades) - len(closed),
            "tp_hit":           tp_count,
            "sl_hit":           sl_count,
            "tp_rate":          round(tp_count / len(closed), 3),
            "directional_acc":  round(dir_acc, 3),
            "total_net_pnl":    round(total_pnl, 2),
            "avg_pnl_per_trade":round(total_pnl / len(closed), 2),
            "by_algorithm":     algo_stats,
        }

    # ------------------------------------------------------------------
    def print_summary(self):
        """Print trade log summary to terminal."""
        s = self.summary()
        print("\n" + "═"*60)
        print("  TRADE LOG SUMMARY")
        print("═"*60)

        if s.get("total_trades", 0) == 0:
            print("  No closed trades yet.\n")
            return

        print(f"  Total closed : {s['total_trades']}")
        print(f"  Still open   : {s['open_trades']}")
        print(f"  TP hit       : {s['tp_hit']}  "
              f"({s['tp_rate']:.0%} of closed)")
        print(f"  SL hit       : {s['sl_hit']}")
        print(f"  Dir accuracy : {s['directional_acc']:.1%}")
        print(f"  Total P&L    : €{s['total_net_pnl']:+,.2f}")
        print(f"  Avg per trade: €{s['avg_pnl_per_trade']:+,.2f}")

        if s["by_algorithm"]:
            print(f"\n  By algorithm:")
            for algo, st in s["by_algorithm"].items():
                print(f"    {algo:<20} "
                      f"{st['trades']} trades  "
                      f"TP rate {st['tp_rate']:.0%}  "
                      f"P&L €{st['total_pnl']:+,.2f}")
        print("═"*60 + "\n")

    # ------------------------------------------------------------------
    def print_open_trades(self):
        """Print all currently open trades."""
        trades = self._read()
        open_t = [t for t in trades if t["outcome"] == "OPEN"]

        print("\n" + "═"*60)
        print("  OPEN TRADES")
        print("═"*60)
        if not open_t:
            print("  No open trades.\n")
            return
        for t in open_t:
            print(f"  {t['symbol']:<10} {t['signal']:<12} "
                  f"Entry €{t['entry_px']:,.2f}  "
                  f"TP €{t['tp_price']:,.2f}  "
                  f"SL €{t['sl_price']:,.2f}")
            print(f"             {t['algorithm']}  "
                  f"conf {t['confidence']:.0%}  "
                  f"regime {t['regime']}")
        print("═"*60 + "\n")
