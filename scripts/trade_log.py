#!/usr/bin/env python3
"""
Trade Log Viewer
----------------
Shows all logged trades and running accuracy stats.

Usage:
  python3 scripts/trade_log.py           # summary
  python3 scripts/trade_log.py --open    # open trades only
  python3 scripts/trade_log.py --all     # every trade
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.trade_logger import TradeLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", action="store_true",
                        help="Show open trades only")
    parser.add_argument("--all",  action="store_true",
                        help="Show every trade in full detail")
    args = parser.parse_args()

    logger = TradeLogger()

    if args.open:
        logger.print_open_trades()
        return

    logger.print_summary()

    if args.all:
        trades = logger._read()
        if not trades:
            print("  No trades logged yet.\n")
            return
        print(f"  ALL TRADES ({len(trades)} total)\n")
        print(f"  {'ID':<35} {'SYM':<8} {'SIGNAL':<12} {'OUTCOME':<10} "
              f"{'ENTRY':>10} {'EXIT':>10} {'P&L':>8}")
        print("  " + "─"*95)
        for t in trades:
            entry = f"€{t['entry_px']:,.2f}" if t['entry_px'] else "N/A"
            exit_ = f"€{t['exit_px']:,.2f}" if t['exit_px'] else "—"
            pnl   = f"€{t['net_pnl']:+.2f}" if t['net_pnl'] is not None else "—"
            print(f"  {t['trade_id']:<35} {t['symbol']:<8} "
                  f"{t['signal']:<12} {t['outcome']:<10} "
                  f"{entry:>10} {exit_:>10} {pnl:>8}")
        print()

if __name__ == "__main__":
    main()
