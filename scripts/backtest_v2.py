#!/usr/bin/env python3
"""
Backtest V2 — March 2025 to March 2026
=======================================
Tests what $100,000 would have become following:
  - V1: daily_advisor (HMM + LightGBM + rules, forex only)
  - V2: MRAT-RL (PatchTST + SAC/PPO + per-class HMM, all assets)

Walk-forward: train on data before each week,
predict that week, move forward one week at a time.
"""

import os, sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from datetime import datetime, timedelta

from src.forecast.patchtst_forecast      import PatchTSTForecaster
from src.rl.trading_env                  import TradingEnvironment
from src.rl.sac_agent                    import SACAgent
from src.rl.ppo_agent                    import PPOAgent
from src.rl.macro_sentiment_features     import build_macro_signals
from src.rl.integrated_pipeline          import build_patchtst_signals
from src.rl.rl_trainer                   import rule_based_baseline

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "data/models"

# Test universe — assets with saved models
TEST_ASSETS_V2 = [
    ("^GSPC",    "S&P 500",   "equity",    "SAC"),
    ("BTC-USD",  "Bitcoin",   "crypto",    "PPO"),
    ("GC=F",     "Gold",      "commodity", "SAC"),
    ("^IXIC",    "NASDAQ",    "equity",    "SAC"),
    ("EURUSD=X", "EUR/USD",   "forex",     "SAC"),
    ("GBPUSD=X", "GBP/USD",   "forex",     "SAC"),
]

# V1 assets — forex only
TEST_ASSETS_V1 = [
    ("EURUSD=X", "EUR/USD",   "forex",     "SAC"),
    ("GBPUSD=X", "GBP/USD",   "forex",     "SAC"),
    ("USDJPY=X", "USD/JPY",   "forex",     "SAC"),
]

BEST_WEIGHTS  = [1.2, 1.2, 1.0, 0.6]

def dynamic_position_size(confidence, max_pos=0.10):
    if confidence <= 0.34:   scale = 0.33
    elif confidence <= 0.68: scale = 0.67
    else:                    scale = 1.00
    return max_pos * scale
START_CAPITAL = 100_000.0
START_DATE    = "2025-03-26"
END_DATE      = "2026-03-26"
RISK_PARAMS   = {"max_pos": 0.10, "sl": 0.08, "tp": 0.15}

def safe_key(sym):
    return sym.replace("^","").replace("-","_").replace("=","")

def apply_weights(signals, weights):
    weighted = signals.copy()
    groups   = [([0,1],weights[0]),([2],weights[1]),
                (list(range(3,10)),weights[2]),
                (list(range(10,14)),weights[3])]
    for cols,w in groups:
        for c in cols:
            if c < weighted.shape[1]:
                weighted[:,c] *= w
    return weighted

def get_hmm(asset_class):
    import joblib
    path = f"{MODELS_DIR}/hmm_{asset_class}.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    from src.regime.pretrained_hmm import get_pretrained_hmm
    return get_pretrained_hmm()

def simulate_trade(entry_price, signal, tp, sl, future_prices):
    """
    Simulate a trade given entry, TP, SL and future daily prices.
    Returns (exit_price, outcome, days_held).
    """
    for i, price in enumerate(future_prices):
        if signal == "BUY":
            if price >= tp:
                return price, "TP", i+1
            if price <= sl:
                return price, "SL", i+1
        elif signal == "SELL":
            if price <= tp:
                return price, "TP", i+1
            if price >= sl:
                return price, "SL", i+1
    # No TP/SL hit — exit at last price
    return future_prices[-1], "TIMEOUT", len(future_prices)

def run_backtest_v2(assets, label="V2"):
    """Run MRAT-RL backtest on given assets."""
    print(f"\n{'='*60}")
    print(f"  BACKTEST {label}: {START_DATE} → {END_DATE}")
    print(f"  Starting capital: ${START_CAPITAL:,.0f}")
    print(f"  Assets: {[a[0] for a in assets]}")
    print(f"{'='*60}")

    capital   = START_CAPITAL
    all_trades= []
    weekly_equity = [{"date": START_DATE, "equity": capital}]

    # Fetch all data upfront
    asset_data = {}
    for sym, name, asset_class, agent_type in assets:
        try:
            df = yf.download(sym, start="2004-01-01",
                             end=END_DATE, progress=False)
            if hasattr(df.columns,'droplevel'):
                df.columns = df.columns.droplevel(1)
            asset_data[sym] = df.dropna()
            print(f"  📥 {sym}: {len(asset_data[sym])} rows")
        except Exception as e:
            print(f"  ❌ {sym}: {e}")

    # Load models
    models = {}
    for sym, name, asset_class, agent_type in assets:
        key      = safe_key(sym)
        ptst_path= f"{MODELS_DIR}/patchtst_{key}.pt"
        rl_path  = f"{MODELS_DIR}/{agent_type.lower()}_{key}.pt"
        if not os.path.exists(ptst_path) or \
           not os.path.exists(rl_path):
            print(f"  ⚠️  {sym}: models not found — skipping")
            continue
        try:
            ptst = PatchTSTForecaster()
            ptst.load(ptst_path)
            state_dim = 17  # default
            if agent_type == "SAC":
                agent = SACAgent(state_dim=state_dim, device=DEVICE)
                agent.net.load_state_dict(
                    torch.load(rl_path, map_location=DEVICE))
            else:
                agent = PPOAgent(state_dim=state_dim,
                                 device=DEVICE, n_steps=256)
                agent.policy.load_state_dict(
                    torch.load(rl_path, map_location=DEVICE))
            models[sym] = {
                "ptst": ptst, "agent": agent,
                "agent_type": agent_type,
                "asset_class": asset_class,
                "name": name
            }
            print(f"  ✅ {sym}: models loaded")
        except Exception as e:
            print(f"  ❌ {sym} model load failed: {e}")

    # Weekly walk-forward
    test_start = pd.Timestamp(START_DATE)
    test_end   = pd.Timestamp(END_DATE)
    current    = test_start

    week_num = 0
    while current < test_end:
        week_end = min(current + timedelta(days=7), test_end)
        week_num += 1

        week_trades = []

        for sym, name, asset_class, agent_type in assets:
            if sym not in models or sym not in asset_data:
                continue

            df_full = asset_data[sym]
            # Data up to start of this week (no lookahead)
            df_hist  = df_full[df_full.index < current]
            # This week's actual prices (for trade simulation)
            df_week  = df_full[(df_full.index >= current) &
                                (df_full.index < week_end)]

            if len(df_hist) < 100 or len(df_week) == 0:
                continue

            entry_price = float(df_hist["Close"].iloc[-1])
            m = models[sym]

            # Generate signal
            try:
                df_recent = df_hist.tail(90).reset_index(drop=True)
                pred_return = m["ptst"].predict_return(df_recent)

                # HMM regime
                hmm_model = get_hmm(asset_class)
                df_hmm    = df_recent.tail(60).copy()
                delta = df_hmm["Close"].diff()
                gain  = delta.clip(lower=0).rolling(14).mean()
                loss  = (-delta.clip(upper=0)).rolling(14).mean()
                df_hmm["RSI"]            = 100-(100/(1+gain/(loss+1e-9)))
                df_hmm["sentiment_mean"] = 0.05
                df_hmm  = df_hmm.dropna()
                regime  = hmm_model.predict(df_hmm)

                # RL signal
                sig = build_patchtst_signals(df_recent, m["ptst"])
                mac = build_macro_signals(df_recent, asset_sym=sym)
                sig = np.concatenate([sig, mac], axis=1)
                sig = apply_weights(sig, BEST_WEIGHTS)
                env = TradingEnvironment(df_recent, sig)
                state = env.reset()
                for _ in range(len(df_recent)-2):
                    if agent_type == "SAC":
                        a = m["agent"].select_action(
                            state, deterministic=True)
                    else:
                        a,_,_ = m["agent"].select_action(
                            state, deterministic=True)
                    state,_,done,_ = env.step(a)
                    if done: break

                if agent_type == "SAC":
                    rl_a = m["agent"].select_action(
                        state, deterministic=True)
                else:
                    rl_a,_,_ = m["agent"].select_action(
                        state, deterministic=True)
                rl_sig = {0:"BUY",1:"SELL",2:"HOLD"}.get(rl_a,"HOLD")

                # Consensus
                votes = []
                if pred_return >  0.003:   votes.append("BUY")
                elif pred_return < -0.003: votes.append("SELL")
                else:                      votes.append("HOLD")
                if regime == "bull":       votes.append("BUY")
                elif regime == "bear":     votes.append("SELL")
                else:                      votes.append("HOLD")
                votes.append(rl_sig)

                buy_v  = votes.count("BUY")
                sell_v = votes.count("SELL")
                if buy_v >= 2:    signal = "BUY"
                elif sell_v >= 2: signal = "SELL"
                else:             signal = "HOLD"

                confidence = max(buy_v, sell_v) / len(votes)

            except Exception as e:
                signal = "HOLD"; confidence = 0.0

            if signal == "HOLD":
                continue

            # Trade sizing
            pos_size  = RISK_PARAMS["max_pos"] * confidence
            pos_value = capital * dynamic_position_size(confidence, RISK_PARAMS["max_pos"])

            # TP/SL
            if signal == "BUY":
                tp = entry_price * (1 + RISK_PARAMS["tp"])
                sl = entry_price * (1 - RISK_PARAMS["sl"])
            else:
                tp = entry_price * (1 - RISK_PARAMS["tp"])
                sl = entry_price * (1 + RISK_PARAMS["sl"])

            # Simulate trade
            future_prices = df_week["Close"].values
            exit_price, outcome, days = simulate_trade(
                entry_price, signal, tp, sl, future_prices)

            pnl_pct = (exit_price/entry_price - 1) * \
                      (1 if signal=="BUY" else -1)
            pnl_usd = pos_value * pnl_pct
            # 10bps transaction cost
            pnl_usd -= pos_value * 0.001

            week_trades.append({
                "week":       week_num,
                "date":       current.strftime("%Y-%m-%d"),
                "symbol":     sym,
                "name":       name,
                "signal":     signal,
                "entry":      round(entry_price, 4),
                "exit":       round(exit_price, 4),
                "outcome":    outcome,
                "pnl_pct":   round(pnl_pct*100, 2),
                "pnl_usd":   round(pnl_usd, 2),
                "confidence": round(confidence, 2),
                "regime":     regime,
                "days_held":  days,
            })

        # Apply week's P&L
        week_pnl = sum(t["pnl_usd"] for t in week_trades)
        capital  += week_pnl
        all_trades.extend(week_trades)

        weekly_equity.append({
            "date":   week_end.strftime("%Y-%m-%d"),
            "equity": round(capital, 2),
            "week_pnl": round(week_pnl, 2),
        })

        if week_num % 4 == 0:  # Print monthly
            total_ret = (capital/START_CAPITAL - 1)*100
            print(f"  Week {week_num:2d} "
                  f"({current.strftime('%Y-%m-%d')}): "
                  f"${capital:>10,.0f}  "
                  f"({total_ret:+.1f}%)  "
                  f"{len(week_trades)} trades")

        current = week_end

    # Final results
    total_return  = (capital/START_CAPITAL - 1) * 100
    n_trades      = len(all_trades)
    wins          = sum(1 for t in all_trades if t["pnl_usd"] > 0)
    win_rate      = wins/n_trades if n_trades > 0 else 0
    total_pnl     = capital - START_CAPITAL

    returns_list  = [t["pnl_pct"]/100 for t in all_trades]
    if len(returns_list) > 1:
        sharpe = (np.mean(returns_list) /
                  (np.std(returns_list) + 1e-9)) * np.sqrt(52)
    else:
        sharpe = 0.0

    # Max drawdown
    equity_vals = [e["equity"] for e in weekly_equity]
    peak        = START_CAPITAL
    max_dd      = 0.0
    for eq in equity_vals:
        peak   = max(peak, eq)
        dd     = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    print(f"\n{'─'*60}")
    print(f"  BACKTEST {label} RESULTS")
    print(f"{'─'*60}")
    print(f"  Start capital : ${START_CAPITAL:>10,.0f}")
    print(f"  End capital   : ${capital:>10,.0f}")
    print(f"  Total return  : {total_return:>+.1f}%")
    print(f"  Total P&L     : ${total_pnl:>+10,.0f}")
    print(f"  Total trades  : {n_trades}")
    print(f"  Win rate      : {win_rate:.1%}")
    print(f"  Sharpe ratio  : {sharpe:.3f}")
    print(f"  Max drawdown  : {max_dd:.1%}")
    print(f"{'─'*60}")

    return {
        "label":         label,
        "start_capital": START_CAPITAL,
        "end_capital":   round(capital, 2),
        "total_return":  round(total_return, 2),
        "total_pnl":     round(total_pnl, 2),
        "n_trades":      n_trades,
        "win_rate":      round(win_rate, 3),
        "sharpe":        round(sharpe, 3),
        "max_drawdown":  round(max_dd, 3),
        "weekly_equity": weekly_equity,
        "trades":        all_trades,
    }

def main():
    print("\n" + "="*60)
    print("  MRAT-RL BACKTEST: Mar 2025 → Mar 2026")
    print(f"  Device: {DEVICE}")
    print("="*60)

    # V1 baseline — rule-based, forex only
    print("\n[V1 BASELINE — HMM+LightGBM+Rules, Forex only]")
    v1_results = {}
    v1_capital = START_CAPITAL
    v1_trades  = []

    for sym, name, asset_class, agent_type in TEST_ASSETS_V1:
        if sym not in [a[0] for a in TEST_ASSETS_V1]:
            continue
        try:
            df = yf.download(sym, start="2004-01-01",
                             end=END_DATE, progress=False)
            if hasattr(df.columns,'droplevel'):
                df.columns = df.columns.droplevel(1)
            df_test = df[df.index >= START_DATE].dropna()
            if len(df_test) < 10:
                continue
            perf = rule_based_baseline(df_test)
            pnl  = v1_capital * \
                   RISK_PARAMS["max_pos"] * perf["total_return"]
            v1_trades.append({
                "symbol":sym, "return":perf["total_return"],
                "sharpe":perf["sharpe_ratio"], "pnl":pnl
            })
            print(f"  {sym}: return={perf['total_return']:+.1%}  "
                  f"sharpe={perf['sharpe_ratio']:+.3f}")
        except Exception as e:
            print(f"  {sym}: ❌ {e}")

    if v1_trades:
        avg_v1_ret = np.mean([t["return"] for t in v1_trades])
        v1_end     = v1_capital * (1 + avg_v1_ret *
                                   RISK_PARAMS["max_pos"] *
                                   len(v1_trades))
        v1_sharpe  = np.mean([t["sharpe"] for t in v1_trades])
        print(f"\n  V1 Summary: ${v1_capital:,.0f} → "
              f"${v1_end:,.0f}  "
              f"({(v1_end/v1_capital-1)*100:+.1f}%)  "
              f"Sharpe={v1_sharpe:.3f}")

    # V2 enhanced — full MRAT-RL
    v2 = run_backtest_v2(TEST_ASSETS_V2, "V2 MRAT-RL")

    # Save results
    os.makedirs("data/output", exist_ok=True)
    output = {
        "backtest_period": f"{START_DATE} to {END_DATE}",
        "start_capital":   START_CAPITAL,
        "v1": {
            "system":      "HMM+LightGBM+Rules (Forex only)",
            "assets":      [a[0] for a in TEST_ASSETS_V1],
            "trades":      v1_trades,
            "end_capital": round(v1_end, 2) if v1_trades else None,
            "avg_sharpe":  round(v1_sharpe, 3) if v1_trades else None,
        },
        "v2": v2,
    }
    path = "data/output/backtest_mar2025_mar2026.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  💾 Results saved → {path}")

    # Final comparison
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON: V1 vs V2")
    print(f"{'='*60}")
    if v1_trades:
        print(f"  V1 (Forex+Rules): "
              f"${v1_capital:,.0f} → ${v1_end:,.0f}  "
              f"Sharpe={v1_sharpe:.3f}")
    print(f"  V2 (MRAT-RL):     "
          f"${START_CAPITAL:,.0f} → "
          f"${v2['end_capital']:,.0f}  "
          f"({v2['total_return']:+.1f}%)  "
          f"Sharpe={v2['sharpe']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
