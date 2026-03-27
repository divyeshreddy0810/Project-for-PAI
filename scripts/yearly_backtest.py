#!/usr/bin/env python3
"""
Yearly Backtest 2020-2026
$100,000 starting capital each year — how much did the system make?
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from datetime import datetime

from src.forecast.patchtst_forecast      import PatchTSTForecaster
from src.rl.trading_env                  import TradingEnvironment
from src.rl.sac_agent                    import SACAgent
from src.rl.ppo_agent                    import PPOAgent
from src.rl.macro_sentiment_features     import build_macro_signals
from src.rl.integrated_pipeline          import build_patchtst_signals

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "data/models"

YEARS = [
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-03-26"),
]

TEST_ASSETS = [
    ("^GSPC",    "S&P 500",   "equity",    "SAC"),
    ("BTC-USD",  "Bitcoin",   "crypto",    "PPO"),
    ("GC=F",     "Gold",      "commodity", "SAC"),
    ("^IXIC",    "NASDAQ",    "equity",    "SAC"),
    ("EURUSD=X", "EUR/USD",   "forex",     "SAC"),
    ("GBPUSD=X", "GBP/USD",   "forex",     "SAC"),
]

RISK_PARAMS   = {"max_pos": 0.10, "sl": 0.08, "tp": 0.15}
START_CAPITAL = 100_000.0
BEST_WEIGHTS  = [1.2, 1.2, 1.0, 0.6]

def safe_key(sym):
    return sym.replace("^","").replace("-","_").replace("=","")

def apply_weights(signals, weights):
    weighted = signals.copy()
    groups = [([0,1],weights[0]),([2],weights[1]),
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

def simulate_trade(entry, signal, tp, sl, future_prices):
    for price in future_prices:
        if signal == "BUY":
            if price >= tp: return price, "TP"
            if price <= sl: return price, "SL"
        else:
            if price <= tp: return price, "TP"
            if price >= sl: return price, "SL"
    return future_prices[-1], "TIMEOUT"

def run_year(year_label, start, end):
    print(f"\n{'='*60}")
    print(f"  YEAR {year_label}: {start} → {end}")
    print(f"{'='*60}")

    capital    = START_CAPITAL
    all_trades = []
    asset_stats= {}

    for sym, name, asset_class, agent_type in TEST_ASSETS:
        key       = safe_key(sym)
        ptst_path = f"{MODELS_DIR}/patchtst_{key}.pt"
        rl_path   = f"{MODELS_DIR}/{agent_type.lower()}_{key}.pt"

        if not os.path.exists(ptst_path) or \
           not os.path.exists(rl_path):
            continue

        try:
            df = yf.download(sym, start="2004-01-01",
                             end=end, progress=False)
            if hasattr(df.columns,'droplevel'):
                df.columns = df.columns.droplevel(1)
            df = df.dropna()

            df_before = df[df.index < start]
            df_period = df[(df.index >= start) &
                           (df.index <= end)]

            if len(df_before) < 200 or len(df_period) < 10:
                continue

            # Buy and hold return for comparison
            bah_ret = (float(df_period["Close"].iloc[-1]) /
                       float(df_period["Close"].iloc[0]) - 1) * 100

        except Exception:
            continue

        try:
            ptst = PatchTSTForecaster()
            ptst.load(ptst_path)
            if agent_type == "SAC":
                agent = SACAgent(state_dim=17, device=DEVICE)
                agent.net.load_state_dict(
                    torch.load(rl_path, map_location=DEVICE))
            else:
                agent = PPOAgent(state_dim=17, device=DEVICE,
                                 n_steps=256)
                agent.policy.load_state_dict(
                    torch.load(rl_path, map_location=DEVICE))
        except Exception:
            continue

        # Walk forward weekly
        idx       = list(df_period.index)
        week_size = 5
        asset_pnl = 0.0
        asset_trades = []

        for w in range(0, len(idx)-week_size, week_size):
            week_start = idx[w]
            df_hist    = df[df.index <= week_start].tail(90)\
                           .reset_index(drop=True)
            df_future  = df_period.iloc[w:w+week_size]

            if len(df_hist) < 60 or len(df_future) == 0:
                continue

            entry_price = float(df_hist["Close"].iloc[-1])

            try:
                pred_ret = ptst.predict_return(df_hist)

                hmm_m  = get_hmm(asset_class)
                df_hmm = df_hist.tail(60).copy()
                d      = df_hmm["Close"].diff()
                g      = d.clip(lower=0).rolling(14).mean()
                l      = (-d.clip(upper=0)).rolling(14).mean()
                df_hmm["RSI"]            = 100-(100/(1+g/(l+1e-9)))
                df_hmm["sentiment_mean"] = 0.05
                df_hmm  = df_hmm.dropna()
                regime  = hmm_m.predict(df_hmm)

                sig = build_patchtst_signals(df_hist, ptst)
                mac = build_macro_signals(df_hist, asset_sym=sym)
                sig = np.concatenate([sig, mac], axis=1)
                sig = apply_weights(sig, BEST_WEIGHTS)
                env = TradingEnvironment(df_hist, sig)
                state = env.reset()
                for _ in range(len(df_hist)-2):
                    if agent_type == "SAC":
                        a = agent.select_action(state,
                                                deterministic=True)
                    else:
                        a,_,_ = agent.select_action(state,
                                                     deterministic=True)
                    state,_,done,_ = env.step(a)
                    if done: break

                if agent_type == "SAC":
                    rl_a = agent.select_action(state,
                                               deterministic=True)
                else:
                    rl_a,_,_ = agent.select_action(state,
                                                    deterministic=True)
                rl_sig = {0:"BUY",1:"SELL",2:"HOLD"}.get(rl_a,"HOLD")

                votes = []
                if pred_ret >  0.003:   votes.append("BUY")
                elif pred_ret < -0.003: votes.append("SELL")
                else:                   votes.append("HOLD")
                if regime == "bull":    votes.append("BUY")
                elif regime == "bear":  votes.append("SELL")
                else:                   votes.append("HOLD")
                votes.append(rl_sig)

                buy_v  = votes.count("BUY")
                sell_v = votes.count("SELL")
                if buy_v >= 2:    signal = "BUY"
                elif sell_v >= 2: signal = "SELL"
                else:             signal = "HOLD"

                if signal == "HOLD":
                    continue

                conf    = max(buy_v,sell_v)/len(votes)
                pos_val = capital * RISK_PARAMS["max_pos"] * conf

                if signal == "BUY":
                    tp = entry_price*(1+RISK_PARAMS["tp"])
                    sl = entry_price*(1-RISK_PARAMS["sl"])
                else:
                    tp = entry_price*(1-RISK_PARAMS["tp"])
                    sl = entry_price*(1+RISK_PARAMS["sl"])

                future_px        = df_future["Close"].values
                exit_px, outcome = simulate_trade(
                    entry_price, signal, tp, sl, future_px)

                pnl_pct = (exit_px/entry_price-1)*\
                          (1 if signal=="BUY" else -1)
                pnl_usd = pos_val*pnl_pct - pos_val*0.001

                capital    += pnl_usd
                asset_pnl  += pnl_usd
                asset_trades.append({
                    "signal":signal,"pnl_pct":round(pnl_pct*100,2),
                    "pnl_usd":round(pnl_usd,2),"outcome":outcome,
                })
                all_trades.append(asset_trades[-1])

            except Exception:
                continue

        if asset_trades:
            asset_stats[sym] = {
                "name":     name,
                "trades":   len(asset_trades),
                "pnl":      round(asset_pnl,2),
                "win_rate": round(sum(1 for t in asset_trades
                                      if t["pnl_usd"]>0)/
                                  len(asset_trades),3),
                "bah_ret":  round(bah_ret,1),
            }
            print(f"  {sym:<12} {name:<12} "
                  f"P&L: ${asset_pnl:>+9,.0f}  "
                  f"({len(asset_trades)} trades)  "
                  f"B&H: {bah_ret:+.1f}%")

    # Year summary
    total_ret  = (capital/START_CAPITAL-1)*100
    total_pnl  = capital - START_CAPITAL
    n          = len(all_trades)
    wins       = sum(1 for t in all_trades if t["pnl_usd"]>0)
    win_rate   = wins/n if n>0 else 0
    rets       = [t["pnl_pct"]/100 for t in all_trades]
    sharpe     = (np.mean(rets)/(np.std(rets)+1e-9))*np.sqrt(52) \
                 if len(rets)>1 else 0.0

    print(f"\n  ── Year {year_label} Summary ──")
    print(f"  ${START_CAPITAL:,.0f} → ${capital:,.0f}  "
          f"({total_ret:+.1f}%)  "
          f"P&L: ${total_pnl:+,.0f}")
    print(f"  Trades: {n}  Win rate: {win_rate:.1%}  "
          f"Sharpe: {sharpe:.3f}")

    return {
        "year":        year_label,
        "end_capital": round(capital,2),
        "total_return":round(total_ret,2),
        "total_pnl":   round(total_pnl,2),
        "n_trades":    n,
        "win_rate":    round(win_rate,3),
        "sharpe":      round(sharpe,3),
        "asset_stats": asset_stats,
    }

def main():
    print("\n"+"="*60)
    print("  YEARLY BACKTEST 2020-2026")
    print(f"  $100,000 fresh capital each year")
    print(f"  Device: {DEVICE}")
    print("="*60)

    results = []
    for year_label, start, end in YEARS:
        r = run_year(year_label, start, end)
        results.append(r)

    # Big summary table
    print(f"\n{'='*60}")
    print(f"  YEARLY RESULTS — $100,000 start each year")
    print(f"{'='*60}")
    print(f"  {'Year':<6} {'End Capital':>12} {'Return':>8} "
          f"{'P&L':>10} {'Sharpe':>8} {'WinRate':>8}")
    print(f"  {'─'*56}")

    total_profit = 0
    for r in results:
        icon = "✅" if r["total_return"] > 0 else "❌"
        print(f"  {r['year']:<6} "
              f"${r['end_capital']:>11,.0f} "
              f"{r['total_return']:>+7.1f}% "
              f"${r['total_pnl']:>+9,.0f} "
              f"{r['sharpe']:>+7.3f}  "
              f"{r['win_rate']:>6.1%}  {icon}")
        total_profit += r["total_pnl"]

    print(f"  {'─'*56}")
    avg_ret    = np.mean([r["total_return"] for r in results])
    avg_sharpe = np.mean([r["sharpe"] for r in results])
    avg_wr     = np.mean([r["win_rate"] for r in results])
    wins       = sum(1 for r in results if r["total_return"]>0)

    print(f"  {'AVG':<6} {'':>12} "
          f"{avg_ret:>+7.1f}% "
          f"${total_profit/len(results):>+9,.0f} "
          f"{avg_sharpe:>+7.3f}  "
          f"{avg_wr:>6.1%}")
    print(f"\n  Profitable years: {wins}/{len(results)}")
    print(f"  Total profit if ran every year: "
          f"${total_profit:+,.0f}")
    print(f"  Avg annual return: {avg_ret:+.1f}%")

    # Compound growth — if you reinvested
    compound = START_CAPITAL
    for r in results:
        compound *= (1 + r["total_return"]/100)
    compound_ret = (compound/START_CAPITAL-1)*100
    print(f"\n  If you reinvested profits each year:")
    print(f"  $100,000 in 2020 → ${compound:,.0f} by 2026")
    print(f"  Total compound return: {compound_ret:+.1f}%")

    os.makedirs("data/output", exist_ok=True)
    with open("data/output/yearly_backtest.json","w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  💾 Saved → data/output/yearly_backtest.json")

if __name__ == "__main__":
    main()
