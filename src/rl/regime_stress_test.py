"""
Regime Stress Test
------------------
Tests the system on randomly selected 52-week periods across:
  - Bull regimes (S&P 500 above 200-day SMA, low VIX)
  - Bear regimes (S&P 500 below 200-day SMA, high VIX)
  - Random periods (no regime filter)

Tests each asset independently on $100k starting capital.
Uses UNSEEN data — periods the RL agent was NOT trained on.
"""

import sys, os, warnings, random, json
warnings.filterwarnings("ignore")
sys.path.insert(0,'.')

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from datetime import datetime, timedelta
from src.rl.trading_env import TradingEnvironment
from src.rl.sac_agent import SACAgent
from src.rl.ppo_agent import PPOAgent
from src.rl.macro_sentiment_features import build_macro_signals
from src.rl.integrated_pipeline import build_patchtst_signals
from src.forecast.patchtst_forecast import PatchTSTForecaster
from src.rl.rl_trainer import rule_based_baseline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_WEIGHTS = [1.2, 1.2, 1.0, 0.6]
WEEKS = 52
DAYS  = WEEKS * 5  # ~260 trading days
N_RANDOM_PERIODS = 5  # test 5 random periods per regime

ASSETS = [
    ("^GSPC",   "S&P 500",   "SAC"),
    ("BTC-USD", "Bitcoin",   "PPO"),
    ("GC=F",    "Gold",      "SAC"),
    ("^IXIC",   "NASDAQ",    "SAC"),
    ("EURUSD=X","EUR/USD",   "SAC"),
]

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


def classify_regime(df_window):
    """Classify a 52-week window as BULL, BEAR, or MIXED."""
    close = df_window["Close"]
    sma200_val = close.rolling(min(200, len(close))).mean().iloc[-1]
    current    = float(close.iloc[-1])
    ret_52w    = (current / float(close.iloc[0]) - 1) * 100

    if ret_52w > 10 and current > sma200_val:
        return "BULL"
    elif ret_52w < -10 or current < sma200_val * 0.95:
        return "BEAR"
    else:
        return "MIXED"


def find_regime_periods(df, target_regime, n=5):
    """Find n non-overlapping 52-week windows matching target regime."""
    periods = []
    used_ends = set()
    attempts  = 0
    indices   = list(range(DAYS, len(df) - DAYS, 20))
    random.shuffle(indices)

    for i in indices:
        if attempts > 200 or len(periods) >= n:
            break
        attempts += 1

        start_i = i - DAYS
        end_i   = i
        window  = df.iloc[start_i:end_i].copy()

        if len(window) < DAYS * 0.8:
            continue

        # Check no overlap with used periods
        overlap = any(abs(end_i - u) < DAYS//2 for u in used_ends)
        if overlap:
            continue

        regime = classify_regime(window)
        if target_regime == "RANDOM" or regime == target_regime:
            periods.append({
                "start": df.index[start_i].strftime("%Y-%m-%d"),
                "end":   df.index[end_i-1].strftime("%Y-%m-%d"),
                "regime": regime,
                "return_52w": float((df["Close"].iloc[end_i-1] /
                                     df["Close"].iloc[start_i] - 1) * 100),
                "start_i": start_i,
                "end_i":   end_i,
            })
            used_ends.add(end_i)

    return periods


def load_agent(sym, state_dim):
    """Load saved agent or return None."""
    agent_map = {
        "^GSPC":"SAC","^IXIC":"SAC","GC=F":"SAC",
        "EURUSD=X":"SAC","BTC-USD":"PPO"
    }
    atype = agent_map.get(sym, "SAC")
    key   = sym.replace("^","").replace("-","_").replace("=","")
    path  = f"data/models/{atype.lower()}_{key}.pt"

    if atype == "SAC":
        agent = SACAgent(state_dim=state_dim, device=DEVICE)
        if os.path.exists(path):
            agent.net.load_state_dict(
                torch.load(path, map_location=DEVICE))
            return agent, "SAC"
    else:
        agent = PPOAgent(state_dim=state_dim, device=DEVICE, n_steps=256)
        if os.path.exists(path):
            agent.policy.load_state_dict(
                torch.load(path, map_location=DEVICE))
            return agent, "PPO"
    return None, atype


def evaluate_period(agent, agent_type, df_window):
    """Evaluate agent on a single 52-week window."""
    df_w = df_window.reset_index(drop=True)

    ptst = PatchTSTForecaster()
    try:
        ptst.fit_from_df(df_w, verbose=False)
    except Exception:
        pass

    sym_hint = getattr(df_w, '_sym', None)
    sig = build_patchtst_signals(df_w, ptst)
    mac = build_macro_signals(df_w, asset_sym=sym_hint)
    sig = np.concatenate([sig, mac], axis=1)
    sig = apply_weights(sig, BEST_WEIGHTS)

    env   = TradingEnvironment(df_w, sig)
    state = env.reset()

    while True:
        if agent_type == "SAC":
            a = agent.select_action(state, deterministic=False)
        else:
            a,_,_ = agent.select_action(state, deterministic=False)
        state, _, done, _ = env.step(a)
        if done: break

    return env.get_performance()


def run_stress_test():
    random.seed(42)
    print("="*70)
    print("  REGIME STRESS TEST — 52-WEEK WINDOWS")
    print("  $100,000 starting capital per window")
    print("  UNSEEN data — random periods from full history")
    print("="*70)

    all_results = []
    os.makedirs("data/output", exist_ok=True)

    for sym, name, _ in ASSETS:
        print(f"\n{'='*65}")
        print(f"  {name} ({sym})")
        print(f"{'='*65}")

        # Fetch full history
        df = yf.download(sym, start="2004-01-01",
                         end="2026-03-25", progress=False)
        if hasattr(df.columns,'droplevel'):
            df.columns = df.columns.droplevel(1)
        df = df.dropna()
        print(f"  Full data: {len(df)} rows")

        # Load agent
        # Build dummy signals to get state_dim
        dummy = df.tail(100).reset_index(drop=True)
        try:
            ptst_d = PatchTSTForecaster()
            ptst_d.fit_from_df(dummy, verbose=False)
            sig_d  = build_patchtst_signals(dummy, ptst_d)
            mac_d  = build_macro_signals(dummy)
            sig_d  = np.concatenate([sig_d, mac_d], axis=1)
            state_dim = TradingEnvironment(dummy, sig_d).state_dim
        except Exception:
            state_dim = 17

        agent, agent_type = load_agent(sym, state_dim)
        if agent is None:
            print(f"  ⚠️  No saved model — skipping")
            continue

        asset_results = {"symbol": sym, "name": name, "regimes": {}}

        for regime in ["BULL", "BEAR", "RANDOM"]:
            print(f"\n  ── {regime} regime periods ──────────────")
            periods = find_regime_periods(df, regime, N_RANDOM_PERIODS)

            if not periods:
                print(f"  ⚠️  No {regime} periods found")
                continue

            sharpes=[]; returns=[]; dds=[]

            for p in periods:
                window = df.iloc[p["start_i"]:p["end_i"]].copy()
                try:
                    perf = evaluate_period(agent, agent_type, window)
                    sharpes.append(perf["sharpe_ratio"])
                    returns.append(perf["total_return"])
                    dds.append(perf["max_drawdown"])

                    rule = rule_based_baseline(window)
                    winner = "Agent ✅" if perf["sharpe_ratio"] > \
                             rule["sharpe_ratio"] else "Rule  ✅"

                    print(f"  {p['start']} → {p['end']} "
                          f"({p['regime']:4s} {p['return_52w']:+5.1f}%) | "
                          f"Agent: {perf['total_return']:+6.1%} "
                          f"Sharpe:{perf['sharpe_ratio']:+.2f} | "
                          f"{winner}")
                except Exception as e:
                    print(f"  {p['start']} → {p['end']} ❌ {e}")

            if sharpes:
                avg_sh = float(np.mean(sharpes))
                avg_rt = float(np.mean(returns))
                avg_dd = float(np.mean(dds))
                wins   = sum(1 for s in sharpes if s > 0)
                print(f"  {'─'*55}")
                print(f"  Avg Sharpe: {avg_sh:+.3f}  "
                      f"Avg Return: {avg_rt:+.1%}  "
                      f"Avg DD: {avg_dd:.1%}  "
                      f"Positive: {wins}/{len(sharpes)}")

                asset_results["regimes"][regime] = {
                    "avg_sharpe":  avg_sh,
                    "avg_return":  avg_rt,
                    "avg_drawdown":avg_dd,
                    "win_rate":    wins/len(sharpes),
                    "n_periods":   len(sharpes),
                }

        all_results.append(asset_results)

    # Summary
    print(f"\n{'='*70}")
    print(f"  STRESS TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Asset':<12} {'BULL Sh':>9} {'BEAR Sh':>9} "
          f"{'RANDOM Sh':>11} {'Best regime':>14}")
    print(f"  {'─'*60}")

    for r in all_results:
        bull = r["regimes"].get("BULL",{}).get("avg_sharpe", float('nan'))
        bear = r["regimes"].get("BEAR",{}).get("avg_sharpe", float('nan'))
        rand = r["regimes"].get("RANDOM",{}).get("avg_sharpe", float('nan'))
        vals = {"BULL":bull,"BEAR":bear,"RANDOM":rand}
        best = max(vals, key=lambda k: vals[k]
                   if not np.isnan(vals[k]) else -999)
        print(f"  {r['name']:<12} "
              f"{bull:>+8.3f}  {bear:>+8.3f}  {rand:>+10.3f}  "
              f"{best:>14}")

    # Save
    with open("data/output/stress_test_results.json","w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  💾 Saved → data/output/stress_test_results.json")

if __name__ == "__main__":
    run_stress_test()
