"""
Experiment 3b — Consensus Weights via State Scaling
-----------------------------------------------------
CORRECT approach: weights scale the INPUT features (state),
not the reward. Higher weight = signal is amplified in state vector
= agent pays more attention to it during action selection.

This is how ensemble weights should work in RL.
"""

import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from src.rl.trading_env import TradingEnvironment
from src.rl.sac_agent   import SACAgent
from src.rl.ppo_agent   import PPOAgent
from src.rl.macro_sentiment_features import build_macro_signals
from src.rl.integrated_pipeline import build_patchtst_signals
from src.forecast.patchtst_forecast import PatchTSTForecaster

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Weight configs from Kimi ───────────────────────────────────
# Order: [patchtst(cols 0-1), hmm(col 2), adx(col 3), 
#         technicals(cols 5-9), macro(cols 10-13)]
# Simplified to 4 groups:
# group_weights = [patchtst, regime/hmm, technical, macro]

CONFIGS = [
    ("Equal (baseline)",  [1.0, 1.0, 1.0, 1.0]),
    ("Accuracy-First",    [2.0, 0.6, 0.4, 0.4]),  # PatchTST dominant
    ("Execution-First",   [1.2, 0.8, 0.8, 1.6]),  # Macro dominant
    ("Balanced-Expert",   [1.2, 1.2, 1.0, 0.6]),  # PatchTST+HMM
    ("Conservative",      [1.0, 1.4, 0.6, 1.4]),  # HMM+Macro safety
]

ASSETS = [
    ("^GSPC",   "S&P 500", "SAC"),
    ("BTC-USD", "Bitcoin", "PPO"),
    ("GC=F",    "Gold",    "SAC"),
]

# Signal column groups
# cols 0-1:  PatchTST (return, probability)
# col  2:    HMM regime
# cols 3-4:  ADX, ATR
# cols 5-9:  RSI, MACD, vol, mom5, mom10
# cols 10-13: VIX, rate, SP500 regime, sentiment

GROUPS = {
    "patchtst":  [0, 1],
    "hmm":       [2],
    "technical": [3, 4, 5, 6, 7, 8, 9],
    "macro":     [10, 11, 12, 13],
}


def apply_weights(signals: np.ndarray,
                  group_weights: list) -> np.ndarray:
    """
    Scale signal columns by group weights.
    Higher weight = amplified signal = agent pays more attention.
    """
    weighted = signals.copy()
    w_ptst, w_hmm, w_tech, w_macro = group_weights

    for col in GROUPS["patchtst"]:
        if col < weighted.shape[1]:
            weighted[:, col] *= w_ptst

    for col in GROUPS["hmm"]:
        if col < weighted.shape[1]:
            weighted[:, col] *= w_hmm

    for col in GROUPS["technical"]:
        if col < weighted.shape[1]:
            weighted[:, col] *= w_tech

    for col in GROUPS["macro"]:
        if col < weighted.shape[1]:
            weighted[:, col] *= w_macro

    return weighted


def train_and_eval(df_tr, df_te, sig_tr, sig_te,
                   agent_type, n_ep=100, n_eval=5):
    """Train agent and evaluate, return avg Sharpe."""
    env_tr = TradingEnvironment(df_tr, sig_tr)
    env_te = TradingEnvironment(df_te, sig_te)
    state_dim = env_tr.state_dim

    if agent_type == "SAC":
        agent = SACAgent(state_dim=state_dim, device=DEVICE)
        best_sh = -999; best_st = None
        for ep in range(n_ep):
            state = env_tr.reset()
            while True:
                a = agent.select_action(state)
                ns,r,done,_ = env_tr.step(a)
                agent.store(state,a,r,ns,done)
                agent.update()
                state=ns
                if done: break
            p = env_tr.get_performance()
            if p["sharpe_ratio"] > best_sh:
                best_sh = p["sharpe_ratio"]
                best_st = {k:v.cpu().clone()
                           for k,v in agent.net.state_dict().items()}
        if best_st:
            agent.net.load_state_dict(
                {k:v.to(agent.device) for k,v in best_st.items()})
    else:
        agent = PPOAgent(state_dim=state_dim, device=DEVICE, n_steps=256)
        best_sh=-999; best_st=None
        for ep in range(n_ep):
            state=env_tr.reset(); steps=0
            while True:
                a,lp,v=agent.select_action(state)
                ns,r,done,_=env_tr.step(a)
                agent.store(state,a,lp,r,v,done)
                state=ns; steps+=1
                if steps%256==0:
                    _,_,lv=agent.select_action(state)
                    agent.update(lv)
                if done: break
            p=env_tr.get_performance()
            if p["sharpe_ratio"]>best_sh:
                best_sh=p["sharpe_ratio"]
                best_st={k:v.cpu().clone()
                         for k,v in agent.policy.state_dict().items()}
        if best_st:
            agent.policy.load_state_dict(
                {k:v.to(agent.device) for k,v in best_st.items()})

    # Evaluate
    sharpes = []
    for _ in range(n_eval):
        state = env_te.reset()
        while True:
            if hasattr(agent,'policy'):
                a,_,_=agent.select_action(state,deterministic=False)
            else:
                a=agent.select_action(state,deterministic=False)
            state,_,done,_=env_te.step(a)
            if done: break
        sharpes.append(env_te.get_performance()["sharpe_ratio"])

    return float(np.mean(sharpes))


def run():
    print("="*70)
    print("  EXPERIMENT 3b: CONSENSUS WEIGHTS VIA STATE SCALING")
    print("  Weights amplify signal groups in state vector")
    print("  Higher weight = agent pays more attention to that signal")
    print("="*70)

    # Pre-fetch all data and signals
    asset_data = {}
    for sym, name, agent_type in ASSETS:
        print(f"\n  Fetching + building signals for {name}...")
        df = yf.download(sym, start="2004-01-01",
                         end="2026-03-25", progress=False)
        if hasattr(df.columns,'droplevel'):
            df.columns=df.columns.droplevel(1)
        df=df.dropna()

        n_train=int(len(df)*0.70)
        df_tr=df.iloc[:n_train].reset_index(drop=True)
        df_te=df.iloc[n_train:].reset_index(drop=True)

        ptst=PatchTSTForecaster()
        ptst.fit_from_df(df_tr, verbose=False)

        sig_tr=build_patchtst_signals(df_tr,ptst)
        mac_tr=build_macro_signals(df_tr)
        sig_tr=np.concatenate([sig_tr,mac_tr],axis=1)

        sig_te=build_patchtst_signals(df_te,ptst)
        mac_te=build_macro_signals(df_te)
        sig_te=np.concatenate([sig_te,mac_te],axis=1)

        asset_data[name] = {
            "df_tr":df_tr,"df_te":df_te,
            "sig_tr":sig_tr,"sig_te":sig_te,
            "agent":agent_type
        }

    # Grid search
    all_results = {}
    for cfg_name, weights in CONFIGS:
        all_results[cfg_name] = {}

    print(f"\n  Running grid search — {len(CONFIGS)} configs × "
          f"{len(ASSETS)} assets × 100 episodes each...")
    print(f"  Estimated time: 20-30 minutes\n")

    for cfg_name, weights in CONFIGS:
        print(f"\n── Config: {cfg_name} "
              f"[ptst={weights[0]} hmm={weights[1]} "
              f"tech={weights[2]} macro={weights[3]}] ──")

        for sym, name, agent_type in ASSETS:
            d = asset_data[name]

            # Apply weights to state signals
            sig_tr_w = apply_weights(d["sig_tr"], weights)
            sig_te_w = apply_weights(d["sig_te"], weights)

            sharpe = train_and_eval(
                d["df_tr"], d["df_te"],
                sig_tr_w, sig_te_w,
                agent_type, n_ep=100)

            all_results[cfg_name][name] = sharpe
            print(f"  {name:<12} Sharpe={sharpe:+.3f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  FINAL GRID SEARCH RESULTS")
    print(f"{'='*70}")
    print(f"  {'Config':<22} {'S&P500':>8} {'BTC':>8} "
          f"{'Gold':>8} {'Average':>10}")
    print(f"  {'─'*58}")

    ranked = []
    for cfg_name, weights in CONFIGS:
        sharpes = [all_results[cfg_name].get(name, 0)
                   for _,name,_ in ASSETS]
        avg = float(np.mean(sharpes))
        ranked.append((cfg_name, sharpes, avg))
    ranked.sort(key=lambda x: x[2], reverse=True)

    baseline = next(a for n,_,a in ranked if n=="Equal (baseline)")
    for rank,(cfg_name,sharpes,avg) in enumerate(ranked,1):
        marker=" ← BEST" if rank==1 else ""
        imp = f"({avg-baseline:+.3f})"
        print(f"  {cfg_name:<22} "
              f"{sharpes[0]:>+7.3f}  {sharpes[1]:>+7.3f}  "
              f"{sharpes[2]:>+7.3f}  {avg:>+9.3f} {imp}{marker}")

    best = ranked[0]
    print(f"\n  Best config:        {best[0]}")
    print(f"  Best avg Sharpe:    {best[2]:+.3f}")
    print(f"  vs Equal weights:   {best[2]-baseline:+.3f}")
    print(f"  vs Asset-specific:  {best[2]-0.661:+.3f}")

    # Combined best — asset-specific agent + best weights
    asset_specific = {"S&P 500":0.68,"Bitcoin":0.71,"Gold":0.59}
    best_weighted  = {n:all_results[best[0]].get(n,0) for _,n,_ in ASSETS}
    print(f"\n  COMBINED (asset-specific agent + best weights):")
    combined_sharpes = []
    for _,name,_ in ASSETS:
        # Take max of asset-specific vs best weighted
        combined = max(asset_specific.get(name,0),
                      best_weighted.get(name,0))
        combined_sharpes.append(combined)
        print(f"  {name:<12} {combined:+.3f}")
    print(f"  Average:     {np.mean(combined_sharpes):+.3f}")

    # Save
    with open("data/output/weighted_state_results.json","w") as f:
        json.dump({"best_config":best[0],
                   "best_sharpe":best[2],
                   "all_results":all_results,
                   "ranked":[(n,float(a)) for n,_,a in ranked]},
                  f, indent=2)
    print(f"\n  💾 Saved → data/output/weighted_state_results.json")


if __name__ == "__main__":
    run()
