"""
Ablation Study
--------------
Removes one component at a time and measures Sharpe impact.
Proves each component earns its place in the system.

Components tested:
  Full system          → Sharpe ???  (baseline for comparison)
  No PatchTST          → Sharpe ???  (replace with momentum proxy)
  No HMM               → Sharpe ???  (remove regime feature)
  No macro features    → Sharpe ???  (remove VIX/rates/sentiment)
  No sentiment         → Sharpe ???  (remove FinBERT score)
  Rule-based only      → Sharpe -0.40 (already known)
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
from src.rl.rl_trainer  import rule_based_baseline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Best weights from Experiment 3b
BEST_WEIGHTS = [1.2, 1.2, 1.0, 0.6]  # ptst, hmm, tech, macro

ASSETS = [
    ("^GSPC",   "S&P 500", "SAC"),
    ("BTC-USD", "Bitcoin", "PPO"),
    ("GC=F",    "Gold",    "SAC"),
]


def apply_weights(signals, weights):
    weighted = signals.copy()
    groups = {
        "patchtst":  ([0,1],   weights[0]),
        "hmm":       ([2],     weights[1]),
        "technical": (list(range(3,10)), weights[2]),
        "macro":     (list(range(10,14)), weights[3]),
    }
    for cols, w in groups.values():
        for c in cols:
            if c < weighted.shape[1]:
                weighted[:,c] *= w
    return weighted


def train_eval(df_tr, df_te, sig_tr, sig_te,
               agent_type, n_ep=100, n_eval=5):
    env_tr = TradingEnvironment(df_tr, sig_tr)
    env_te = TradingEnvironment(df_te, sig_te)
    state_dim = env_tr.state_dim

    if agent_type == "SAC":
        agent = SACAgent(state_dim=state_dim, device=DEVICE)
        best_sh=-999; best_st=None
        for ep in range(n_ep):
            state=env_tr.reset()
            while True:
                a=agent.select_action(state)
                ns,r,done,_=env_tr.step(a)
                agent.store(state,a,r,ns,done)
                agent.update()
                state=ns
                if done: break
            p=env_tr.get_performance()
            if p["sharpe_ratio"]>best_sh:
                best_sh=p["sharpe_ratio"]
                best_st={k:v.cpu().clone()
                         for k,v in agent.net.state_dict().items()}
        if best_st:
            agent.net.load_state_dict(
                {k:v.to(agent.device) for k,v in best_st.items()})
    else:
        agent=PPOAgent(state_dim=state_dim,device=DEVICE,n_steps=256)
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

    sharpes=[]
    for _ in range(n_eval):
        state=env_te.reset()
        while True:
            if hasattr(agent,'policy'):
                a,_,_=agent.select_action(state,deterministic=False)
            else:
                a=agent.select_action(state,deterministic=False)
            state,_,done,_=env_te.step(a)
            if done: break
        sharpes.append(env_te.get_performance()["sharpe_ratio"])
    return float(np.mean(sharpes))


def ablate_signals(sig, ablation):
    """Zero out specific signal groups."""
    s = sig.copy()
    if ablation == "no_patchtst":
        # Replace PatchTST cols with simple momentum
        s[:,0] = sig[:,8]  # mom5 as proxy
        s[:,1] = (sig[:,8]>0).astype(float)
    elif ablation == "no_hmm":
        s[:,2] = 0.0  # zero out regime
    elif ablation == "no_macro":
        s[:,10:14] = 0.0  # zero out VIX/rates/sentiment
    elif ablation == "no_sentiment":
        s[:,13] = 0.0  # zero out sentiment only
    return s


def run_ablation():
    print("="*65)
    print("  ABLATION STUDY")
    print("  Removing one component at a time")
    print("  Measures contribution of each component to Sharpe")
    print("="*65)

    ABLATIONS = [
        ("Full system",    None),
        ("No PatchTST",    "no_patchtst"),
        ("No HMM regime",  "no_hmm"),
        ("No macro",       "no_macro"),
        ("No sentiment",   "no_sentiment"),
    ]

    all_results = {a[0]:{} for a in ABLATIONS}
    full_sharpes = {}

    for sym, name, agent_type in ASSETS:
        print(f"\n── {name} ({sym}) ──────────────────────")

        df=yf.download(sym,start="2004-01-01",end="2026-03-25",progress=False)
        if hasattr(df.columns,'droplevel'): df.columns=df.columns.droplevel(1)
        df=df.dropna()

        n_train=int(len(df)*0.70)
        df_tr=df.iloc[:n_train].reset_index(drop=True)
        df_te=df.iloc[n_train:].reset_index(drop=True)

        # Build full signals once
        ptst=PatchTSTForecaster()
        ptst.fit_from_df(df_tr,verbose=False)
        sig_tr=build_patchtst_signals(df_tr,ptst)
        mac_tr=build_macro_signals(df_tr, asset_sym=sym)
        sig_tr=np.concatenate([sig_tr,mac_tr],axis=1)
        sig_tr=apply_weights(sig_tr, BEST_WEIGHTS)

        sig_te=build_patchtst_signals(df_te,ptst)
        mac_te=build_macro_signals(df_te, asset_sym=sym)
        sig_te=np.concatenate([sig_te,mac_te],axis=1)
        sig_te=apply_weights(sig_te, BEST_WEIGHTS)

        print(f"  {'Ablation':<20} {'Sharpe':>8} {'Impact':>10}")
        print(f"  {'─'*42}")

        for abl_name, abl_type in ABLATIONS:
            # Apply ablation
            s_tr = ablate_signals(sig_tr, abl_type) \
                   if abl_type else sig_tr
            s_te = ablate_signals(sig_te, abl_type) \
                   if abl_type else sig_te

            sharpe = train_eval(df_tr, df_te, s_tr, s_te,
                               agent_type, n_ep=100)
            all_results[abl_name][name] = sharpe

            if abl_name == "Full system":
                full_sharpes[name] = sharpe
                impact_str = "(baseline)"
            else:
                impact = sharpe - full_sharpes[name]
                direction = "✅ helps" if impact > 0.02 \
                           else "❌ hurts" if impact < -0.02 \
                           else "~ neutral"
                impact_str = f"{impact:+.3f} {direction}"

            print(f"  {abl_name:<20} {sharpe:>+7.3f}  {impact_str}")

    # Summary
    print(f"\n{'='*65}")
    print(f"  ABLATION SUMMARY — Avg Sharpe across assets")
    print(f"{'='*65}")
    print(f"  {'Component':<22} {'S&P500':>8} {'BTC':>8} "
          f"{'Gold':>8} {'Avg':>8} {'Contribution':>14}")
    print(f"  {'─'*65}")

    full_avg = np.mean(list(full_sharpes.values()))

    for abl_name, _ in ABLATIONS:
        sharpes = [all_results[abl_name].get(n,0)
                   for _,n,_ in ASSETS]
        avg = float(np.mean(sharpes))
        if abl_name == "Full system":
            contrib = "(baseline)"
        else:
            contrib = f"{avg-full_avg:+.3f} " + \
                     ("← important" if full_avg-avg > 0.1
                      else "← minor" if full_avg-avg > 0.02
                      else "← negligible")
        print(f"  {abl_name:<22} "
              f"{sharpes[0]:>+7.3f}  {sharpes[1]:>+7.3f}  "
              f"{sharpes[2]:>+7.3f}  {avg:>+7.3f}  {contrib}")

    # Save
    with open("data/output/ablation_results.json","w") as f:
        json.dump({"full_avg":full_avg,
                   "results":all_results}, f, indent=2)
    print(f"\n  💾 Saved → data/output/ablation_results.json")
    print(f"\n  Components that matter most:")
    print(f"  (largest drop when removed = most important)")


if __name__ == "__main__":
    run_ablation()
