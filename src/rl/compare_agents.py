"""
PPO vs SAC vs Rule-Based — Final Comparison
--------------------------------------------
Trains both agents on same data, evaluates on same test set.
Produces the academic comparison table for the report.
"""

import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from datetime import datetime
from src.rl.trading_env  import TradingEnvironment
from src.rl.ppo_agent    import PPOAgent
from src.rl.sac_agent    import SACAgent
from src.rl.rl_trainer   import rule_based_baseline
from src.rl.integrated_pipeline import (
    build_patchtst_signals, run_integrated)
from src.forecast.patchtst_forecast import PatchTSTForecaster
from src.rl.macro_sentiment_features import build_macro_signals

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EPISODES = 200
N_EVAL_RUNS = 5

ASSETS = [
    ("^GSPC",   "S&P 500",  "equity"),
    ("BTC-USD", "Bitcoin",  "crypto"),
    ("GC=F",    "Gold",     "commodity"),
]


def train_evaluate_ppo(env_train, env_test, state_dim, label="PPO"):
    agent = PPOAgent(state_dim=state_dim, device=DEVICE, n_steps=256)
    best_sharpe=-999; best_state=None

    for ep in range(N_EPISODES):
        state=env_train.reset(); steps=0
        while True:
            a,lp,v=agent.select_action(state)
            ns,r,done,_=env_train.step(a)
            agent.store(state,a,lp,r,v,done)
            state=ns; steps+=1
            if steps%256==0:
                _,_,lv=agent.select_action(state)
                agent.update(lv)
            if done: break
        p=env_train.get_performance()
        if p["sharpe_ratio"]>best_sharpe:
            best_sharpe=p["sharpe_ratio"]
            best_state={k:v.cpu().clone()
                        for k,v in agent.policy.state_dict().items()}

    if best_state:
        agent.policy.load_state_dict(
            {k:v.to(agent.device) for k,v in best_state.items()})

    # Evaluate
    returns=[]; sharpes=[]; dds=[]
    for _ in range(N_EVAL_RUNS):
        state=env_test.reset()
        while True:
            a,_,_=agent.select_action(state,deterministic=False)
            state,_,done,_=env_test.step(a)
            if done: break
        p=env_test.get_performance()
        returns.append(p["total_return"])
        sharpes.append(p["sharpe_ratio"])
        dds.append(p["max_drawdown"])

    return {"label":label,
            "return": float(np.mean(returns)),
            "sharpe": float(np.mean(sharpes)),
            "dd":     float(np.mean(dds)),
            "train_sharpe": best_sharpe}


def train_evaluate_sac(env_train, env_test, state_dim, label="SAC"):
    agent = SACAgent(state_dim=state_dim, device=DEVICE)
    best_sharpe=-999; best_state=None

    for ep in range(N_EPISODES):
        state=env_train.reset()
        while True:
            a=agent.select_action(state)
            ns,r,done,_=env_train.step(a)
            agent.store(state,a,r,ns,done)
            agent.update()  # SAC updates every step
            state=ns
            if done: break
        p=env_train.get_performance()
        if p["sharpe_ratio"]>best_sharpe:
            best_sharpe=p["sharpe_ratio"]
            best_state={k:v.cpu().clone()
                        for k,v in agent.net.state_dict().items()}

    if best_state:
        agent.net.load_state_dict(
            {k:v.to(agent.device) for k,v in best_state.items()})

    # Evaluate
    returns=[]; sharpes=[]; dds=[]
    for _ in range(N_EVAL_RUNS):
        state=env_test.reset()
        while True:
            a=agent.select_action(state,deterministic=False)
            state,_,done,_=env_test.step(a)
            if done: break
        p=env_test.get_performance()
        returns.append(p["total_return"])
        sharpes.append(p["sharpe_ratio"])
        dds.append(p["max_drawdown"])

    return {"label":label,
            "return": float(np.mean(returns)),
            "sharpe": float(np.mean(sharpes)),
            "dd":     float(np.mean(dds)),
            "train_sharpe": best_sharpe}


def run_comparison():
    all_results = []
    os.makedirs("data/output", exist_ok=True)

    for sym, name, aclass in ASSETS:
        print(f"\n{'='*65}")
        print(f"  {name} ({sym})")
        print(f"{'='*65}")

        df = yf.download(sym, start="2004-01-01",
                         end="2026-03-25", progress=False)
        if hasattr(df.columns,'droplevel'):
            df.columns=df.columns.droplevel(1)
        df=df.dropna()

        n_train=int(len(df)*0.70)
        df_tr=df.iloc[:n_train].reset_index(drop=True)
        df_te=df.iloc[n_train:].reset_index(drop=True)
        print(f"  Train: {len(df_tr)}  Test: {len(df_te)}")

        # PatchTST
        print(f"  Training PatchTST...")
        ptst=PatchTSTForecaster()
        ptst.fit_from_df(df_tr, verbose=False)

        # Build enriched signals
        print(f"  Building signals...")
        sig_tr=build_patchtst_signals(df_tr, ptst)
        mac_tr=build_macro_signals(df_tr, asset_sym=sym)
        sig_tr=np.concatenate([sig_tr, mac_tr], axis=1)

        sig_te=build_patchtst_signals(df_te, ptst)
        mac_te=build_macro_signals(df_te, asset_sym=sym)
        sig_te=np.concatenate([sig_te, mac_te], axis=1)

        env_tr=TradingEnvironment(df_tr, sig_tr)
        env_te=TradingEnvironment(df_te, sig_te)
        state_dim=env_tr.state_dim
        print(f"  State dim: {state_dim}")

        # Rule-based
        rule=rule_based_baseline(df_te)
        rule_res={"label":"Rule-Based",
                  "return":rule["total_return"],
                  "sharpe":rule["sharpe_ratio"],
                  "dd":rule["max_drawdown"]}

        # PPO
        print(f"  Training PPO ({N_EPISODES} episodes)...")
        ppo_res=train_evaluate_ppo(env_tr, env_te, state_dim)

        # SAC
        print(f"  Training SAC ({N_EPISODES} episodes)...")
        sac_res=train_evaluate_sac(env_tr, env_te, state_dim)

        # Results
        print(f"\n  RESULTS ({name}):")
        for res in [ppo_res, sac_res, rule_res]:
            print(f"  {res['label']:<15} "
                  f"return={res['return']:+.2%}  "
                  f"sharpe={res['sharpe']:+.2f}  "
                  f"dd={res['dd']:.2%}")

        winner=max([ppo_res,sac_res],key=lambda x:x["sharpe"])["label"]
        all_results.append({
            "asset":name,"class":aclass,
            "ppo":ppo_res,"sac":sac_res,"rule":rule_res,
            "winner":winner
        })

    # Final table
    print(f"\n{'='*70}")
    print(f"  FINAL: PPO vs SAC vs RULE-BASED (Week 2 — with macro+sentiment)")
    print(f"{'='*70}")
    print(f"  {'Asset':<10} {'PPO Ret':>9} {'SAC Ret':>9} "
          f"{'Rule Ret':>9} {'PPO Sh':>8} {'SAC Sh':>8} {'Winner':>8}")
    print(f"  {'─'*65}")

    for r in all_results:
        print(f"  {r['asset']:<10} "
              f"{r['ppo']['return']:>+8.1%}  "
              f"{r['sac']['return']:>+8.1%}  "
              f"{r['rule']['return']:>+8.1%}  "
              f"{r['ppo']['sharpe']:>+7.2f}  "
              f"{r['sac']['sharpe']:>+7.2f}  "
              f"{r['winner']:>8}")

    avg_ppo  = np.mean([r['ppo']['sharpe']  for r in all_results])
    avg_sac  = np.mean([r['sac']['sharpe']  for r in all_results])
    avg_rule = np.mean([r['rule']['sharpe'] for r in all_results])
    print(f"\n  Avg Sharpe — PPO: {avg_ppo:.2f}  "
          f"SAC: {avg_sac:.2f}  Rule: {avg_rule:.2f}")
    overall = "PPO" if avg_ppo>avg_sac else "SAC"
    print(f"  Overall winner: {overall}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "summary": {
            "avg_ppo_sharpe":  avg_ppo,
            "avg_sac_sharpe":  avg_sac,
            "avg_rule_sharpe": avg_rule,
            "overall_winner":  overall
        }
    }
    with open("data/output/ppo_vs_sac_results.json","w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  💾 Results saved → data/output/ppo_vs_sac_results.json")


if __name__ == "__main__":
    run_comparison()
