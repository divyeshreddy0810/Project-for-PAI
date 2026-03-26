"""
Consensus Weight Grid Search — Experiment 3
--------------------------------------------
Tests 5 weight configurations across 3 assets.
Finds optimal weighting of PatchTST, SAC/PPO, HMM, LightGBM.
Optimisation target: Sharpe ratio (risk-adjusted returns).
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

# ── Weight configurations from Kimi ───────────────────────────
CONFIGS = [
    # name, [patchtst, rl_agent, hmm, lgbm]
    ("Equal (baseline)",   [0.25, 0.25, 0.25, 0.25]),
    ("Accuracy-First",     [0.50, 0.25, 0.15, 0.10]),
    ("Execution-First",    [0.30, 0.40, 0.20, 0.10]),
    ("Balanced-Expert",    [0.30, 0.30, 0.25, 0.15]),
    ("Conservative",       [0.25, 0.35, 0.35, 0.05]),
]

ASSETS = [
    ("^GSPC",   "S&P 500", "SAC"),   # SAC wins on GSPC
    ("BTC-USD", "Bitcoin", "PPO"),   # PPO wins on BTC
    ("GC=F",    "Gold",    "SAC"),   # SAC wins on Gold
]


class WeightedConsensusEnv(TradingEnvironment):
    """
    Environment that applies consensus weights to the reward signal.
    Higher weight = signal has more influence on action selection.
    """
    def __init__(self, df, signals, weights,
                 patchtst_cols, hmm_col, lgbm_col, rl_col=None):
        super().__init__(df, signals)
        self.weights      = weights  # [patchtst, rl, hmm, lgbm]
        self.ptst_cols    = patchtst_cols
        self.hmm_col      = hmm_col
        self.lgbm_col     = lgbm_col

    def step(self, action):
        ns, reward, done, info = super().step(action)

        # Weight the reward by signal agreement score
        state  = self._get_state()
        ptst_signal = float(np.mean(state[self.ptst_cols]))
        hmm_signal  = float(state[self.hmm_col])
        lgbm_signal = float(state[self.lgbm_col]) if self.lgbm_col < len(state) else 0

        # Agreement score — how much do signals agree with action taken?
        signals_arr = np.array([ptst_signal, hmm_signal, lgbm_signal])
        weights_arr = np.array([self.weights[0], self.weights[2], self.weights[3]])
        weights_arr = weights_arr / weights_arr.sum()

        agreement = float(np.dot(signals_arr, weights_arr))
        # Scale reward by weighted agreement
        reward_scaled = reward * (1 + 0.3 * agreement)

        return ns, reward_scaled, done, info


def train_agent(env_train, state_dim, agent_type="SAC", n_ep=100):
    """Train SAC or PPO agent."""
    if agent_type == "SAC":
        agent = SACAgent(state_dim=state_dim, device=DEVICE)
        for ep in range(n_ep):
            state = env_train.reset()
            while True:
                a = agent.select_action(state)
                ns,r,done,_ = env_train.step(a)
                agent.store(state,a,r,ns,done)
                agent.update()
                state=ns
                if done: break
    else:
        agent = PPOAgent(state_dim=state_dim, device=DEVICE, n_steps=256)
        best_sh=-999; best_st=None
        for ep in range(n_ep):
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
            if p["sharpe_ratio"]>best_sh:
                best_sh=p["sharpe_ratio"]
                best_st={k:v.cpu().clone()
                         for k,v in agent.policy.state_dict().items()}
        if best_st:
            agent.policy.load_state_dict(
                {k:v.to(agent.device) for k,v in best_st.items()})
    return agent


def evaluate_agent(agent, env_test, n_runs=5):
    """Evaluate agent, average over n_runs."""
    sharpes=[]
    for _ in range(n_runs):
        state=env_test.reset()
        while True:
            if hasattr(agent,'policy'):
                a,_,_=agent.select_action(state,deterministic=False)
            else:
                a=agent.select_action(state,deterministic=False)
            state,_,done,_=env_test.step(a)
            if done: break
        p=env_test.get_performance()
        sharpes.append(p["sharpe_ratio"])
    return float(np.mean(sharpes))


def run_grid_search():
    print("="*70)
    print("  EXPERIMENT 3: CONSENSUS WEIGHT GRID SEARCH")
    print("  Target: Sharpe ratio  |  5 configs × 3 assets")
    print("="*70)

    all_results = {}
    for cfg_name, _ in CONFIGS:
        all_results[cfg_name] = {}

    for sym, name, best_agent in ASSETS:
        print(f"\n── {name} ({sym}) — best agent: {best_agent} ──")

        df = yf.download(sym, start="2004-01-01",
                         end="2026-03-25", progress=False)
        if hasattr(df.columns,'droplevel'):
            df.columns=df.columns.droplevel(1)
        df=df.dropna()

        n_train=int(len(df)*0.70)
        df_tr=df.iloc[:n_train].reset_index(drop=True)
        df_te=df.iloc[n_train:].reset_index(drop=True)

        # Build signals once
        ptst=PatchTSTForecaster()
        ptst.fit_from_df(df_tr, verbose=False)

        sig_tr=build_patchtst_signals(df_tr,ptst)
        mac_tr=build_macro_signals(df_tr)
        sig_tr=np.concatenate([sig_tr,mac_tr],axis=1)

        sig_te=build_patchtst_signals(df_te,ptst)
        mac_te=build_macro_signals(df_te)
        sig_te=np.concatenate([sig_te,mac_te],axis=1)

        # Signal column indices
        # sig columns: 0=ptst_ret, 1=ptst_prob, 2=hmm, 3=adx, 4=atr,
        #              5=rsi, 6=macd, 7=vol, 8=mom5, 9=mom10
        #              10=vix, 11=rate, 12=sp500_regime, 13=sentiment
        ptst_cols = [0, 1]
        hmm_col   = 2
        lgbm_col  = 5  # using RSI as LightGBM proxy in signals

        # Test each weight config
        print(f"  {'Config':<22} {'Sharpe':>8} {'vs Equal':>10}")
        print(f"  {'─'*45}")

        baseline_sharpe = None
        for cfg_name, weights in CONFIGS:
            env_tr = WeightedConsensusEnv(
                df_tr, sig_tr, weights,
                ptst_cols, hmm_col, lgbm_col)
            env_te = WeightedConsensusEnv(
                df_te, sig_te, weights,
                ptst_cols, hmm_col, lgbm_col)

            agent  = train_agent(env_tr, env_tr.state_dim,
                                 best_agent, n_ep=100)
            sharpe = evaluate_agent(agent, env_te)

            if cfg_name == "Equal (baseline)":
                baseline_sharpe = sharpe

            vs_equal = sharpe - baseline_sharpe \
                       if baseline_sharpe is not None else 0
            marker   = "★" if vs_equal > 0.02 else ""

            print(f"  {cfg_name:<22} {sharpe:>+7.3f}  "
                  f"{vs_equal:>+9.3f} {marker}")

            all_results[cfg_name][name] = sharpe

    # Summary table
    print(f"\n{'='*70}")
    print(f"  GRID SEARCH SUMMARY — Avg Sharpe across all assets")
    print(f"{'='*70}")
    print(f"  {'Config':<22} {'S&P500':>8} {'BTC':>8} "
          f"{'Gold':>8} {'Average':>10} {'Rank':>6}")
    print(f"  {'─'*65}")

    ranked = []
    for cfg_name, _ in CONFIGS:
        sharpes = [all_results[cfg_name].get(n,0)
                   for _,n,_ in ASSETS]
        avg = np.mean(sharpes)
        ranked.append((cfg_name, sharpes, avg))

    ranked.sort(key=lambda x: x[2], reverse=True)

    for rank,(cfg_name,sharpes,avg) in enumerate(ranked,1):
        marker = " ← BEST" if rank==1 else ""
        print(f"  {cfg_name:<22} "
              f"{sharpes[0]:>+7.3f}  {sharpes[1]:>+7.3f}  "
              f"{sharpes[2]:>+7.3f}  {avg:>+9.3f}  "
              f"#{rank}{marker}")

    best_cfg = ranked[0][0]
    best_avg = ranked[0][2]
    baseline = next(avg for n,_,avg in ranked if n=="Equal (baseline)")

    print(f"\n  Best config:     {best_cfg}")
    print(f"  Best avg Sharpe: {best_avg:+.3f}")
    print(f"  vs Equal weights:{best_avg-baseline:+.3f}")
    print(f"  vs Mixed agents: {best_avg-0.661:+.3f}")

    # Save
    summary = {
        "best_config":   best_cfg,
        "best_sharpe":   best_avg,
        "all_results":   all_results,
        "ranked":        [(n,float(a)) for n,_,a in ranked]
    }
    with open("data/output/grid_search_results.json","w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  💾 Saved → data/output/grid_search_results.json")


if __name__ == "__main__":
    run_grid_search()
