"""
Train all missing PatchTST + SAC/PPO models for the full 40-asset universe.
Skips assets that already have saved models.
Runs overnight — takes 2-4 hours for all 39 assets.
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from datetime import datetime

from src.forecast.patchtst_forecast import PatchTSTForecaster
from src.rl.trading_env             import TradingEnvironment
from src.rl.sac_agent               import SACAgent
from src.rl.ppo_agent               import PPOAgent
from src.rl.macro_sentiment_features import build_macro_signals
from src.rl.integrated_pipeline     import build_patchtst_signals

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "data/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Full 40-asset universe with asset class and agent type
UNIVERSE = [
    # Indices — equity class, SAC
    ("^GSPC",    "S&P 500",        "equity",    "SAC"),
    ("^IXIC",    "NASDAQ",         "equity",    "SAC"),
    ("^DJI",     "Dow Jones",      "equity",    "SAC"),
    # US Stocks — equity class, SAC
    ("AAPL",     "Apple",          "equity",    "SAC"),
    ("MSFT",     "Microsoft",      "equity",    "SAC"),
    ("NVDA",     "NVIDIA",         "equity",    "SAC"),
    ("TSLA",     "Tesla",          "equity",    "SAC"),
    ("AMZN",     "Amazon",         "equity",    "SAC"),
    ("META",     "Meta",           "equity",    "SAC"),
    ("GOOGL",    "Alphabet",       "equity",    "SAC"),
    ("JPM",      "JPMorgan",       "equity",    "SAC"),
    # Crypto — crypto class, PPO
    ("BTC-USD",  "Bitcoin",        "crypto",    "PPO"),
    ("ETH-USD",  "Ethereum",       "crypto",    "PPO"),
    ("SOL-USD",  "Solana",         "crypto",    "PPO"),
    ("BNB-USD",  "BNB",            "crypto",    "PPO"),
    ("XRP-USD",  "XRP",            "crypto",    "PPO"),
    # Forex Major — forex class, SAC
    ("EURUSD=X", "EUR/USD",        "forex",     "SAC"),
    ("GBPUSD=X", "GBP/USD",        "forex",     "SAC"),
    ("USDJPY=X", "USD/JPY",        "forex",     "SAC"),
    ("USDCHF=X", "USD/CHF",        "forex",     "SAC"),
    ("AUDUSD=X", "AUD/USD",        "forex",     "SAC"),
    ("NZDUSD=X", "NZD/USD",        "forex",     "SAC"),
    ("USDCAD=X", "USD/CAD",        "forex",     "SAC"),
    # Forex Cross — forex class, SAC
    ("EURGBP=X", "EUR/GBP",        "forex",     "SAC"),
    ("EURJPY=X", "EUR/JPY",        "forex",     "SAC"),
    ("GBPJPY=X", "GBP/JPY",        "forex",     "SAC"),
    ("AUDNZD=X", "AUD/NZD",        "forex",     "SAC"),
    # African/Emerging — forex class, SAC
    ("USDNGN=X", "USD/NGN",        "forex",     "SAC"),
    ("EURNGN=X", "EUR/NGN",        "forex",     "SAC"),
    ("USDZAR=X", "USD/ZAR",        "forex",     "SAC"),
    ("USDKES=X", "USD/KES",        "forex",     "SAC"),
    ("USDGHS=X", "USD/GHS",        "forex",     "SAC"),
    # Commodities — commodity class, SAC
    ("GC=F",     "Gold",           "commodity", "SAC"),
    ("SI=F",     "Silver",         "commodity", "SAC"),
    ("CL=F",     "Crude Oil",      "commodity", "SAC"),
    ("NG=F",     "Natural Gas",    "commodity", "SAC"),
    ("HG=F",     "Copper",         "commodity", "SAC"),
    ("ZW=F",     "Wheat",          "commodity", "SAC"),
    ("ZC=F",     "Corn",           "commodity", "SAC"),
]

BEST_WEIGHTS = [1.2, 1.2, 1.0, 0.6]

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

def train_asset(sym, name, asset_class, agent_type):
    key = safe_key(sym)
    ptst_path = f"{MODELS_DIR}/patchtst_{key}.pt"
    rl_path   = f"{MODELS_DIR}/{agent_type.lower()}_{key}.pt"

    print(f"\n{'='*60}")
    print(f"  {name} ({sym})  [{asset_class}] [{agent_type}]")
    print(f"{'='*60}")

    # ── Fetch data ─────────────────────────────────────────────
    try:
        df = yf.download(sym, start="2004-01-01",
                         end=datetime.now().strftime("%Y-%m-%d"),
                         progress=False)
        if hasattr(df.columns, 'droplevel'):
            df.columns = df.columns.droplevel(1)
        df = df.dropna()
        if len(df) < 200:
            print(f"  ⚠️  Only {len(df)} rows — skipping")
            return False
        print(f"  Data: {len(df)} rows  "
              f"({df.index[0].date()} → {df.index[-1].date()})")
    except Exception as e:
        print(f"  ❌ Data fetch failed: {e}")
        return False

    n_train = int(len(df) * 0.70)
    df_train = df.iloc[:n_train].reset_index(drop=True)

    # ── PatchTST ───────────────────────────────────────────────
    if os.path.exists(ptst_path):
        print(f"  📂 PatchTST already saved — skipping training")
        ptst = PatchTSTForecaster()
        try:
            ptst.load(ptst_path)
        except Exception:
            print(f"  ⚠️  Load failed — retraining")
            os.remove(ptst_path)
            ptst = PatchTSTForecaster()
            ptst.fit_from_df(df_train, verbose=False)
            ptst.save(ptst_path)
    else:
        print(f"  Training PatchTST on {len(df_train)} rows...")
        ptst = PatchTSTForecaster()
        ptst.fit_from_df(df_train, verbose=False)
        ptst.save(ptst_path)
        print(f"  ✅ PatchTST saved → {ptst_path}")

    # ── RL Agent ───────────────────────────────────────────────
    if os.path.exists(rl_path):
        print(f"  📂 {agent_type} already saved — skipping training")
        return True

    print(f"  Building signals for RL training...")
    try:
        sig = build_patchtst_signals(df_train, ptst)
        mac = build_macro_signals(df_train, asset_sym=sym)
        sig = np.concatenate([sig, mac], axis=1)
        sig = apply_weights(sig, BEST_WEIGHTS)
    except Exception as e:
        print(f"  ❌ Signal build failed: {e}")
        return False

    env       = TradingEnvironment(df_train, sig)
    state_dim = env.state_dim

    print(f"  Training {agent_type} agent "
          f"(state_dim={state_dim}, 150 episodes)...")

    best_sharpe = -999
    best_state  = None

    if agent_type == "SAC":
        agent = SACAgent(state_dim=state_dim, device=DEVICE)
        for ep in range(150):
            state = env.reset()
            while True:
                a = agent.select_action(state)
                ns,r,done,_ = env.step(a)
                agent.store(state,a,r,ns,done)
                agent.update()
                state = ns
                if done: break
            perf = env.get_performance()
            if perf["sharpe_ratio"] > best_sharpe:
                best_sharpe = perf["sharpe_ratio"]
                best_state = {k:v.cpu().clone()
                              for k,v in agent.net.state_dict().items()}
            if (ep+1) % 50 == 0:
                print(f"    ep {ep+1}/150  best_sharpe={best_sharpe:.3f}")
        if best_state:
            agent.net.load_state_dict(
                {k:v.to(DEVICE) for k,v in best_state.items()})
        torch.save(agent.net.state_dict(), rl_path)

    else:  # PPO
        agent = PPOAgent(state_dim=state_dim, device=DEVICE, n_steps=256)
        for ep in range(150):
            state = env.reset(); steps = 0
            while True:
                a,lp,v = agent.select_action(state)
                ns,r,done,_ = env.step(a)
                agent.store(state,a,lp,r,v,done)
                state = ns; steps += 1
                if steps % 256 == 0:
                    _,_,lv = agent.select_action(state)
                    agent.update(lv)
                if done: break
            perf = env.get_performance()
            if perf["sharpe_ratio"] > best_sharpe:
                best_sharpe = perf["sharpe_ratio"]
                best_state = {k:v.cpu().clone()
                              for k,v in agent.policy.state_dict().items()}
            if (ep+1) % 50 == 0:
                print(f"    ep {ep+1}/150  best_sharpe={best_sharpe:.3f}")
        if best_state:
            agent.policy.load_state_dict(
                {k:v.to(DEVICE) for k,v in best_state.items()})
        torch.save(agent.policy.state_dict(), rl_path)

    print(f"  ✅ {agent_type} saved → {rl_path}  "
          f"best_sharpe={best_sharpe:.3f}")
    return True

def main():
    print("="*60)
    print("  TRAINING ALL 39 ASSETS — FULL UNIVERSE")
    print(f"  Device: {DEVICE}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    results = []
    for sym, name, asset_class, agent_type in UNIVERSE:
        try:
            ok = train_asset(sym, name, asset_class, agent_type)
            results.append({"sym":sym, "ok":ok})
        except Exception as e:
            print(f"  ❌ {sym} failed: {e}")
            results.append({"sym":sym, "ok":False, "error":str(e)})

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Successful: {sum(1 for r in results if r['ok'])}/{len(results)}")
    failed = [r['sym'] for r in results if not r['ok']]
    if failed:
        print(f"  Failed: {failed}")
    print("="*60)

    with open("data/output/training_log_v2.json","w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
