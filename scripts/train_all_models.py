"""
Train all missing PatchTST + LightGBM + SAC/PPO/TD3 models for the full 48-asset universe.
Skips assets that already have saved models.
Runs overnight — takes 3-5 hours for all 48 assets.
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from datetime import datetime

from src.forecast.patchtst_forecast  import PatchTSTForecaster
from src.forecast.lgbm_forecast      import LightGBMForecaster
from src.rl.trading_env              import TradingEnvironment
from src.rl.sac_agent                import SACAgent
from src.rl.ppo_agent                import PPOAgent
from src.rl.td3_agent                import TD3Agent
from src.rl.macro_sentiment_features import build_macro_signals
from src.rl.integrated_pipeline      import build_patchtst_signals
import joblib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "data/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Full 48-asset universe with asset class and agent type
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
    # GPU / Neocloud supply chain — TD3 (better for volatile news-driven equities)
    ("CRWV",     "CoreWeave",      "equity",    "TD3"),
    ("MU",       "Micron",         "equity",    "TD3"),
    ("SMCI",     "Super Micro",    "equity",    "TD3"),
    ("ARM",      "Arm Holdings",   "equity",    "TD3"),
    ("TSM",      "TSMC",           "equity",    "TD3"),
    ("VRT",      "Vertiv",         "equity",    "TD3"),
    ("MRVL",     "Marvell",        "equity",    "TD3"),
    ("NBIS",     "Nebius",         "equity",    "TD3"),
    ("IREN",     "IREN",           "equity",    "TD3"),
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
              (list(range(10,16)),weights[3])]   # 10-15: all 6 macro cols
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

    # ── LightGBM forecaster ────────────────────────────────────
    lgbm_path = f"{MODELS_DIR}/lgbm_{key}.pkl"
    if os.path.exists(lgbm_path):
        print(f"  📂 LightGBM already saved — skipping")
    else:
        print(f"  Training LightGBM on {len(df_train)} rows...")
        try:
            lgbm     = LightGBMForecaster()
            feat_df  = LightGBMForecaster.engineer_features(df_train)
            feat_arr = feat_df.fillna(0).values
            prices   = df_train["Close"].values
            lgbm.fit(feat_arr, prices)
            joblib.dump(lgbm, lgbm_path)
            print(f"  ✅ LightGBM saved → {lgbm_path}")
        except Exception as e:
            print(f"  ⚠️  LightGBM training failed: {e}")

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

    elif agent_type == "TD3":
        agent = TD3Agent(state_dim=state_dim, device=DEVICE)
        for ep in range(150):
            state = env.reset()
            while True:
                a = agent.select_action(state)
                ns, r, done, _ = env.step(a)
                agent.store(state, a, r, ns, done)
                agent.update()
                state = ns
                if done: break
            perf = env.get_performance()
            if perf["sharpe_ratio"] > best_sharpe:
                best_sharpe = perf["sharpe_ratio"]
                best_state  = {k: v.cpu().clone()
                               for k, v in agent.net.state_dict().items()}
            if (ep+1) % 50 == 0:
                print(f"    ep {ep+1}/150  best_sharpe={best_sharpe:.3f}")
        if best_state:
            agent.net.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_state.items()})
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
    print("  TRAINING ALL 48 ASSETS — FULL UNIVERSE")
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

    # ── Asset-class HMM training ───────────────────────────────
    # Train separate regime models per asset class so each class gets a
    # regime detector calibrated on its own price dynamics:
    #   equity      → ^GSPC (200-day SMA cycle)
    #   crypto      → BTC-USD (crypto bull/bear cycle)
    #   commodity   → GC=F   (commodity cycle)
    #   forex       → EURUSD=X (FX cycle — already the pretrained HMM default)
    #   semiconductor → ^SOX  (capex-driven cycle, very different from S&P)
    print(f"\n{'='*60}")
    print(f"  TRAINING ASSET-CLASS HMMs")
    print(f"{'='*60}")

    _HMM_TARGETS = [
        ("equity",        "^GSPC"),
        ("crypto",        "BTC-USD"),
        ("commodity",     "GC=F"),
        ("forex",         "EURUSD=X"),
        ("semiconductor", "^SOX"),    # semiconductor index
    ]

    import pickle
    from sklearn.preprocessing import StandardScaler
    try:
        from hmmlearn.hmm import GaussianHMM
        HMM_AVAILABLE = True
    except ImportError:
        HMM_AVAILABLE = False
        print("  ⚠️  hmmlearn not installed — skipping HMM training")

    if HMM_AVAILABLE:
        for class_name, ticker in _HMM_TARGETS:
            hmm_path = f"{MODELS_DIR}/hmm_{class_name}.pkl"
            if os.path.exists(hmm_path):
                print(f"  📂 {class_name} HMM already saved — skipping")
                continue
            print(f"\n  Training {class_name} HMM on {ticker}...")
            try:
                df_h = yf.download(ticker, start="2004-01-01",
                                   end=datetime.now().strftime("%Y-%m-%d"),
                                   progress=False)
                if hasattr(df_h.columns, "droplevel"):
                    df_h.columns = df_h.columns.droplevel(1)
                df_h = df_h.dropna()
                if len(df_h) < 500:
                    print(f"  ⚠️  Only {len(df_h)} rows — skipping")
                    continue

                # Feature engineering (mirrors pretrained_hmm.py)
                df_h["daily_return"]  = df_h["Close"].pct_change()
                df_h["log_vol_10d"]   = np.log(
                    df_h["daily_return"].rolling(10).std()
                    .replace(0, np.nan).fillna(1e-8) + 1e-8)
                delta  = df_h["Close"].diff()
                gain   = delta.clip(lower=0).rolling(14).mean()
                loss   = (-delta.clip(upper=0)).rolling(14).mean()
                df_h["RSI_norm"]      = (100-(100/(1+gain/(loss+1e-8))))/100.0
                ema12  = df_h["Close"].ewm(span=12).mean()
                ema26  = df_h["Close"].ewm(span=26).mean()
                df_h["MACD_norm"]     = (ema12 - ema26) / (df_h["Close"] + 1e-8)
                df_h["momentum_10d"]  = df_h["Close"].pct_change(10)

                feats = df_h[["daily_return","log_vol_10d","RSI_norm",
                              "MACD_norm","momentum_10d"]].dropna()
                if len(feats) < 200:
                    print(f"  ⚠️  Too few feature rows — skipping")
                    continue

                scaler = StandardScaler()
                X      = scaler.fit_transform(feats)
                model  = GaussianHMM(n_components=3, covariance_type="diag",
                                     n_iter=1000, random_state=42, tol=1e-4)
                model.fit(X)

                with open(hmm_path, "wb") as f:
                    pickle.dump({
                        "model":         model,
                        "scaler":        scaler,
                        "feature_names": list(feats.columns),
                        "training_date": datetime.now(),
                        "n_states":      3,
                        "ticker":        ticker,
                    }, f)
                print(f"  ✅ {class_name} HMM saved → {hmm_path}  "
                      f"({len(feats)} rows)")
            except Exception as e:
                print(f"  ❌ {class_name} HMM failed: {e}")

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
