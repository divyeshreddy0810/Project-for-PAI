"""
Integrated Pipeline — Week 1
-----------------------------
Wires real PatchTST predictions into RL state space.
Replaces momentum proxies with actual ML model outputs.

Pipeline:
1. Train PatchTST on full history (2004-present)
2. Generate PatchTST predictions for every timestep
3. Feed predictions into RL environment as state features
4. Train PPO on enriched multimodal state
5. Compare vs previous rule-based + proxy-state results
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from src.forecast.patchtst_forecast import (
    PatchTSTForecaster, build_features, make_sequences)
from src.rl.trading_env import TradingEnvironment
from src.rl.ppo_agent   import PPOAgent
from src.rl.rl_trainer  import rule_based_baseline
from src.rl.macro_sentiment_features import build_macro_signals

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOOKBACK = 60


def build_patchtst_signals(df: pd.DataFrame,
                            patchtst: PatchTSTForecaster,
                            hmm_model=None) -> np.ndarray:
    """
    Build signal matrix using REAL PatchTST predictions.
    Each row = one timestep's multimodal state.

    Features (13 total):
    0: patchtst_expected_return   — real ML prediction
    1: patchtst_up_probability    — real ML confidence
    2: hmm_regime                 — bull=1, sideways=0, bear=-1
    3: adx_normalised             — trend strength
    4: atr_normalised             — volatility
    5: rsi_normalised             — momentum
    6: macd_normalised            — trend direction
    7: rolling_volatility         — risk measure
    8: momentum_5d                — short momentum
    9: momentum_10d               — medium momentum
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    n = len(df)
    signals = np.zeros((n, 10), dtype=np.float32)

    # ── Technical features (all timesteps) ────────────────────
    ret   = df["Close"].pct_change().fillna(0).values
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = (100-(100/(1+gain/(loss+1e-8)))).fillna(50).values/100.0

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd  = ((ema12-ema26)/df["Close"]).fillna(0).values

    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"] -df["Close"].shift()).abs()],
                   axis=1).max(axis=1)
    atr  = (tr.rolling(14).mean()/df["Close"]).fillna(0).values
    vol  = pd.Series(ret).rolling(20).std().fillna(0.01).values
    m5   = df["Close"].pct_change(5).fillna(0).values
    m10  = df["Close"].pct_change(10).fillna(0).values

    # ADX
    pdm  = df["High"].diff().clip(lower=0)
    ndm  = (-df["Low"].diff()).clip(lower=0)
    pdm[pdm<ndm]=0; ndm[ndm<pdm]=0
    atr_s= tr.rolling(14).mean()
    pdi  = 100*(pdm.rolling(14).mean()/(atr_s+1e-8))
    ndi  = 100*(ndm.rolling(14).mean()/(atr_s+1e-8))
    dx   = 100*((pdi-ndi).abs()/(pdi+ndi+1e-8))
    adx  = dx.rolling(14).mean().fillna(20).values/100.0

    # ── Fill baseline signals ──────────────────────────────────
    signals[:,2] = np.where(ret>0.001,1,np.where(ret<-0.001,-1,0))
    signals[:,3] = adx
    signals[:,4] = atr
    signals[:,5] = rsi
    signals[:,6] = macd
    signals[:,7] = vol
    signals[:,8] = m5
    signals[:,9] = m10

    # ── Real PatchTST predictions (rolling) ───────────────────
    print(f"  🔮 Generating PatchTST predictions for {n} timesteps...")
    feat_df = build_features(df, regime_label=0)

    # Get aligned indices
    aligned_idx = feat_df.index

    predicted_count = 0
    for i in range(LOOKBACK, len(feat_df)):
        try:
            window = feat_df.values[i-LOOKBACK:i].astype(np.float32)
            scaled = patchtst.scaler.transform(window)
            x = torch.tensor(scaled).unsqueeze(0).to(patchtst.device)
            patchtst.model.eval()
            with torch.no_grad():
                pred_ret = float(patchtst.model(x).cpu().item())
            pred_ret *= patchtst.target_std

            # Map back to original df index
            orig_i = df.index.get_loc(aligned_idx[i]) \
                     if aligned_idx[i] in df.index else i

            signals[orig_i, 0] = pred_ret
            signals[orig_i, 1] = 1.0 if pred_ret > 0.003 else \
                                  0.0 if pred_ret < -0.003 else 0.5
            predicted_count += 1
        except Exception:
            # Fall back to momentum proxy
            signals[i, 0] = ret[i]
            signals[i, 1] = 1.0 if ret[i] > 0 else 0.0

    print(f"  ✅ PatchTST predictions: {predicted_count}/{n} timesteps")
    return signals


def run_integrated(sym: str, name: str,
                   model_path: str = None) -> dict:
    """
    Full integrated pipeline:
    Train PatchTST → generate signals → train PPO → evaluate.
    """
    print(f"\n{'='*60}")
    print(f"  {name} ({sym}) — INTEGRATED PIPELINE")
    print(f"{'='*60}")

    # ── 1. Fetch full history ──────────────────────────────────
    df_full = yf.download(sym, start="2004-01-01",
                          end="2026-03-25", progress=False)
    if hasattr(df_full.columns,'droplevel'):
        df_full.columns = df_full.columns.droplevel(1)
    df_full = df_full.dropna()
    print(f"  Full data: {len(df_full)} rows")

    # Train/test split
    n_train  = int(len(df_full) * 0.70)
    df_train = df_full.iloc[:n_train].reset_index(drop=True)
    df_test  = df_full.iloc[n_train:].reset_index(drop=True)
    print(f"  Train: {len(df_train)}  Test: {len(df_test)}")

    # ── 2. Train PatchTST on full training data ────────────────
    print(f"\n  [Step 1] Training PatchTST on {len(df_train)} rows...")
    patchtst = PatchTSTForecaster()
    patchtst.fit_from_df(df_train, verbose=True)

    # ── 3. Build enriched signal matrix for training ───────────
    print(f"\n  [Step 2] Building multimodal signals (train)...")
    signals_train = build_patchtst_signals(df_train, patchtst)
    # Add macro + sentiment features
    print(f"  Adding macro+sentiment features...")
    macro_train   = build_macro_signals(df_train, asset_sym=symbol)
    signals_train = np.concatenate([signals_train, macro_train], axis=1)
    print(f"  ✅ Enriched state dim: {signals_train.shape[1]} features")

    # ── 4. Train PPO on enriched signals ──────────────────────
    print(f"\n  [Step 3] Training PPO on enriched state...")
    env_train = TradingEnvironment(df_train, signals_train)
    agent     = PPOAgent(state_dim=env_train.state_dim, device=DEVICE,
                         n_steps=256)

    best_sharpe = -999
    best_state  = None

    for ep in range(200):
        state = env_train.reset()
        steps = 0
        while True:
            action, lp, val = agent.select_action(state)
            nstate, reward, done, _ = env_train.step(action)
            agent.store(state, action, lp, reward, val, done)
            state = nstate; steps += 1
            if steps % 256 == 0:
                _, _, lv = agent.select_action(state)
                agent.update(lv)
            if done: break

        perf = env_train.get_performance()
        if perf["sharpe_ratio"] > best_sharpe:
            best_sharpe = perf["sharpe_ratio"]
            best_state  = {k:v.cpu().clone()
                           for k,v in agent.policy.state_dict().items()}

        if (ep+1) % 50 == 0:
            print(f"    Ep {ep+1:3d}/200  "
                  f"sharpe={perf['sharpe_ratio']:+.2f}  "
                  f"return={perf['total_return']:+.2%}")

    if best_state:
        agent.policy.load_state_dict(
            {k:v.to(agent.device) for k,v in best_state.items()})
    print(f"  ✅ PPO trained  best_sharpe={best_sharpe:.3f}")

    # ── 5. Evaluate on test data ───────────────────────────────
    print(f"\n  [Step 4] Evaluating on test data...")
    signals_test = build_patchtst_signals(df_test, patchtst)
    macro_test   = build_macro_signals(df_test)
    signals_test = np.concatenate([signals_test, macro_test], axis=1)
    env_test     = TradingEnvironment(df_test, signals_test)

    # Average over 5 stochastic runs
    returns=[]; sharpes=[]; dds=[]
    for _ in range(5):
        state = env_test.reset()
        while True:
            action,_,_ = agent.select_action(state, deterministic=False)
            state,_,done,_ = env_test.step(action)
            if done: break
        p = env_test.get_performance()
        returns.append(p["total_return"])
        sharpes.append(p["sharpe_ratio"])
        dds.append(p["max_drawdown"])

    ppo_result = {
        "return":   float(np.mean(returns)),
        "sharpe":   float(np.mean(sharpes)),
        "dd":       float(np.mean(dds)),
    }

    # Rule-based baseline
    rule = rule_based_baseline(df_test)

    print(f"\n  RESULTS ({name} — test period):")
    print(f"  Integrated PPO+PatchTST: "
          f"return={ppo_result['return']:+.2%}  "
          f"sharpe={ppo_result['sharpe']:+.2f}  "
          f"dd={ppo_result['dd']:.2%}")
    print(f"  Rule-based baseline:     "
          f"return={rule['total_return']:+.2%}  "
          f"sharpe={rule['sharpe_ratio']:+.2f}  "
          f"dd={rule['max_drawdown']:.2%}")

    winner = "PPO+PatchTST ✅" if ppo_result["sharpe"] > rule["sharpe_ratio"] \
             else "Rule-Based ✅"
    print(f"  Winner: {winner}")

    return {
        "asset":       name,
        "ppo_return":  ppo_result["return"],
        "ppo_sharpe":  ppo_result["sharpe"],
        "ppo_dd":      ppo_result["dd"],
        "rule_return": rule["total_return"],
        "rule_sharpe": rule["sharpe_ratio"],
        "winner":      winner,
    }


if __name__ == "__main__":
    ASSETS = [
        ("^GSPC",   "S&P 500"),
        ("BTC-USD", "Bitcoin"),
        ("GC=F",    "Gold"),
    ]

    all_results = []
    for sym, name in ASSETS:
        r = run_integrated(sym, name)
        all_results.append(r)

    # Final comparison table
    print(f"\n{'='*65}")
    print(f"  WEEK 1 FINAL — INTEGRATED PPO+PatchTST vs RULE-BASED")
    print(f"{'='*65}")
    print(f"  {'Asset':<12} {'PPO+TST Ret':>12} {'Rule Ret':>10} "
          f"{'PPO Sharpe':>11} {'Winner':>15}")
    print(f"  {'─'*60}")

    ppo_wins = 0
    for r in all_results:
        if "PPO" in r["winner"]: ppo_wins+=1
        print(f"  {r['asset']:<12} {r['ppo_return']:>+11.2%}  "
              f"{r['rule_return']:>+9.2%}  "
              f"{r['ppo_sharpe']:>+10.2f}  {r['winner']}")

    print(f"\n  PPO+PatchTST wins: {ppo_wins}/{len(all_results)}")
    print(f"  Avg Sharpe PPO+PatchTST: "
          f"{np.mean([r['ppo_sharpe'] for r in all_results]):.2f}")
    print(f"  Avg Sharpe Rule-Based:   "
          f"{np.mean([r['rule_sharpe'] for r in all_results]):.2f}")
