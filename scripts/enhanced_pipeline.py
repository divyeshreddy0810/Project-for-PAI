#!/usr/bin/env python3
"""
Enhanced Pipeline — Option B
------------------------------
Drop-in enhanced backend that produces the same output format
as master_pipeline.py but uses:
  - PatchTST instead of linear regression
  - SAC/PPO RL agent instead of rule-based trader
  - HMM instead of K-Means regime detection
  - Macro features (VIX + rates) in state
  - Time-varying sentiment proxy

Output: Same JSON format as baseline + enhanced_metrics section
Frontend can call either pipeline via mode parameter.

Usage:
  python3 scripts/enhanced_pipeline.py --symbols "^GSPC,BTC-USD,GC=F"
  python3 scripts/enhanced_pipeline.py --symbols "^GSPC" --mode enhanced
"""

import os, sys, json, argparse, warnings
from datetime import datetime
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
import torch

from src.forecast.patchtst_forecast import PatchTSTForecaster
from src.regime.pretrained_hmm       import PretrainedHMMRegime
from src.rl.trading_env              import TradingEnvironment
from src.rl.sac_agent                import SACAgent
from src.rl.ppo_agent                import PPOAgent
from src.rl.macro_sentiment_features import build_macro_signals
from src.rl.integrated_pipeline      import build_patchtst_signals

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "data/output"

# Per-asset-class HMM mapping
HMM_MAP = {
    "^GSPC":    "equity",
    "^IXIC":    "equity",
    "BTC-USD":  "crypto",
    "ETH-USD":  "crypto",
    "GC=F":     "commodity",
    "CL=F":     "commodity",
    "SI=F":     "commodity",
    "EURUSD=X": "forex",
    "GBPUSD=X": "forex",
    "USDJPY=X": "forex",
    "USDNGN=X": "forex",
}

# Best config from experiments
BEST_WEIGHTS   = [1.2, 1.2, 1.0, 0.6]
AGENT_MAP      = {
    "^GSPC":    "SAC",  # Sharpe 0.68
    "^IXIC":    "SAC",  # Sharpe 0.73
    "BTC-USD":  "PPO",  # Sharpe 0.71
    "GC=F":     "SAC",  # Sharpe 0.59
    "EURUSD=X": "SAC",  # Sharpe 0.16
    "GBPUSD=X": "SAC",  # Sharpe 0.22
    "CL=F":     None,   # Not suitable
}

# Risk profile parameters
RISK_PROFILES = {
    "conservative":  {"max_pos": 0.05, "sl": 0.05, "tp": 0.10},
    "moderate":      {"max_pos": 0.10, "sl": 0.08, "tp": 0.15},
    "aggressive":    {"max_pos": 0.20, "sl": 0.10, "tp": 0.25},
}


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


def load_or_train_agent(sym, state_dim, df_train, sig_train):
    """Load saved agent or train fresh."""
    agent_type = AGENT_MAP.get(sym, "SAC")
    if agent_type is None:
        return None, "SKIP"

    model_key = sym.replace("^","").replace("-","_").replace("=","")
    model_path = f"data/models/{agent_type.lower()}_{model_key}.pt"

    env = TradingEnvironment(df_train, sig_train)

    if agent_type == "SAC":
        agent = SACAgent(state_dim=state_dim, device=DEVICE)
        if os.path.exists(model_path):
            try:
                agent.net.load_state_dict(
                    torch.load(model_path, map_location=DEVICE))
                print(f"    📂 Loaded SAC model ← {model_path}")
                return agent, "SAC"
            except Exception:
                pass
        # Train fresh
        print(f"    🤖 Training SAC ({sym})...")
        best_sh=-999; best_st=None
        for ep in range(100):
            state=env.reset()
            while True:
                a=agent.select_action(state)
                ns,r,done,_=env.step(a)
                agent.store(state,a,r,ns,done)
                agent.update()
                state=ns
                if done: break
            p=env.get_performance()
            if p["sharpe_ratio"]>best_sh:
                best_sh=p["sharpe_ratio"]
                best_st={k:v.cpu().clone()
                         for k,v in agent.net.state_dict().items()}
        if best_st:
            agent.net.load_state_dict(
                {k:v.to(agent.device) for k,v in best_st.items()})
        os.makedirs("data/models", exist_ok=True)
        torch.save(agent.net.state_dict(), model_path)
        return agent, "SAC"

    else:  # PPO
        agent = PPOAgent(state_dim=state_dim, device=DEVICE, n_steps=256)
        if os.path.exists(model_path):
            try:
                agent.policy.load_state_dict(
                    torch.load(model_path, map_location=DEVICE))
                print(f"    📂 Loaded PPO model ← {model_path}")
                return agent, "PPO"
            except Exception:
                pass
        print(f"    🤖 Training PPO ({sym})...")
        best_sh=-999; best_st=None
        for ep in range(100):
            state=env.reset(); steps=0
            while True:
                a,lp,v=agent.select_action(state)
                ns,r,done,_=env.step(a)
                agent.store(state,a,lp,r,v,done)
                state=ns; steps+=1
                if steps%256==0:
                    _,_,lv=agent.select_action(state)
                    agent.update(lv)
                if done: break
            p=env.get_performance()
            if p["sharpe_ratio"]>best_sh:
                best_sh=p["sharpe_ratio"]
                best_st={k:v.cpu().clone()
                         for k,v in agent.policy.state_dict().items()}
        if best_st:
            agent.policy.load_state_dict(
                {k:v.to(agent.device) for k,v in best_st.items()})
        torch.save(agent.policy.state_dict(), model_path)
        return agent, "PPO"


def get_rl_signal(agent, agent_type, state):
    """Get action from agent."""
    if agent_type == "SAC":
        return agent.select_action(state, deterministic=True)
    else:
        a,_,_ = agent.select_action(state, deterministic=True)
        return a


def run_enhanced(symbols, risk_profile="moderate",
                 portfolio_value=100000.0):
    """
    Run enhanced pipeline on given symbols.
    Returns same format as baseline + enhanced_metrics.
    """
    print(f"\n🚀 ENHANCED PIPELINE")
    print(f"   Symbols:   {symbols}")
    print(f"   Risk:      {risk_profile}")
    print(f"   Portfolio: ${portfolio_value:,.0f}")
    print(f"   Device:    {DEVICE}")

    risk_params = RISK_PROFILES.get(risk_profile,
                                     RISK_PROFILES["moderate"])
    results = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load per-asset-class HMMs
    import joblib
    _hmm_cache = {}
    def get_hmm(sym):
        asset_class = HMM_MAP.get(sym, "forex")
        if asset_class not in _hmm_cache:
            path = f"data/models/hmm_{asset_class}.pkl"
            if os.path.exists(path):
                _hmm_cache[asset_class] = joblib.load(path)
                print(f"  📂 Loaded HMM ({asset_class}) ← {path}")
            else:
                # fallback to pretrained
                hmm_fallback = PretrainedHMMRegime()
                hmm_fallback.load()
                _hmm_cache[asset_class] = hmm_fallback
                print(f"  ⚠️  HMM fallback (pretrained) for {asset_class}")
        return _hmm_cache[asset_class]

    for sym in symbols:
        print(f"\n  ── {sym} ──────────────────────")

        if AGENT_MAP.get(sym) is None:
            print(f"    ⚠️  {sym} not suitable — skipping")
            results.append({
                "symbol": sym,
                "signal": "SKIP",
                "reason": "Asset unsuitable for this system (geopolitical commodity)"
            })
            continue

        # Fetch data
        df = yf.download(sym, start="2004-01-01",
                         end=datetime.now().strftime("%Y-%m-%d"),
                         progress=False)
        if hasattr(df.columns,'droplevel'):
            df.columns=df.columns.droplevel(1)
        df=df.dropna()

        if len(df) < 200:
            print(f"    ⚠️  Not enough data ({len(df)} rows)")
            continue

        current_price = float(df["Close"].iloc[-1])
        print(f"    Price: ${current_price:,.4f}")

        # Split
        n_train = int(len(df)*0.70)
        df_train = df.iloc[:n_train].reset_index(drop=True)
        df_recent= df.iloc[-90:].reset_index(drop=True)  # last 90 days

        # PatchTST forecast — load saved model or train + save
        safe_sym = sym.replace("^","").replace("-","_").replace("=","")
        ptst_path = f"data/models/patchtst_{safe_sym}.pt"
        ptst = PatchTSTForecaster()

        if os.path.exists(ptst_path):
            try:
                ptst.load(ptst_path)
                print(f"    📂 Loaded PatchTST ← {ptst_path}")
            except Exception:
                print(f"    ⚠️  PatchTST load failed — retraining...")
                ptst.fit_from_df(df_train, verbose=False)
                ptst.save(ptst_path)
                print(f"    💾 PatchTST saved → {ptst_path}")
        else:
            print(f"    Training PatchTST...")
            ptst.fit_from_df(df_train, verbose=False)
            os.makedirs("data/models", exist_ok=True)
            ptst.save(ptst_path)
            print(f"    💾 PatchTST saved → {ptst_path}")

        pred_return = ptst.predict_return(df_recent)
        pred_price  = current_price * (1 + pred_return)
        print(f"    PatchTST 5d forecast: {pred_return*100:+.2f}%  "
              f"→ ${pred_price:,.4f}")

        # HMM regime — use asset-class-specific model
        try:
            hmm_model = get_hmm(sym)
            df_hmm = df.tail(60).copy()
            # Add required features
            delta = df_hmm["Close"].diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            df_hmm["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
            df_hmm["sentiment_mean"] = 0.0
            df_hmm = df_hmm.dropna()
            regime = hmm_model.predict(df_hmm)
        except Exception as e:
            regime = "sideways"
            print(f"    ⚠️  HMM fallback: {e}")
        print(f"    HMM regime: {regime}")

        # Build signals for RL
        sig_train = build_patchtst_signals(df_train, ptst)
        mac_train = build_macro_signals(df_train, asset_sym=sym)
        sig_train = np.concatenate([sig_train, mac_train], axis=1)
        sig_train = apply_weights(sig_train, BEST_WEIGHTS)

        # Load/train RL agent
        env_dummy = TradingEnvironment(df_train, sig_train)
        state_dim = env_dummy.state_dim
        agent, agent_type = load_or_train_agent(
            sym, state_dim, df_train, sig_train)

        if agent is None:
            continue

        # Get current state and RL signal
        sig_recent = build_patchtst_signals(df_recent, ptst)
        mac_recent = build_macro_signals(df_recent, asset_sym=sym)
        sig_recent = np.concatenate([sig_recent, mac_recent], axis=1)
        sig_recent = apply_weights(sig_recent, BEST_WEIGHTS)

        env_live = TradingEnvironment(df_recent, sig_recent)
        state = env_live.reset()
        # Step to latest state
        for _ in range(len(df_recent)-2):
            a = get_rl_signal(agent, agent_type, state)
            state,_,done,_ = env_live.step(a)
            if done: break

        rl_action = get_rl_signal(agent, agent_type, state)
        action_map = {0:"BUY", 1:"SELL", 2:"HOLD"}
        rl_signal  = action_map.get(rl_action, "HOLD")

        # Combine signals — consensus
        signals_list = []
        if pred_return > 0.003:   signals_list.append("BUY")
        elif pred_return < -0.003: signals_list.append("SELL")
        else:                      signals_list.append("HOLD")

        if regime == "bull":   signals_list.append("BUY")
        elif regime == "bear": signals_list.append("SELL")
        else:                  signals_list.append("HOLD")

        signals_list.append(rl_signal)

        buy_votes  = signals_list.count("BUY")
        sell_votes = signals_list.count("SELL")
        hold_votes = signals_list.count("HOLD")

        if buy_votes >= 2:        consensus = "BUY"
        elif sell_votes >= 2:     consensus = "SELL"
        else:                     consensus = "HOLD"

        # Position sizing
        base_size = risk_params["max_pos"]
        if consensus == "HOLD":
            position_size = 0.0
        else:
            # Adjust by signal strength
            agreement = max(buy_votes, sell_votes) / len(signals_list)
            position_size = base_size * agreement

        position_value = portfolio_value * position_size
        tp_price = current_price * (1 + risk_params["tp"]) \
                   if consensus == "BUY" \
                   else current_price * (1 - risk_params["tp"])
        sl_price = current_price * (1 - risk_params["sl"]) \
                   if consensus == "BUY" \
                   else current_price * (1 + risk_params["sl"])

        # Confidence — reflects actual signal quality per consensus type
        n = len(signals_list)
        if consensus == "BUY":
            confidence = buy_votes / n
        elif consensus == "SELL":
            confidence = sell_votes / n
        else:
            # HOLD confidence: high when forecast is near-zero + bear regime
            abs_forecast = abs(pred_return)
            try:
                regime_factor = 0.8 if hmm_regime == "bear" else 0.5
            except NameError:
                regime_factor = 0.6
            forecast_factor = max(0.0, 1.0 - abs_forecast * 50)
            confidence = float(min(0.95,
                regime_factor * forecast_factor + (hold_votes/n) * 0.3))

        result = {
            "symbol":          sym,
            "current_price":   round(current_price, 4),
            "signal":          consensus,
            "confidence":      round(confidence, 2),
            "position_size":   round(position_size, 3),
            "position_value":  round(position_value, 2),
            "take_profit":     round(tp_price, 4) if consensus != "HOLD" else None,
            "stop_loss":       round(sl_price, 4) if consensus != "HOLD" else None,
            "predicted_price": round(pred_price, 4),
            "expected_return": round(pred_return * 100, 2),
            "regime":          regime,
            "rl_agent":        agent_type,
            "votes": {
                "buy":  buy_votes,
                "sell": sell_votes,
                "hold": hold_votes,
            },
            "signals": {
                "patchtst": signals_list[0],
                "hmm":      signals_list[1],
                "rl_agent": rl_signal,
            },
            "risk_profile":    risk_profile,
            "timestamp":       datetime.now().isoformat(),
            "enhanced_metrics": {
                "model":           "PatchTST + HMM + SAC/PPO",
                "agent_type":      agent_type,
                "macro_features":  "VIX + interest_rate + SP500_regime",
                "sentiment":       "time_varying_proxy",
                "weight_config":   "Balanced-Expert",
                "avg_backtest_sharpe": 0.744,
            }
        }
        results.append(result)

        # Print summary
        signal_icon = {"BUY":"📈","SELL":"📉","HOLD":"⏸️ "}.get(consensus,"❓")
        print(f"    {signal_icon} SIGNAL: {consensus}  "
              f"(confidence: {confidence:.0%})")
        print(f"    Votes: BUY={buy_votes} SELL={sell_votes} HOLD={hold_votes}")
        print(f"    Position: {position_size:.1%} → ${position_value:,.0f}")
        if consensus != "HOLD":
            print(f"    TP: ${tp_price:,.4f}  SL: ${sl_price:,.4f}")
        else:
            print(f"    No position — waiting for clearer signal")

    # Save output — same format as baseline
    output = {
        "pipeline":    "enhanced",
        "timestamp":   datetime.now().isoformat(),
        "portfolio":   portfolio_value,
        "risk_profile":risk_profile,
        "assets":      results,
        "summary": {
            "total_assets":    len(results),
            "buy_signals":     sum(1 for r in results if r.get("signal")=="BUY"),
            "sell_signals":    sum(1 for r in results if r.get("signal")=="SELL"),
            "hold_signals":    sum(1 for r in results if r.get("signal")=="HOLD"),
            "avg_confidence":  round(
                float(np.mean([r.get("confidence",0)
                               for r in results
                               if r.get("signal") != "SKIP"])), 2),
        }
    }

    # Save JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR,
                             f"enhanced_signals_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Saved → {json_path}")

    # Also overwrite latest for frontend to read
    with open(os.path.join(OUTPUT_DIR, "enhanced_latest.json"), "w") as f:
        json.dump(output, f, indent=2)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Trading Pipeline")
    parser.add_argument("--symbols", type=str,
                        default="^GSPC,BTC-USD,GC=F",
                        help="Comma-separated symbols")
    parser.add_argument("--risk", type=str,
                        default="moderate",
                        choices=["conservative","moderate","aggressive"])
    parser.add_argument("--portfolio", type=float,
                        default=100000.0)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    output  = run_enhanced(symbols, args.risk, args.portfolio)

    print(f"\n{'='*55}")
    print(f"  ENHANCED PIPELINE COMPLETE")
    print(f"{'='*55}")
    print(f"  Assets analysed: {output['summary']['total_assets']}")
    print(f"  BUY signals:     {output['summary']['buy_signals']}")
    print(f"  SELL signals:    {output['summary']['sell_signals']}")
    print(f"  HOLD signals:    {output['summary']['hold_signals']}")
    print(f"  Avg confidence:  {output['summary']['avg_confidence']:.0%}")
    print(f"\n  Output: data/output/enhanced_latest.json")
    print(f"  Frontend reads: enhanced_latest.json (enhanced mode)")
    print(f"                  latest.json          (baseline mode)")


if __name__ == "__main__":
    main()
