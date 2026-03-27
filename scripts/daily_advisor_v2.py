#!/usr/bin/env python3
"""
Daily Advisor V2 — Full MRAT-RL System
=======================================
Upgrades from V1:
  - 40 assets (stocks, crypto, forex, commodities)
  - PatchTST Transformer forecasting (saved models)
  - SAC/PPO RL agents per asset (saved models)
  - Per-asset-class HMM regime detection
  - Per-asset macro benchmarks (no cross-asset contamination)
  - Real FinBERT sentiment per asset
  - Consensus vote: PatchTST + HMM + RL agent
  - Top 3 daily + Top 3 swing trades
  - MT4 trade cards in EUR and NGN

Usage:
    python3 scripts/daily_advisor_v2.py
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
from src.utils.currency                  import get_all_rates, parse_amount_input

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "data/models"
OUTPUT_DIR = "data/output"

# ── Full 40-asset universe ─────────────────────────────────────
UNIVERSE = [
    # Indices
    ("^GSPC",    "S&P 500",        "equity",    "SAC"),
    ("^IXIC",    "NASDAQ",         "equity",    "SAC"),
    ("^DJI",     "Dow Jones",      "equity",    "SAC"),
    # US Stocks
    ("AAPL",     "Apple",          "equity",    "SAC"),
    ("MSFT",     "Microsoft",      "equity",    "SAC"),
    ("NVDA",     "NVIDIA",         "equity",    "SAC"),
    ("TSLA",     "Tesla",          "equity",    "SAC"),
    ("AMZN",     "Amazon",         "equity",    "SAC"),
    ("META",     "Meta",           "equity",    "SAC"),
    ("GOOGL",    "Alphabet",       "equity",    "SAC"),
    ("JPM",      "JPMorgan",       "equity",    "SAC"),
    # Crypto
    ("BTC-USD",  "Bitcoin",        "crypto",    "PPO"),
    ("ETH-USD",  "Ethereum",       "crypto",    "PPO"),
    ("SOL-USD",  "Solana",         "crypto",    "PPO"),
    ("BNB-USD",  "BNB",            "crypto",    "PPO"),
    ("XRP-USD",  "XRP",            "crypto",    "PPO"),
    # Forex Major
    ("EURUSD=X", "EUR/USD",        "forex",     "SAC"),
    ("GBPUSD=X", "GBP/USD",        "forex",     "SAC"),
    ("USDJPY=X", "USD/JPY",        "forex",     "SAC"),
    ("USDCHF=X", "USD/CHF",        "forex",     "SAC"),
    ("AUDUSD=X", "AUD/USD",        "forex",     "SAC"),
    ("NZDUSD=X", "NZD/USD",        "forex",     "SAC"),
    ("USDCAD=X", "USD/CAD",        "forex",     "SAC"),
    # Forex Cross
    ("EURGBP=X", "EUR/GBP",        "forex",     "SAC"),
    ("EURJPY=X", "EUR/JPY",        "forex",     "SAC"),
    ("GBPJPY=X", "GBP/JPY",        "forex",     "SAC"),
    ("AUDNZD=X", "AUD/NZD",        "forex",     "SAC"),
    # African/Emerging
    ("USDNGN=X", "USD/NGN",        "forex",     "SAC"),
    ("EURNGN=X", "EUR/NGN",        "forex",     "SAC"),
    ("USDZAR=X", "USD/ZAR",        "forex",     "SAC"),
    ("USDKES=X", "USD/KES",        "forex",     "SAC"),
    ("USDGHS=X", "USD/GHS",        "forex",     "SAC"),
    # Commodities
    ("GC=F",     "Gold",           "commodity", "SAC"),
    ("SI=F",     "Silver",         "commodity", "SAC"),
    ("CL=F",     "Crude Oil",      "commodity", "SAC"),
    ("NG=F",     "Natural Gas",    "commodity", "SAC"),
    ("HG=F",     "Copper",         "commodity", "SAC"),
    ("ZW=F",     "Wheat",          "commodity", "SAC"),
    ("ZC=F",     "Corn",           "commodity", "SAC"),
]

# ── Risk profiles ──────────────────────────────────────────────
RISK_PROFILES = {
    1: {"name":"Conservative", "max_pos":0.05, "sl":0.05, "tp":0.10,
        "min_confidence":1.00},   # 4/4 unanimous only
    2: {"name":"Moderate",     "max_pos":0.10, "sl":0.08, "tp":0.15,
        "min_confidence":0.75},   # 3/4 models agree
    3: {"name":"Aggressive",   "max_pos":0.20, "sl":0.10, "tp":0.25,
        "min_confidence":0.50},   # 2/4 signals included
}

BEST_WEIGHTS = [1.2, 1.2, 1.0, 0.6]

def dynamic_position_size(confidence, risk_profile_max=0.10):
    """
    Dynamic position sizing based on consensus confidence.
    Confidence is always 0.33/0.67/1.00 (1/2/3 models agree).
    
    conservative max=0.05:  0.33→1.7%  0.67→3.3%  1.00→5.0%
    moderate     max=0.10:  0.33→3.3%  0.67→6.7%  1.00→10.0%  (old flat)
    aggressive   max=0.20:  0.33→5.0%  0.67→15.0% 1.00→20.0%
    
    Scaling: non-linear — rewards high conviction signals
    """
    if confidence <= 0.34:      # 1/3 votes — weak signal
        scale = 0.33
    elif confidence <= 0.68:    # 2/3 votes — strong signal
        scale = 0.67
    else:                       # 3/3 votes — maximum conviction
        scale = 1.00
    return risk_profile_max * scale


# ── HMM cache ─────────────────────────────────────────────────
_HMM_CACHE = {}

def get_hmm(asset_class):
    import joblib
    if asset_class not in _HMM_CACHE:
        path = f"{MODELS_DIR}/hmm_{asset_class}.pkl"
        if os.path.exists(path):
            _HMM_CACHE[asset_class] = joblib.load(path)
        else:
            from src.regime.pretrained_hmm import get_pretrained_hmm
            _HMM_CACHE[asset_class] = get_pretrained_hmm()
    return _HMM_CACHE[asset_class]

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

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ── Analyse one asset ──────────────────────────────────────────
def analyse_asset(sym, name, asset_class, agent_type, risk):
    key      = safe_key(sym)
    ptst_path = f"{MODELS_DIR}/patchtst_{key}.pt"
    rl_path   = f"{MODELS_DIR}/{agent_type.lower()}_{key}.pt"

    # Check models exist
    if not os.path.exists(ptst_path) or not os.path.exists(rl_path):
        return None  # Not trained yet — skip

    # Fetch data
    try:
        df = yf.download(sym, start="2004-01-01",
                         end=datetime.now().strftime("%Y-%m-%d"),
                         progress=False)
        if hasattr(df.columns, 'droplevel'):
            df.columns = df.columns.droplevel(1)
        df = df.dropna()
        if len(df) < 100:
            return None
    except Exception:
        return None

    current_price = float(df["Close"].iloc[-1])
    df_recent     = df.tail(90).reset_index(drop=True)

    # Load PatchTST
    ptst = PatchTSTForecaster()
    try:
        ptst.load(ptst_path)
    except Exception:
        return None

    pred_return = ptst.predict_return(df_recent)
    pred_price  = current_price * (1 + pred_return)

    # HMM regime
    try:
        hmm_model = get_hmm(asset_class)
        df_hmm    = df_recent.tail(60).copy()
        delta     = df_hmm["Close"].diff()
        gain      = delta.clip(lower=0).rolling(14).mean()
        loss      = (-delta.clip(upper=0)).rolling(14).mean()
        df_hmm["RSI"]            = 100-(100/(1+gain/(loss+1e-9)))
        df_hmm["sentiment_mean"] = 0.05
        df_hmm = df_hmm.dropna()
        regime = hmm_model.predict(df_hmm)
    except Exception:
        regime = "sideways"

    # Load RL agent
    try:
        sig = build_patchtst_signals(df_recent, ptst)
        mac = build_macro_signals(df_recent, asset_sym=sym)
        sig = np.concatenate([sig, mac], axis=1)
        sig = apply_weights(sig, BEST_WEIGHTS)

        env   = TradingEnvironment(df_recent, sig)
        state_dim = env.state_dim

        if agent_type == "SAC":
            agent = SACAgent(state_dim=state_dim, device=DEVICE)
            agent.net.load_state_dict(
                torch.load(rl_path, map_location=DEVICE))
            state = env.reset()
            for _ in range(len(df_recent)-2):
                a = agent.select_action(state, deterministic=True)
                state,_,done,_ = env.step(a)
                if done: break
            rl_action = agent.select_action(state, deterministic=True)
        else:
            agent = PPOAgent(state_dim=state_dim, device=DEVICE,
                             n_steps=256)
            agent.policy.load_state_dict(
                torch.load(rl_path, map_location=DEVICE))
            state = env.reset()
            for _ in range(len(df_recent)-2):
                a,_,_ = agent.select_action(state, deterministic=True)
                state,_,done,_ = env.step(a)
                if done: break
            rl_action,_,_ = agent.select_action(state,
                                                  deterministic=True)

        action_map = {0:"BUY", 1:"SELL", 2:"HOLD"}
        rl_signal  = action_map.get(rl_action, "HOLD")
    except Exception:
        rl_signal = "HOLD"

    # ADX trend strength as 4th voter
    try:
        high = df_recent["High"].values
        low  = df_recent["Low"].values
        close= df_recent["Close"].values
        # True range
        tr   = np.maximum(high[1:]-low[1:],
               np.maximum(abs(high[1:]-close[:-1]),
                          abs(low[1:]-close[:-1])))
        atr14= float(np.mean(tr[-14:]))
        # Directional movement
        up_move  = high[1:] - high[:-1]
        dn_move  = low[:-1] - low[1:]
        pos_dm   = np.where((up_move > dn_move) & (up_move > 0), up_move, 0)
        neg_dm   = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0)
        di_plus  = 100 * np.mean(pos_dm[-14:]) / (atr14 + 1e-9)
        di_minus = 100 * np.mean(neg_dm[-14:]) / (atr14 + 1e-9)
        adx_signal = "BUY" if di_plus > di_minus * 1.1 else                      "SELL" if di_minus > di_plus * 1.1 else "HOLD"
    except Exception:
        adx_signal = "HOLD"

    # Consensus vote — 4 voters now
    votes = []
    if pred_return >  0.003: votes.append("BUY")
    elif pred_return < -0.003: votes.append("SELL")
    else:                    votes.append("HOLD")

    if regime == "bull":     votes.append("BUY")
    elif regime == "bear":   votes.append("SELL")
    else:                    votes.append("HOLD")

    votes.append(rl_signal)
    votes.append(adx_signal)  # 4th voter

    buy_v  = votes.count("BUY")
    sell_v = votes.count("SELL")
    hold_v = votes.count("HOLD")

    if buy_v >= 3:      consensus = "BUY"   # 3/4 agree
    elif sell_v >= 3:   consensus = "SELL"  # 3/4 agree
    elif buy_v >= 2 and sell_v == 0: consensus = "BUY"   # 2/4, no opposition
    elif sell_v >= 2 and buy_v == 0: consensus = "SELL"  # 2/4, no opposition
    else:               consensus = "HOLD"

    # Confidence
    n = len(votes)
    if consensus == "BUY":
        confidence = buy_v / n
    elif consensus == "SELL":
        confidence = sell_v / n
    else:
        abs_f = abs(pred_return)
        rf    = 0.8 if regime == "bear" else 0.5
        ff    = max(0.0, 1.0 - abs_f * 50)
        confidence = float(min(0.95, rf*ff + (hold_v/n)*0.3))

    # Signal strength score for ranking
    strength = confidence
    if consensus != "HOLD":
        strength += abs(pred_return) * 10
        if regime in ("bull","bear"):
            strength += 0.1

    # Position sizing
    base_size = risk["max_pos"]
    if consensus == "HOLD":
        position_size = 0.0
    else:
        position_size = dynamic_position_size(confidence, base_size)

    # ATR for daily TP/SL
    try:
        high  = df_recent["High"].values
        low   = df_recent["Low"].values
        close = df_recent["Close"].values
        tr    = np.maximum(high[1:]-low[1:],
                np.maximum(abs(high[1:]-close[:-1]),
                           abs(low[1:]-close[:-1])))
        atr   = float(np.mean(tr[-14:]))
    except Exception:
        atr = current_price * 0.015

    # TP/SL levels
    if consensus == "BUY":
        tp_daily  = current_price + 1.5 * atr
        sl_daily  = current_price - 1.0 * atr
        tp_swing  = current_price * (1 + risk["tp"])
        sl_swing  = current_price * (1 - risk["sl"])
    elif consensus == "SELL":
        tp_daily  = current_price - 1.5 * atr
        sl_daily  = current_price + 1.0 * atr
        tp_swing  = current_price * (1 - risk["tp"])
        sl_swing  = current_price * (1 + risk["sl"])
    else:
        tp_daily = tp_swing = sl_daily = sl_swing = None

    return {
        "symbol":       sym,
        "name":         name,
        "asset_class":  asset_class,
        "agent":        agent_type,
        "current_price":current_price,
        "signal":       consensus,
        "confidence":   round(confidence, 2),
        "strength":     round(strength, 4),
        "regime":       regime,
        "pred_return":  round(pred_return * 100, 3),
        "pred_price":   round(pred_price, 4),
        "position_size":round(position_size, 3),
        "atr":          round(atr, 4),
        "tp_daily":     round(tp_daily, 4) if tp_daily else None,
        "sl_daily":     round(sl_daily, 4) if sl_daily else None,
        "tp_swing":     round(tp_swing, 4) if tp_swing else None,
        "sl_swing":     round(sl_swing, 4) if sl_swing else None,
        "votes": {
            "buy":  buy_v,
            "sell": sell_v,
            "hold": hold_v,
        },
        "signals": {
            "patchtst": votes[0],
            "hmm":      votes[1],
            "rl_agent": rl_signal,
            "adx":      adx_signal,
        },
        "timestamp": datetime.now().isoformat(),
    }

# ── Print MT4 trade card ───────────────────────────────────────
def print_trade_card(result, rank, portfolio, risk, currency, fx):
    sym    = result["symbol"]
    name   = result["name"]
    sig    = result["signal"]
    conf   = result["confidence"]
    regime = result["regime"]
    price  = result["current_price"]
    pred_r = result["pred_return"]
    pos    = result["position_size"]

    pos_val_eur = portfolio * pos
    pos_val_ngn = pos_val_eur * fx.get("EUR_NGN", 1700)

    sig_arrow = "▲" if sig == "BUY" else "▼" if sig == "SELL" else "⏸"
    regime_icon = {"bull":"🟢","bear":"🔴","sideways":"🟡"}.get(regime,"⚪")
    if conf >= 1.00:   conviction = "★ UNANIMOUS"
    elif conf >= 0.75: conviction = "◆ HIGH"
    elif conf >= 0.50: conviction = "◇ MODERATE"
    else:              conviction = "○ WEAK"

    print(f"\n  ┌─ #{rank} {sym}  {name}")
    print(f"  │  {sig_arrow} {sig}  · {regime_icon} {regime.upper()} "
          f"· {conviction} · conf {conf:.0%} · {result['agent']}")
    print(f"  │  Entry      : {currency}{price:>12.4f}")
    if result["tp_daily"]:
        pct = (result['tp_daily']/price-1)*100
        print(f"  │  TP (daily) : {currency}{result['tp_daily']:>12.4f}"
              f"  ({pct:+.2f}%)")
        pct = (result['sl_daily']/price-1)*100
        print(f"  │  SL (daily) : {currency}{result['sl_daily']:>12.4f}"
              f"  ({pct:+.2f}%)")
        pct = (result['tp_swing']/price-1)*100
        print(f"  │  TP (swing) : {currency}{result['tp_swing']:>12.4f}"
              f"  ({pct:+.2f}%)")
        pct = (result['sl_swing']/price-1)*100
        print(f"  │  SL (swing) : {currency}{result['sl_swing']:>12.4f}"
              f"  ({pct:+.2f}%)")
    print(f"  │  Position   : {pos:.1%} → "
          f"{currency}{pos_val_eur:,.0f}  "
          f"(₦{pos_val_ngn:,.0f})")
    print(f"  │  5d forecast : {pred_r:+.2f}%  → "
          f"{currency}{result['pred_price']:.4f}")
    print(f"  │  Votes      : "
          f"BUY={result['votes']['buy']} "
          f"SELL={result['votes']['sell']} "
          f"HOLD={result['votes']['hold']}")
    print(f"  └─ PatchTST:{result['signals']['patchtst']}  "
          f"HMM:{result['signals']['hmm']}  "
          f"RL:{result['signals']['rl_agent']}  "
          f"ADX:{result['signals'].get('adx','N/A')}")

# ── Main ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  MRAT-RL DAILY ADVISOR V2")
    print("  Full 40-asset universe · PatchTST + HMM + RL")
    print(f"  {datetime.now().strftime('%A %d %B %Y  %H:%M')}")
    print("="*60)

    # ── User input ─────────────────────────────────────────────
    section("PORTFOLIO SETUP")
    raw = input("  Portfolio amount (e.g. ₦100000 or €500 or $300): ").strip()
    try:
        rates = get_all_rates()
        # Manual parse to handle ₦ NGN symbol
        raw_clean = raw.strip()
        if raw_clean.startswith("₦"):
            ngn = float(raw_clean[1:].replace(",",""))
            portfolio_eur = ngn / rates.get("EUR_NGN", 1700)
            currency = "€"
        elif raw_clean.startswith("$"):
            usd = float(raw_clean[1:].replace(",",""))
            portfolio_eur = usd / rates.get("EUR_USD", 1.08)
            currency = "€"
        elif raw_clean.startswith("€"):
            portfolio_eur = float(raw_clean[1:].replace(",",""))
            currency = "€"
        else:
            portfolio_eur, currency_sym, _ = parse_amount_input(
                raw_clean, rates)
            currency = "€"
    except Exception:
        portfolio_eur = 1000.0
        rates = {"EUR_NGN": 1700, "EUR_USD": 1.08}
        currency = "€"
        print("  ⚠️  Could not parse amount — using €1,000")

    portfolio_ngn = portfolio_eur * rates.get("EUR_NGN", 1700)
    print(f"  Portfolio: €{portfolio_eur:,.2f}  "
          f"(₦{portfolio_ngn:,.0f})")

    print("\n  Risk profile:")
    print("    1 = Conservative (5% per trade)")
    print("    2 = Moderate     (10% per trade)")
    print("    3 = Aggressive   (20% per trade)")
    try:
        risk_choice = int(input("  Choose [1/2/3]: ").strip())
        if risk_choice not in RISK_PROFILES:
            risk_choice = 2
    except Exception:
        risk_choice = 2
    risk = RISK_PROFILES[risk_choice]
    print(f"  ✅ {risk['name']} — max {risk['max_pos']:.0%} per trade")

    input("\n  Press Enter to scan all 40 assets...\n")

    # ── Scan all assets ────────────────────────────────────────
    section("SCANNING ALL ASSETS")
    results = []
    skipped = []

    for i, (sym, name, asset_class, agent_type) in enumerate(UNIVERSE):
        print(f"  [{i+1:2d}/39] {sym:<12} {name:<20}", end="\r")
        result = analyse_asset(sym, name, asset_class,
                               agent_type, risk)
        if result:
            results.append(result)
        else:
            skipped.append(sym)

    print(f"\n  ✅ Scanned {len(results)} assets  "
          f"({len(skipped)} skipped — models not ready)")
    if skipped:
        print(f"  ⚠️  Skipped: {skipped}")

    if not results:
        print("  ❌ No results — run train_all_models.py first")
        return

    # ── Rank signals ───────────────────────────────────────────
    active = [r for r in results if r["signal"] != "HOLD"
              and r["confidence"] >= risk["min_confidence"]
              and not (r["signal"] == "BUY" and r.get("pred_return", 0) < 0)]
    active.sort(key=lambda x: x["strength"], reverse=True)
    active = active[:3]  # MAX 3 SIGNALS

    holds  = [r for r in results if r["signal"] == "HOLD"]
    all_sorted = active + holds

    # ── Print top 3 daily trades ───────────────────────────────
    section("TOP 3 DAILY TRADES  (ATR-based TP/SL)")
    daily_top = active[:3] if active else all_sorted[:3]
    for i, r in enumerate(daily_top, 1):
        print_trade_card(r, i, portfolio_eur, risk, currency, rates)

    # ── Print top 3 swing trades ───────────────────────────────
    section("TOP 3 SWING TRADES  (5-day horizon)")
    swing_top = active[3:6] if len(active) >= 6 else \
                (active[:3] if active else all_sorted[:3])
    for i, r in enumerate(swing_top, 1):
        print_trade_card(r, i, portfolio_eur, risk, currency, rates)

    # ── Market summary ─────────────────────────────────────────
    section("MARKET SUMMARY")
    buy_count  = sum(1 for r in results if r["signal"]=="BUY")
    sell_count = sum(1 for r in results if r["signal"]=="SELL")
    hold_count = sum(1 for r in results if r["signal"]=="HOLD")
    avg_conf   = float(np.mean([r["confidence"] for r in results]))

    bull_count = sum(1 for r in results if r["regime"]=="bull")
    bear_count = sum(1 for r in results if r["regime"]=="bear")
    side_count = sum(1 for r in results if r["regime"]=="sideways")

    print(f"\n  Signals  : BUY={buy_count}  "
          f"SELL={sell_count}  HOLD={hold_count}")
    print(f"  Regimes  : BULL={bull_count}  "
          f"BEAR={bear_count}  SIDEWAYS={side_count}")
    print(f"  Avg conf : {avg_conf:.0%}")
    print(f"  Assets   : {len(results)} analysed  "
          f"{len(skipped)} skipped")

    # ── Asset breakdown by class ───────────────────────────────
    print(f"\n  By asset class:")
    for cls in ["equity","crypto","commodity","forex"]:
        cls_results = [r for r in results
                       if r["asset_class"] == cls]
        cls_active  = [r for r in cls_results
                       if r["signal"] != "HOLD"]
        print(f"    {cls:<12}: "
              f"{len(cls_results):2d} scanned  "
              f"{len(cls_active):2d} active signals")

    # ── Save output ────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "version":      "v2",
        "timestamp":    datetime.now().isoformat(),
        "portfolio_eur":round(portfolio_eur, 2),
        "portfolio_ngn":round(portfolio_ngn, 2),
        "risk_profile": risk["name"],
        "assets":       results,
        "summary": {
            "total":    len(results),
            "buy":      buy_count,
            "sell":     sell_count,
            "hold":     hold_count,
            "avg_conf": round(avg_conf, 2),
        },
        "top_daily": [r["symbol"] for r in daily_top],
        "top_swing": [r["symbol"] for r in swing_top],
    }
    path = os.path.join(OUTPUT_DIR, f"daily_advice_v2_{ts}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    # Also write latest for frontend
    with open(os.path.join(OUTPUT_DIR, "latest_v2.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  💾 Saved → {path}")
    print(f"\n{'='*60}")
    print(f"  RUN COMPLETE — Check back tomorrow")
    print(f"  Next run: python3 scripts/daily_advisor_v2.py")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
