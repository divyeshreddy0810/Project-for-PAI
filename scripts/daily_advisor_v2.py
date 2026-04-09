#!/usr/bin/env python3
"""
Daily Advisor V2 — Full MRAT-RL System
=======================================
Upgrades from V1:
  - 48 assets (stocks, crypto, forex, commodities)
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
from src.lab.forecaster                  import LabForecaster
from src.rl.trading_env                  import TradingEnvironment
from src.rl.sac_agent                    import SACAgent
from src.rl.ppo_agent                    import PPOAgent
from src.rl.td3_agent                    import TD3Agent
from src.forecast.lgbm_forecast          import LightGBMForecaster
from src.rl.macro_sentiment_features     import build_macro_signals, load_sentiment_features
from src.rl.integrated_pipeline          import build_patchtst_signals
from src.utils.currency                  import get_all_rates, parse_amount_input

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "data/models"
OUTPUT_DIR = "data/output"

# ── Full 48-asset universe ─────────────────────────────────────
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
    # GPU / Neocloud supply chain — TD3 (better for volatile news-driven equities)
    ("CRWV",     "CoreWeave",      "equity",    "TD3"),
    ("NBIS",     "Nebius",         "equity",    "TD3"),
    ("IREN",     "IREN",           "equity",    "TD3"),
    ("MU",       "Micron",         "equity",    "TD3"),
    ("SMCI",     "Super Micro",    "equity",    "TD3"),
    ("ARM",      "Arm Holdings",   "equity",    "TD3"),
    ("TSM",      "TSMC",           "equity",    "TD3"),
    ("VRT",      "Vertiv",         "equity",    "TD3"),
    ("MRVL",     "Marvell",        "equity",    "TD3"),
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
        "min_confidence":1.00, "tp_mult":0.5},   # 4/4 unanimous only
    2: {"name":"Moderate",     "max_pos":0.10, "sl":0.08, "tp":0.15,
        "min_confidence":0.75, "tp_mult":0.5},   # 3/4 models agree
    3: {"name":"Aggressive",   "max_pos":0.20, "sl":0.10, "tp":0.25,
        "min_confidence":0.50, "tp_mult":0.5},   # 2/4 signals included
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

def kelly_position_size(confidence, max_pos, sym="default",
                        asset_vol=0.02, kelly_frac=0.25):
    """
    Blend fractional-Kelly sizing with conviction-based sizing.
    Uses per-asset win rates from validation results.
    Falls back to dynamic_position_size if Kelly is zero.
    kelly_frac: 0.25 = Quarter-Kelly, 0.5 = Half-Kelly, 1.0 = Full-Kelly
    """
    import numpy as np
    WIN_RATES = {
        "AMZN":0.92,"XRP-USD":0.80,"SOL-USD":0.89,"NVDA":0.84,
        "META":0.84,"GOOGL":0.84,"GC=F":0.77,"MSFT":0.76,
        "^GSPC":0.76,"^DJI":0.76,"GBPUSD=X":0.61,
    }
    win_rate = WIN_RATES.get(sym, 0.65)
    avg_win  = 0.03
    avg_loss = 0.02
    r        = avg_win / max(avg_loss, 0.001)
    kelly    = win_rate - (1 - win_rate) / r
    if kelly <= 0:
        return dynamic_position_size(confidence, max_pos)
    frac_kelly  = kelly * kelly_frac
    TARGET_VOL  = 0.15
    vol_adj     = min(1.0, TARGET_VOL / max(asset_vol * (252**0.5), 0.01))
    kelly_sized = min(frac_kelly * vol_adj, max_pos)
    conviction  = dynamic_position_size(confidence, max_pos)
    return round((kelly_sized + conviction) / 2, 3)


# Semiconductor symbols that use the dedicated SOX-trained HMM instead of
# the generic equity HMM — their regime is driven by capex cycles not S&P.
_SEMICONDUCTOR_SYMS = {
    "CRWV","NBIS","IREN","MU","SMCI","ARM","TSM","VRT","MRVL","NVDA",
    "AVGO","INTC","AMD","QCOM","TXN","AMAT","LRCX","KLAC","ASML",
}

# Sector-level concentration cap: prevent the AI infrastructure cluster
# from dominating the portfolio when all 9 GPU stocks signal BUY at once.
_GPU_STOCKS      = {"CRWV","NBIS","IREN","MU","SMCI","ARM","TSM","VRT","MRVL"}
SECTOR_CAP_SEMI  = 0.25   # max 25% portfolio in semiconductor/AI infra


def get_hmm(asset_class, sym=None):
    """
    Return asset-class-specific HMM.
    Semiconductors use hmm_semiconductor.pkl (trained on SOX) if it exists,
    otherwise fall back to hmm_equity.pkl, then the EUR/USD pretrained HMM.
    """
    import joblib
    # Semiconductor override — regime driven by capex cycle, not S&P SMA
    if sym and sym in _SEMICONDUCTOR_SYMS:
        cache_key = "semiconductor"
        if cache_key not in _HMM_CACHE:
            path = f"{MODELS_DIR}/hmm_semiconductor.pkl"
            if os.path.exists(path):
                _HMM_CACHE[cache_key] = joblib.load(path)
            elif os.path.exists(f"{MODELS_DIR}/hmm_equity.pkl"):
                _HMM_CACHE[cache_key] = joblib.load(f"{MODELS_DIR}/hmm_equity.pkl")
            else:
                from src.regime.pretrained_hmm import get_pretrained_hmm
                _HMM_CACHE[cache_key] = get_pretrained_hmm()
        return _HMM_CACHE[cache_key]

    if asset_class not in _HMM_CACHE:
        path = f"{MODELS_DIR}/hmm_{asset_class}.pkl"
        if os.path.exists(path):
            _HMM_CACHE[asset_class] = joblib.load(path)
        else:
            from src.regime.pretrained_hmm import get_pretrained_hmm
            _HMM_CACHE[asset_class] = get_pretrained_hmm()
    return _HMM_CACHE[asset_class]

def predict_with_proba(hmm_model, df_hmm, bull_count=34):
    """
    Use posterior probabilities with correct state labeling.
    Lower threshold in broad bull markets to reduce HOLD bias.
    """
    import numpy as np
    try:
        df_h = df_hmm.copy()
        delta = df_h["Close"].diff()
        df_h["_ret"] = delta / (df_h["Close"].shift(1) + 1e-9)
        df_h["_vol"] = df_h["_ret"].rolling(5).std()
        features = df_h[["_ret","_vol","RSI","sentiment_mean"]].dropna().values
        if len(features) < 10:
            return hmm_model.predict(df_hmm)
        proba     = hmm_model.model.predict_proba(features)
        last      = proba[-1]
        means     = hmm_model.model.means_[:, 0]
        sorted_st = np.argsort(means)
        bear_idx  = sorted_st[0]
        bull_idx  = sorted_st[-1]
        bull_prob = last[bull_idx]
        bear_prob = last[bear_idx]
        thresh = 0.40 if bull_count > 30 else 0.50
        if bull_prob > thresh:
            return "bull"
        elif bear_prob > thresh:
            return "bear"
        else:
            return "sideways"
    except Exception:
        try:
            return hmm_model.predict(df_hmm)
        except Exception:
            return "sideways"


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

    # Load forecaster — PatchTST or promoted LabForecaster (LSTM/CNN/MLP)
    try:
        ckpt_meta = torch.load(ptst_path, map_location="cpu", weights_only=False)
        if "algo" in ckpt_meta and ckpt_meta["algo"] in ("LSTM", "CNN", "MLP"):
            ptst = LabForecaster(algo=ckpt_meta["algo"])
        else:
            ptst = PatchTSTForecaster()
        ptst.load(ptst_path)
    except Exception:
        return None

    pred_return = ptst.predict_return(df_recent)

    # LGBM ensemble blend — average PatchTST and LightGBM predictions 50/50
    lgbm_path = f"{MODELS_DIR}/lgbm_{key}.pkl"
    try:
        if os.path.exists(lgbm_path):
            lgbm     = joblib.load(lgbm_path)
            feat_df  = LightGBMForecaster.engineer_features(df_recent)
            feat_arr = feat_df.fillna(0).values
            lgbm_prices  = lgbm.predict(feat_arr)   # returns predicted prices per horizon
            lgbm_price_5 = float(lgbm_prices[0])    # first horizon (5-day)
            lgbm_return  = (lgbm_price_5 - current_price) / current_price
            pred_return  = 0.5 * pred_return + 0.5 * lgbm_return
    except Exception:
        pass  # fallback to PatchTST-only if LGBM fails

    pred_price  = current_price * (1 + pred_return)

    # HMM regime
    try:
        hmm_model = get_hmm(asset_class, sym=sym)
        df_hmm    = df_recent.tail(60).copy()
        delta     = df_hmm["Close"].diff()
        gain      = delta.clip(lower=0).rolling(14).mean()
        loss      = (-delta.clip(upper=0)).rolling(14).mean()
        df_hmm["RSI"]            = 100-(100/(1+gain/(loss+1e-9)))
        df_hmm["sentiment_mean"] = load_sentiment_features(asset_sym=sym)["mean"]
        df_hmm = df_hmm.dropna()
        regime = predict_with_proba(hmm_model, df_hmm)
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
        elif agent_type == "TD3":
            agent = TD3Agent(state_dim=state_dim, device=DEVICE)
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

    # Position sizing — compute realized vol from recent data so Kelly
    # vol-scaling actually works (previously always used 0.02 default,
    # treating SMCI and EURUSD identically)
    base_size = risk["max_pos"]
    if consensus == "HOLD":
        position_size = 0.0
    else:
        realized_vol = float(df_recent["Close"].pct_change().std())
        realized_vol = max(realized_vol, 0.001)  # floor to avoid div-by-zero
        position_size = kelly_position_size(
            confidence, base_size, sym=sym,
            asset_vol=realized_vol,
            kelly_frac=risk.get("kelly_frac", 0.25)
        )

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
    # Quick exit mode — user preference: take profit in 1-2 hours
    # tp_mult=0.5 means half ATR (tight, intraday target)
    # tp_mult=1.5 means full ATR (original, full-day target)
    tp_mult = risk.get("tp_mult", 1.5)
    # TP/SL: daily=ATR-based, swing=forecast-based
    atr_pct = atr / current_price
    pr      = abs(pred_return / 100)
    _mult   = {"forex":2.0,"crypto":3.0,
               "commodity":2.5,"equity":2.5}
    _m      = _mult.get(asset_class, 2.5)
    swing_move = pr * _m if pr > 0.001 else atr_pct * 1.2
    if consensus == "BUY":
        tp_daily  = current_price + tp_mult * atr
        sl_daily  = current_price - 1.0 * atr
        tp_swing  = current_price * (1 + swing_move)
        sl_swing  = current_price - 1.5 * atr
    elif consensus == "SELL":
        tp_daily  = current_price - tp_mult * atr
        sl_daily  = current_price + 1.0 * atr
        tp_swing  = current_price * (1 - swing_move)
        sl_swing  = current_price + 1.5 * atr
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
    exit_mode = risk.get("exit_mode", "normal")
    if exit_mode == "trail" and sig != "HOLD":
        trail_pct  = 2.0  # trail SL 2% below peak
        trigger    = round(price * 1.005, 4)  # +0.5% breakeven trigger
        print(f"  │  ── TRAILING STOP INSTRUCTIONS ──")
        print(f"  │  Step 1: Enter at {currency}{price:.4f} (Market order)")
        print(f"  │  Step 2: Set initial SL at {currency}{result['sl_daily']:.4f} (Stop order)")
        print(f"  │  Step 3: When price hits {currency}{trigger:.4f} (+0.5%) →")
        print(f"  │           Move SL to {currency}{price:.4f} (breakeven)")
        print(f"  │  Step 4: Trail SL {trail_pct:.0f}% below highest price reached")
        print(f"  │  Step 5: If price returns to entry → close manually")
    print(f"  └─ PatchTST:{result['signals']['patchtst']}  "
          f"HMM:{result['signals']['hmm']}  "
          f"RL:{result['signals']['rl_agent']}  "
          f"ADX:{result['signals'].get('adx','N/A')}")

# ── Main ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  MRAT-RL DAILY ADVISOR V2")
    print("  Full 48-asset universe · PatchTST + HMM + RL")
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

    print("\n  Kelly fraction:")
    print("    1 = Quarter-Kelly (0.25) — conservative, lower drawdown")
    print("    2 = Half-Kelly    (0.50) — balanced")
    print("    3 = Full-Kelly    (1.00) — aggressive, higher variance")
    try:
        kf_choice = int(input("  Choose [1/2/3]: ").strip())
    except Exception:
        kf_choice = 1
    kelly_fracs = {1: 0.25, 2: 0.50, 3: 1.00}
    kelly_labels = {1: "Quarter-Kelly (0.25)", 2: "Half-Kelly (0.50)", 3: "Full-Kelly (1.00)"}
    risk["kelly_frac"] = kelly_fracs.get(kf_choice, 0.25)
    print(f"  ✅ {kelly_labels.get(kf_choice, 'Quarter-Kelly (0.25)')}")

    print("\n  Exit mode:")
    print("    1 = Quick Fixed    — TP at 0.5x ATR, in and out in 1-2 hours")
    print("    2 = Trailing Stop  — breakeven at +0.5%, trail SL behind price")
    print("    3 = Normal Fixed   — TP at 1.5x ATR, hold full day")
    try:
        exit_choice = int(input("  Choose [1/2/3]: ").strip())
        if exit_choice not in [1,2,3]: exit_choice = 2
    except Exception:
        exit_choice = 2
    if exit_choice == 1:
        risk["tp_mult"]    = 0.5
        risk["exit_mode"]  = "quick"
        print("  ✅ Quick Fixed — tight TP, fast exit")
    elif exit_choice == 2:
        risk["tp_mult"]    = 0.5   # initial SL still tight
        risk["exit_mode"]  = "trail"
        print("  ✅ Trailing Stop — breakeven at +0.5%, trail from there")
    else:
        risk["tp_mult"]    = 1.5
        risk["exit_mode"]  = "normal"
        print("  ✅ Normal Fixed — full day TP")
    input("\n  Press Enter to scan all 40 assets...\n")

    # ── Scan all assets ────────────────────────────────────────
    section("SCANNING ALL ASSETS")
    results = []
    skipped = []

    for i, (sym, name, asset_class, agent_type) in enumerate(UNIVERSE):
        print(f"  [{i+1:2d}/48] {sym:<12} {name:<20}", end="\r")
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

    # ── Sector concentration cap ───────────────────────────────
    # Prevent all 9 GPU stocks from signalling BUY simultaneously and
    # consuming >25% of portfolio in one sector cluster.
    gpu_active = [r for r in results
                  if r["symbol"] in _GPU_STOCKS and r["signal"] != "HOLD"]
    if gpu_active:
        total_gpu = sum(r["position_size"] for r in gpu_active)
        if total_gpu > SECTOR_CAP_SEMI:
            scale = SECTOR_CAP_SEMI / total_gpu
            for r in gpu_active:
                r["position_size"] = round(r["position_size"] * scale, 3)
            print(f"\n  ⚠️  GPU/Semi sector cap: {total_gpu:.1%} → "
                  f"{SECTOR_CAP_SEMI:.0%}  "
                  f"(scaled {len(gpu_active)} positions ×{scale:.2f})")

    # ── Rank signals ───────────────────────────────────────────
    active = [r for r in results
              if r["signal"] != "HOLD"
              and r["confidence"] >= risk["min_confidence"]
              and not (r["signal"] == "BUY"
                       and r.get("pred_return", 0) < 0.05)]
    active.sort(key=lambda x: x["strength"], reverse=True)
    holds     = [r for r in results if r["signal"] == "HOLD"]
    all_sorted= active + holds

    # Daily top 3
    daily_top  = active[:3] if active else all_sorted[:3]
    daily_syms = {r["symbol"] for r in daily_top}

    # Swing top 3 — no overlap with daily
    swing_seen = set()
    swing_top  = []
    for _r in (active + holds):
        if _r["symbol"] not in daily_syms and \
           _r["symbol"] not in swing_seen:
            swing_seen.add(_r["symbol"])
            swing_top.append(_r)
        if len(swing_top) >= 3: break

    # ── Print top 3 daily trades ───────────────────────────────
    section("TOP 3 DAILY TRADES  (ATR-based TP/SL)")
    daily_top = active[:3] if active else all_sorted[:3]
    for i, r in enumerate(daily_top, 1):
        print_trade_card(r, i, portfolio_eur, risk, currency, rates)

    # ── Print top 3 swing trades ───────────────────────────────
    section("TOP 3 SWING TRADES  (5-day horizon)")
    # swing_top already built above
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
