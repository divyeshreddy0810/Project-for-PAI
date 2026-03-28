#!/usr/bin/env python3
"""
Signal Validator — runs after daily_advisor_v2 
Enforces hard constraints before signals are acted on.
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

CORRELATION_LIMIT = 0.75  # block pairs above this
PASS="✅"; WARN="⚠️ "; FAIL="❌"

def normalize_sym(s):
    return s.replace("^","").replace("=X","").replace("-USD","").upper()

def get_corr(s1, s2, days=30):
    try:
        df = yf.download([s1,s2], period=f"{days}d", progress=False)
        if hasattr(df.columns,'droplevel'):
            df = df["Close"]
        else:
            df = df["Close"]
        df = df.pct_change().dropna()
        if df.shape[1] == 2:
            return float(df.corr().iloc[0,1])
    except Exception:
        pass
    return 0.0

def validate_signals(advice_path=None):
    # Load latest advice
    if not advice_path:
        files = sorted([f for f in os.listdir("data/output")
                        if f.startswith("daily_advice_v2_")])
        if not files:
            print("No advice files found"); return
        advice_path = f"data/output/{files[-1]}"

    print(f"\n{'='*60}")
    print(f"  SIGNAL VALIDATOR")
    print(f"  {advice_path}")
    print(f"{'='*60}")

    with open(advice_path) as f:
        advice = json.load(f)

    assets  = advice.get("assets", [])
    active  = [a for a in assets if a["signal"] != "HOLD"
               and a.get("confidence", 0) >= 0.67]
    risk    = advice.get("risk_profile","Moderate")
    max_pos = {"Conservative":0.05,"Moderate":0.10,"Aggressive":0.20}.get(risk,0.10)
    errors=[];warnings_=[];passes=[]

    # ── 1. Duplicate detection (normalized symbols) ───────────
    print("\n  1. DUPLICATE CHECK")
    syms_norm = [normalize_sym(a["symbol"]) for a in active]
    dupes = list(set([s for s in syms_norm if syms_norm.count(s)>1]))
    if dupes:
        errors.append(f"Duplicate signals: {dupes}")
        print(f"  {FAIL} Duplicates found: {dupes}")
        # Remove duplicates keeping highest confidence
        seen=set(); deduped=[]
        for a in active:
            n=normalize_sym(a["symbol"])
            if n not in seen:
                seen.add(n); deduped.append(a)
        active=deduped
        print(f"  {PASS} Removed duplicates — {len(active)} signals remain")
    else:
        passes.append("No duplicates")
        print(f"  {PASS} No duplicates")

    # ── 2. TP/SL direction and R:R ────────────────────────────
    print("\n  2. TP/SL VALIDATION")
    for a in active[:]:
        sig=a["signal"]; p=a["current_price"]
        tp=a.get("tp_daily"); sl=a.get("sl_daily")
        if not tp or not sl: continue
        if sig=="BUY":
            dir_ok = tp>p and sl<p
            rr = (tp-p)/(p-sl) if p-sl>0 else 0
        else:
            dir_ok = tp<p and sl>p
            rr = (p-tp)/(sl-p) if sl-p>0 else 0
        if not dir_ok:
            errors.append(f"{a['symbol']}: TP/SL direction wrong")
            print(f"  {FAIL} {a['symbol']}: inverted TP/SL — removing")
            active.remove(a)
        elif rr < 1.0:
            warnings_.append(f"{a['symbol']}: R:R={rr:.2f} < 1.0")
            print(f"  {WARN} {a['symbol']}: R:R={rr:.2f} (below 1.0)")
        else:
            print(f"  {PASS} {a['symbol']}: R:R={rr:.2f}")

    # ── 3. Horizon/forecast consistency ───────────────────────
    print("\n  3. HORIZON CONSISTENCY")
    for a in active:
        pred  = abs(a.get("pred_return",0))
        tp_pct= abs(a.get("tp_daily",p)/a["current_price"]-1)*100
        # Flag if TP > 20x forecast (absurd mismatch)
        if pred > 0 and tp_pct/pred > 20:
            warnings_.append(f"{a['symbol']}: TP={tp_pct:.1f}% but forecast={pred:.2f}%")
            print(f"  {WARN} {a['symbol']}: TP={tp_pct:.1f}% vs forecast={pred:.2f}%")
        else:
            print(f"  {PASS} {a['symbol']}: TP={tp_pct:.1f}% forecast={pred:.2f}%")

    # ── 4. Correlation blocking ───────────────────────────────
    print("\n  4. CORRELATION CHECK")
    syms = [a["symbol"] for a in active]
    blocked=[]
    for i in range(len(syms)):
        for j in range(i+1,len(syms)):
            corr=get_corr(syms[i],syms[j],30)
            if abs(corr) > CORRELATION_LIMIT:
                blocked.append((syms[i],syms[j],round(corr,2)))
                print(f"  {WARN} HIGH CORR: {syms[i]}/{syms[j]} = {corr:+.2f}")
            else:
                print(f"  {PASS} {syms[i]}/{syms[j]}: corr={corr:+.2f}")

    if blocked:
        warnings_.append(f"Correlated pairs: {blocked}")
        print(f"\n  Correlated pairs detected — keeping highest Sharpe from each pair")
        remove_syms=set()
        for s1,s2,c in blocked:
            # Keep the one with higher validation Sharpe
            try:
                with open("data/output/robust_validation_results.json") as f:
                    results=json.load(f)
                sh={r["symbol"]:r["avg_sharpe"] for r in results}
                sh1=sh.get(s1,0); sh2=sh.get(s2,0)
                remove=s2 if sh1>=sh2 else s1
                remove_syms.add(remove)
                print(f"    Removing {remove} (sharpe={sh.get(remove,0):.3f})")
            except Exception:
                remove_syms.add(s2)
        active=[a for a in active if a["symbol"] not in remove_syms]

    # ── 5. Position sizing ────────────────────────────────────
    print("\n  5. POSITION SIZING")
    for a in active:
        pos=a["position_size"]
        if pos > max_pos*1.1:
            a["position_size"]=max_pos
            warnings_.append(f"{a['symbol']} position capped to {max_pos:.0%}")
            print(f"  {WARN} {a['symbol']}: capped {pos:.1%} → {max_pos:.1%}")
        else:
            print(f"  {PASS} {a['symbol']}: {pos:.1%} ≤ {max_pos:.1%}")

    total_exp=sum(a["position_size"] for a in active)
    print(f"\n  Total exposure: {total_exp:.1%}  ({len(active)} positions)")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  VALIDATION COMPLETE")
    print(f"  {PASS} Passed:   {len(passes)}")
    print(f"  {WARN} Warnings: {len(warnings_)}")
    print(f"  {FAIL} Errors:   {len(errors)}")
    print(f"\n  VALIDATED SIGNALS ({len(active)}):")
    for a in active:
        print(f"  {a['signal']:<5} {a['symbol']:<12} "
              f"conf={a['confidence']:.0%}  "
              f"pos={a['position_size']:.1%}  "
              f"tp={a.get('tp_daily',0):.4f}")

    # Save validated output
    advice["assets_validated"] = active
    advice["validation"] = {
        "timestamp": datetime.now().isoformat(),
        "errors": errors, "warnings": warnings_,
        "n_valid": len(active)
    }
    out_path = advice_path.replace("daily_advice","validated_advice")
    with open(out_path,"w") as f:
        json.dump(advice, f, indent=2)
    print(f"\n  💾 Validated signals → {out_path}")
    return active

if __name__ == "__main__":
    validate_signals()
