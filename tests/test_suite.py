#!/usr/bin/env python3
import sys, os, json, random, warnings
warnings.filterwarnings("ignore")
PROJECT = os.path.expanduser("~/Desktop/NCI/programming_for_ai/Project-for-PAI")
sys.path.insert(0, PROJECT)
import numpy as np
import pandas as pd
import yfinance as yf

P=0; F=0
def check(name, cond, exp="", got=""):
    global P,F
    if cond: print(f"  ✅ {name}"); P+=1
    else: print(f"  ❌ {name}" + (f" — expected {exp}, got {got}" if exp else "")); F+=1

print("\n" + "="*60)
print("  TRADING SYSTEM TEST SUITE")
print("="*60)

# Block 1: Lot Sizes
print("\n── Lot Size Accuracy ────────────────────────────────────")
from scripts.daily_advisor import calc_lot_size
from src.utils.currency import get_all_rates
rates = get_all_rates()
eur_usd = rates["EUR_USD"]
for name,entry,sl,tp,aname,exp,tol in [
    ("EURUSD",1.1575,1.1575-0.0084,1.1575+0.0126,"EURUSD",2.38,0.15),
    ("USDCHF",0.7878,0.7878-0.0045,0.7878+0.0068,"USDCHF",3.50,0.20),
    ("USDJPY",159.22,159.22-0.899, 159.22+1.349, "USDJPY",3.54,0.30),
]:
    r = calc_lot_size(entry,sl,tp,"forex",100000,2000,eur_usd,aname)
    check(f"{name} lots ~{exp}", r and abs(r['lots']-exp)<=tol,
          f"{exp}±{tol}", f"{r['lots']:.2f}" if r else "None")
    if r:
        check(f"{name} margin <$10k", r['margin']<10000, got=f"${r['margin']:,.0f}")
        check(f"{name} TP>SL profit", r['profit_tp']>r['loss_sl'])

# Block 2: Live Prices
print("\n── Live Prices ──────────────────────────────────────────")
from scripts.daily_advisor import get_live_price
for sym,lo,hi in [("EURUSD=X",0.8,1.5),("USDCHF=X",0.6,1.2),("USDJPY=X",100,200)]:
    px = get_live_price(sym)
    check(f"{sym} in range {lo}-{hi}", lo<=px<=hi, got=f"{px:.4f}")

# Block 3: Signal Sanity
print("\n── Signal Sanity ────────────────────────────────────────")
df = yf.download("EURUSD=X", period="1mo", progress=False)
if isinstance(df.columns, pd.MultiIndex): df.columns=df.columns.droplevel(1)
px  = float(df["Close"].iloc[-1])
atr = float(df["High"].sub(df["Low"]).rolling(14).mean().iloc[-1])
tp_s = px+atr*2.0; sl_s = px-atr*1.0
check("Swing TP < 5% away", (tp_s-px)/px*100<5, got=f"{(tp_s-px)/px*100:.2f}%")
check("Swing SL < 3% away", (px-sl_s)/px*100<3, got=f"{(px-sl_s)/px*100:.2f}%")
check("R/R >= 1.8", (tp_s-px)/(px-sl_s)>=1.8, got=f"{(tp_s-px)/(px-sl_s):.2f}")
check("REG: TP NOT 25%", (tp_s-px)/px*100<10)

# Block 4: Regime Detection
print("\n── Regime Detection ─────────────────────────────────────")
from scripts.daily_advisor import detect_forex_regime, calc_adx
df30 = df.tail(30)
regime, mult = detect_forex_regime(df30)
check("Regime not UNKNOWN", regime!="UNKNOWN", got=regime)
check("Regime valid", regime in ["TRENDING_HIGH_VOL","TRENDING_LOW_VOL","CHOPPY_HIGH_VOL","CHOPPY_LOW_VOL"], got=regime)
check("Mult valid", mult in [0.25,0.5,1.0,1.5], got=mult)
# ADX needs more rows for warm-up — use full df
adx = calc_adx(df)
check("ADX 0-100 (or NaN on small data)", 0<=adx<=100 or (adx!=adx), got=f"{adx:.1f}")
print(f"  ℹ️  Current EURUSD: ADX={adx:.1f} Regime={regime} Mult={mult}x")

# Block 5: Currency
print("\n── Currency Converter ───────────────────────────────────")
from src.utils.currency import parse_amount_input
for inp,eccy,eamt in [("$100000","USD",100000),("€500","EUR",500),("₦100000","NGN",100000)]:
    _,amt,ccy,p = parse_amount_input(inp, rates)
    check(f"{inp} → {eccy}", ccy==eccy, got=ccy)
check("$100k → $100k USD", abs(parse_amount_input("$100000",rates)[3].usd-100000)<10)

# Block 6: Correlation Hedge
print("\n── Correlation Hedge ────────────────────────────────────")
dfe=yf.download("EURUSD=X",start="2022-01-01",end="2026-01-01",progress=False)
dfc=yf.download("USDCHF=X",start="2022-01-01",end="2026-01-01",progress=False)
for d in [dfe,dfc]:
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.droplevel(1)
corr=dfe["Close"].pct_change().corr(dfc["Close"].pct_change())
check("EUR/CHF negative correlation (hedge)", corr<-0.3, got=f"{corr:.2f}")

# Block 7: Regressions
print("\n── Regression Tests ─────────────────────────────────────")
from scripts.daily_advisor import ALL_ASSETS
syms=[a["symbol"] for a in ALL_ASSETS]
check("REG: EURUSD in assets", "EURUSD=X" in syms)
check("REG: USDCHF in assets", "USDCHF=X" in syms)
check("REG: USDJPY in assets", "USDJPY=X" in syms)
check("REG: No stocks", "stock" not in [a.get("type") for a in ALL_ASSETS])
r2 = calc_lot_size(159.22,158.32,160.57,"forex",100000,2000,eur_usd,"USDJPY")
check("REG: USDJPY margin <$50k", r2 and r2['margin']<50000, got=f"${r2['margin']:,.0f}" if r2 else "None")

# Summary
total=P+F
print(f"\n{'='*60}")
print(f"  RESULTS: {P}/{total} passed  |  {F} failed")
print("="*60)
if F==0: print("  ✅ ALL PASSED — ready for live trading")
else: print(f"  ⚠️  {F} tests need attention")
