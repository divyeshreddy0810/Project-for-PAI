"""Multimodal consensus engine — requires 3-model agreement before trading."""
from typing import Dict, Any

def calculate_consensus(hmm_regime, lgbm_expected_return, lgbm_up_probability,
                        adx_regime, signal_direction) -> Dict[str, Any]:
    agreements=0; reasons=[]
    is_buy = signal_direction.upper() in ("BUY","STRONG_BUY")

    # 1 HMM
    if is_buy and hmm_regime=="bull":
        agreements+=1; reasons.append("HMM=BULL ✓")
    elif not is_buy and hmm_regime=="bear":
        agreements+=1; reasons.append("HMM=BEAR ✓")
    else:
        reasons.append(f"HMM={hmm_regime.upper()} ✗")

    # 2 LightGBM
    lgbm_ok=(lgbm_expected_return>0.003 and lgbm_up_probability>0.52) if is_buy \
            else (lgbm_expected_return<-0.003 and lgbm_up_probability<0.48)
    if lgbm_ok:
        agreements+=1; reasons.append(f"LightGBM={lgbm_expected_return*100:+.2f}% ✓")
    else:
        reasons.append(f"LightGBM={lgbm_expected_return*100:+.2f}% ✗")

    # 3 ADX
    if "TRENDING" in adx_regime.upper():
        agreements+=1; reasons.append(f"ADX={adx_regime.replace('_',' ')} ✓")
    else:
        reasons.append("ADX=CHOPPY ✗")

    if agreements==3:   consensus="STRONG";   mult=1.5
    elif agreements==2: consensus="WEAK";     mult=0.75
    else:               consensus="NO_TRADE"; mult=0.0

    return {"consensus":consensus,"size_multiplier":mult,
            "agreements":agreements,"reasoning":"  ".join(reasons)}

def format_consensus_line(r, base_risk=2000.0):
    emoji={"STRONG":"✅","WEAK":"⚠️","NO_TRADE":"❌"}.get(r["consensus"],"")
    return (f"  │  Consensus  : {emoji} {r['consensus']}  ({r['reasoning']})\n"
            f"  │  Size mult  : {r['size_multiplier']:.2f}x → risk ${base_risk*r['size_multiplier']:,.0f}")
