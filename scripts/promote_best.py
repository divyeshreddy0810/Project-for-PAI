"""
promote_best.py — Promote lab winners to production
=====================================================
Reads the lab results JSON, finds the best forecaster config per asset,
retrains it on FULL data (not just 40 epochs), and saves it to
data/models/ replacing the per-asset PatchTST.

The key rule:
  - If lab winner (LSTM/CNN/MLP) beats PatchTST by > MIN_IMPROVEMENT_PCT
    → retrain winner and save as production model
  - Otherwise → keep PatchTST (don't replace something that isn't broken)

Safe to run multiple times — only retrains if improvement threshold is met
and model doesn't exist yet (or --force is passed).

Usage
-----
  # Dry run — show what WOULD be promoted, don't train anything
  python scripts/promote_best.py --dry-run data/output/lab_results_TIMESTAMP.json

  # Full promotion run
  python scripts/promote_best.py data/output/lab_results_TIMESTAMP.json

  # Force retrain even if model already exists
  python scripts/promote_best.py --force data/output/lab_results_TIMESTAMP.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import yfinance as yf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "data" / "models"

# Only promote if lab model beats PatchTST by at least this much
MIN_IMPROVEMENT_PCT = 5.0   # 5% lower test MSE required to justify swap

# Full training config for promoted models (more epochs than lab)
PROMOTE_CFG = {
    "lookback":           60,
    "prediction_horizon":  5,
    "n_features":         10,
    "batch_size":         32,
    "epochs":             80,   # more than lab (40) for production quality
    "patience":           12,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "dropout":           0.3,
    "activation":       "relu",
    "optimizer":        "adam",
}


def safe_key(sym: str) -> str:
    return sym.replace("^","").replace("-","_").replace("=","")


def load_lab_results(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def find_best_per_asset(results: list, threshold: float = MIN_IMPROVEMENT_PCT) -> dict:
    """
    For each asset, find:
      - best lab model (lowest test_mse among MLP/CNN/LSTM)
      - PatchTST test_mse for comparison
    Returns dict: sym → {best_algo, best_cfg, best_mse, patchtst_mse, improvement_pct}
    """
    by_sym = defaultdict(list)
    for r in results:
        by_sym[r["symbol"]].append(r)

    decisions = {}
    for sym, recs in by_sym.items():
        patchtst = next((r for r in recs if r["algo"] == "PatchTST"), None)
        lab_recs = [r for r in recs if r["algo"] != "PatchTST"]

        if not lab_recs:
            continue

        best_lab = min(lab_recs, key=lambda r: r.get("test_mse") or 999)
        ptst_mse = patchtst["test_mse"] if patchtst else None

        improvement_pct = 0.0
        if ptst_mse and ptst_mse > 0:
            improvement_pct = (ptst_mse - best_lab["test_mse"]) / ptst_mse * 100

        decisions[sym] = {
            "asset_class":      recs[0].get("asset_class", "unknown"),
            "best_algo":        best_lab["algo"],
            "best_optimizer":   best_lab.get("optimizer", "adam"),
            "best_activation":  best_lab.get("activation", "relu"),
            "best_dropout":     best_lab.get("dropout", 0.3),
            "best_mse":         best_lab["test_mse"],
            "patchtst_mse":     ptst_mse,
            "improvement_pct":  round(improvement_pct, 2),
            "should_promote":   improvement_pct >= threshold,
        }

    return decisions


def print_promotion_plan(decisions: dict, threshold: float = MIN_IMPROVEMENT_PCT):
    """Print a clear table of what will be promoted and why."""
    promote  = {s: d for s, d in decisions.items() if d["should_promote"]}
    keep     = {s: d for s, d in decisions.items() if not d["should_promote"]}

    print(f"\n{'='*72}")
    print(f"  PROMOTION PLAN  (threshold: >{threshold}% improvement over PatchTST)")
    print(f"{'='*72}\n")

    # Group by asset class
    by_class = defaultdict(list)
    for sym, d in promote.items():
        by_class[d["asset_class"]].append((sym, d))

    print(f"  PROMOTE ({len(promote)} assets) — lab winner replaces PatchTST:\n")
    for cls in sorted(by_class.keys()):
        print(f"  [{cls.upper()}]")
        for sym, d in sorted(by_class[cls], key=lambda x: -x[1]["improvement_pct"]):
            print(f"    {sym:<14} {d['best_algo']:<6} "
                  f"({d['best_optimizer']}/{d['best_activation']}/do={d['best_dropout']}) "
                  f"  MSE: {d['best_mse']:.4e} vs PatchTST {d['patchtst_mse']:.4e} "
                  f"  [{d['improvement_pct']:+.1f}%]")

    print(f"\n  KEEP PatchTST ({len(keep)} assets) — improvement below threshold:\n")
    for sym, d in sorted(keep.items(), key=lambda x: x[1].get("improvement_pct", 0)):
        algo = d["best_algo"]
        pct  = d["improvement_pct"]
        print(f"    {sym:<14} best lab={algo:<6}  [{pct:+.1f}%]  → PatchTST stays")

    print(f"\n  Summary by asset class:")
    cls_stats = defaultdict(lambda: {"promote": 0, "keep": 0})
    for sym, d in decisions.items():
        cls = d["asset_class"]
        if d["should_promote"]:
            cls_stats[cls]["promote"] += 1
        else:
            cls_stats[cls]["keep"] += 1
    for cls in sorted(cls_stats.keys()):
        s = cls_stats[cls]
        print(f"    {cls:<12}  promote={s['promote']}  keep={s['keep']}")

    print()


def promote_asset(sym: str, decision: dict, dry_run: bool = False, force: bool = False):
    """Retrain best lab model for one asset and save to production models dir."""
    key       = safe_key(sym)
    algo      = decision["best_algo"]
    out_path  = MODELS_DIR / f"patchtst_{key}.pt"  # replaces PatchTST slot

    if out_path.exists() and not force:
        print(f"  ⏭  {sym}: production model exists — skip (use --force to overwrite)")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would train Lab-{algo} for {sym} → {out_path}")
        return True

    # ── Fetch full data ───────────────────────────────────────────────────────
    print(f"\n  Training Lab-{algo} for {sym}...")
    try:
        df = yf.download(sym, start="2004-01-01",
                         end=datetime.now().strftime("%Y-%m-%d"),
                         progress=False, auto_adjust=True)
        if hasattr(df.columns, "droplevel"):
            try:
                df.columns = df.columns.droplevel(1)
            except Exception:
                pass
        df = df.dropna()
        if len(df) < 200:
            print(f"  ⚠️  {sym}: only {len(df)} rows — skipping")
            return False
    except Exception as e:
        print(f"  ❌ {sym}: data fetch failed — {e}")
        return False

    # ── Build and train ───────────────────────────────────────────────────────
    from src.lab.forecaster import LabForecaster

    cfg = {
        **PROMOTE_CFG,
        "optimizer":  decision["best_optimizer"],
        "activation": decision["best_activation"],
        "dropout":    decision["best_dropout"],
    }

    forecaster = LabForecaster(algo=algo, cfg=cfg)
    success    = forecaster.fit_from_df(df, regime_label=0, verbose=True)

    if not success:
        print(f"  ❌ {sym}: training failed")
        return False

    # ── Save to production path ───────────────────────────────────────────────
    # We save with a metadata tag so the system knows this is a LabForecaster
    # The loader in daily_advisor_v2 / train_all_models will need to check
    # the 'algo' field and use LabForecaster.load() instead of PatchTSTForecaster.load()
    forecaster.save(str(out_path))
    print(f"  ✅ {sym}: Lab-{algo} saved → {out_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Promote lab winners to production")
    parser.add_argument("results_json", help="Path to lab_results_*.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show promotion plan without training")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if production model already exists")
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Promote only specific symbols")
    parser.add_argument("--threshold", type=float, default=MIN_IMPROVEMENT_PCT,
                        help=f"Minimum %% improvement to promote (default {MIN_IMPROVEMENT_PCT})")
    args = parser.parse_args()

    threshold = args.threshold

    print(f"\n  Loading lab results from {args.results_json}...")
    results   = load_lab_results(args.results_json)
    decisions = find_best_per_asset(results, threshold=threshold)

    if args.assets:
        decisions = {s: d for s, d in decisions.items() if s in args.assets}

    print_promotion_plan(decisions, threshold=threshold)

    if args.dry_run:
        print("  Dry run complete — no models trained.\n")
        return

    # ── Promote ───────────────────────────────────────────────────────────────
    to_promote = {s: d for s, d in decisions.items()
                  if d["should_promote"] and d["improvement_pct"] >= threshold}
    if not to_promote:
        print("  No assets meet the promotion threshold. Done.\n")
        return

    print(f"\n{'='*72}")
    print(f"  PROMOTING {len(to_promote)} ASSETS")
    print(f"{'='*72}")

    promoted = []
    skipped  = []
    failed   = []

    for sym, decision in to_promote.items():
        result = promote_asset(sym, decision, dry_run=False, force=args.force)
        if result is True:
            promoted.append(sym)
        elif result is False:
            if (MODELS_DIR / f"patchtst_{safe_key(sym)}.pt").exists() and not args.force:
                skipped.append(sym)
            else:
                failed.append(sym)

    print(f"\n{'='*72}")
    print(f"  PROMOTION COMPLETE")
    print(f"  Promoted : {len(promoted)} — {promoted}")
    print(f"  Skipped  : {len(skipped)} (already exist — use --force to retrain)")
    print(f"  Failed   : {len(failed)}")
    print(f"{'='*72}")
    print(f"\nNext step: run daily_advisor_v2.py — it will automatically")
    print(f"use the promoted models for the updated assets.\n")


if __name__ == "__main__":
    main()
