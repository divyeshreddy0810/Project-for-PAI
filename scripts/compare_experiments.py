"""
Compare Experiments
-------------------
Reads all results_*.json files from data/output/ and produces
a side-by-side comparison table printed to the terminal and
saved as data/output/comparison_<timestamp>.csv.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --results_dir data/output
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

OUTPUT_DIR = "data/output"

METRICS = [
    'forecast_mae',
    'forecast_rmse',
    'directional_accuracy',
    'total_return',
    'sharpe_ratio',
    'max_drawdown',
    'win_rate',
]

# Higher is better for these metrics
HIGHER_IS_BETTER = {
    'forecast_mae':           False,
    'forecast_rmse':          False,
    'directional_accuracy':   True,
    'total_return':           True,
    'sharpe_ratio':           True,
    'max_drawdown':           True,   # less negative = better
    'win_rate':               True,
}


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all results_*.json files from the output directory."""
    files = sorted([
        f for f in os.listdir(results_dir)
        if f.startswith('results_') and f.endswith('.json')
    ])

    if not files:
        print(f"❌ No results_*.json files found in {results_dir}")
        sys.exit(1)

    loaded = []
    for fname in files:
        path = os.path.join(results_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            agg = data.get('aggregated', {})
            agg['_source_file'] = fname
            loaded.append(agg)
            print(f"   ✅ Loaded: {fname}  ({agg.get('label', '?')})")
        except Exception as e:
            print(f"   ⚠️  Skipping {fname}: {e}")

    return loaded


def build_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """Build a DataFrame with one row per experiment."""
    rows = []
    for r in results:
        row = {'Experiment': r.get('label', '?'),
               'Windows':    r.get('n_windows', '?')}
        for m in METRICS:
            mean_key = m + '_mean'
            std_key  = m + '_std'
            mean_val = r.get(mean_key, np.nan)
            std_val  = r.get(std_key,  np.nan)
            row[m + '_mean'] = mean_val
            row[m + '_std']  = std_val
        rows.append(row)
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> str:
    """Pretty-print the comparison table with winner highlighting."""
    lines = []
    sep   = "=" * 90

    lines.append(sep)
    lines.append("EXPERIMENT COMPARISON")
    lines.append(sep)

    # Header
    lines.append(f"\n{'Metric':<28} " +
                 "  ".join(f"{row['Experiment']:>18}"
                            for _, row in df.iterrows()))
    lines.append("-" * 90)

    for m in METRICS:
        col  = m + '_mean'
        scol = m + '_std'

        if col not in df.columns:
            continue

        vals = df[col].values.astype(float)

        # Find winner index
        valid = ~np.isnan(vals)
        if valid.sum() > 0:
            if HIGHER_IS_BETTER[m]:
                winner = int(np.nanargmax(vals))
            else:
                winner = int(np.nanargmin(vals))
        else:
            winner = -1

        # Format each cell
        cells = []
        for i, (_, row) in enumerate(df.iterrows()):
            v = row.get(col, np.nan)
            s = row.get(scol, np.nan)

            if np.isnan(v):
                cell = f"{'N/A':>18}"
            else:
                # Format value
                if m in ('total_return', 'directional_accuracy',
                         'win_rate', 'max_drawdown'):
                    val_str = f"{v:+.1%}"
                elif m in ('sharpe_ratio',):
                    val_str = f"{v:+.3f}"
                else:
                    val_str = f"{v:.4f}"

                std_str = f"±{s:.3f}" if not np.isnan(s) else ""
                cell    = f"{val_str} {std_str:>8}"

                if i == winner:
                    cell = f"★ {cell}"
                else:
                    cell = f"  {cell}"

                cell = f"{cell:>20}"

            cells.append(cell)

        lines.append(f"{m:<28}" + "".join(cells))

    lines.append("-" * 90)
    lines.append("★ = better value for that metric")
    lines.append(sep)
    return "\n".join(lines)


def save_comparison_csv(df: pd.DataFrame, output_dir: str) -> str:
    """Save comparison table to CSV."""
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(output_dir, f"comparison_{ts}.csv")

    # Flatten to one row per experiment, one col per metric
    export_rows = []
    for _, row in df.iterrows():
        flat = {'Experiment': row['Experiment'],
                'Windows':    row['Windows']}
        for m in METRICS:
            flat[m] = row.get(m + '_mean', np.nan)
            flat[m + '_std'] = row.get(m + '_std', np.nan)
        export_rows.append(flat)

    pd.DataFrame(export_rows).to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Compare backtest experiment results'
    )
    parser.add_argument(
        '--results_dir', default=OUTPUT_DIR,
        help=f'Directory containing results_*.json files (default: {OUTPUT_DIR})'
    )
    args = parser.parse_args()

    print(f"\n📊 EXPERIMENT COMPARISON")
    print(f"   Loading results from: {args.results_dir}\n")

    results = load_results(args.results_dir)

    if len(results) < 2:
        print(f"\n⚠️  Only {len(results)} experiment(s) found. "
              f"Run both configs before comparing.")
        if len(results) == 1:
            print(f"   Found: {results[0].get('label', '?')}")
        sys.exit(0)

    df  = build_comparison_table(results)
    tbl = format_table(df)

    print(tbl)

    csv_path = save_comparison_csv(df, args.results_dir)
    print(f"\n✅ Comparison saved → {csv_path}\n")


if __name__ == '__main__':
    main()
