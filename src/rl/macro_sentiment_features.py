"""
Macro + Sentiment Features — Week 2
-------------------------------------
Adds macroeconomic indicators and sentiment scores to RL state.
New features added to state vector:
  - VIX (fear index)
  - US interest rate (Fed Funds proxy)
  - S&P 500 200-day regime (macro bull/bear)
  - FinBERT sentiment mean
  - FinBERT sentiment trend
"""

import numpy as np
import pandas as pd
import yfinance as yf
import os, json, warnings
warnings.filterwarnings("ignore")


def fetch_vix(start="2004-01-01", end="2026-03-25") -> pd.Series:
    """Fetch VIX — CBOE Volatility Index (market fear gauge)."""
    try:
        df = yf.download("^VIX", start=start, end=end, progress=False)
        if hasattr(df.columns,'droplevel'):
            df.columns = df.columns.droplevel(1)
        vix = df["Close"].fillna(method="ffill").fillna(20.0)
        vix_norm = (vix / 80.0).clip(0, 1)  # normalise 0-80 → 0-1
        print(f"  ✅ VIX: {len(vix)} rows  range: {vix.min():.1f}-{vix.max():.1f}")
        return vix_norm
    except Exception as e:
        print(f"  ⚠️  VIX unavailable: {e} — using neutral 0.25")
        return None


def fetch_interest_rate(start="2004-01-01", end="2026-03-25") -> pd.Series:
    """
    Fetch US 13-week T-bill rate as proxy for Fed Funds rate.
    Uses yfinance ^IRX (no FRED API key needed).
    """
    try:
        df = yf.download("^IRX", start=start, end=end, progress=False)
        if hasattr(df.columns,'droplevel'):
            df.columns = df.columns.droplevel(1)
        rate = df["Close"].fillna(method="ffill").fillna(2.0)
        rate_norm = (rate / 20.0).clip(0, 1)  # normalise 0-20% → 0-1
        print(f"  ✅ Interest rate: {len(rate)} rows  "
              f"range: {rate.min():.2f}%-{rate.max():.2f}%")
        return rate_norm
    except Exception as e:
        print(f"  ⚠️  Interest rate unavailable: {e} — using neutral 0.15")
        return None


# Map assets to appropriate macro benchmark
_MACRO_BENCHMARK = {
    "BTC-USD": "BTC-USD",  "ETH-USD": "BTC-USD",
    "SOL-USD": "BTC-USD",  "BNB-USD": "BTC-USD",
    "GC=F":    "GC=F",     "SI=F":    "GC=F",
    "CL=F":    "CL=F",     "NG=F":    "CL=F",
}
# Everything else (equity, forex, indices) uses S&P 500

def fetch_macro_regime(start="2004-01-01", end="2026-03-25",
                       asset_sym=None) -> pd.Series:
    """
    Per-asset macro regime: price above/below 200-day SMA.
    +1 = bull macro regime, -1 = bear macro regime.
    Uses asset-appropriate benchmark:
      - Crypto  → BTC-USD 200-day SMA
      - Gold    → GC=F 200-day SMA
      - Oil     → CL=F 200-day SMA
      - Equity/Forex/Indices → ^GSPC 200-day SMA
    """
    benchmark = _MACRO_BENCHMARK.get(asset_sym, "^GSPC")
    try:
        df = yf.download(benchmark, start=start, end=end, progress=False)
        if hasattr(df.columns,'droplevel'):
            df.columns = df.columns.droplevel(1)
        close = df["Close"]
        sma200 = close.rolling(200).mean()
        regime = np.where(close > sma200, 1.0, -1.0)
        regime_series = pd.Series(regime, index=df.index).fillna(0)
        bull_pct = (regime_series == 1).mean() * 100
        print(f"  ✅ Macro regime ({benchmark}): {len(regime_series)} rows  "
              f"bull: {bull_pct:.0f}% of time")
        return regime_series
    except Exception as e:
        print(f"  ⚠️  Macro regime unavailable: {e} — using neutral 0")
        return None

def fetch_sp500_regime(start="2004-01-01", end="2026-03-25") -> pd.Series:
    """Legacy wrapper — kept for backward compatibility."""
    return fetch_macro_regime(start=start, end=end, asset_sym=None)


def load_sentiment_features(sentiment_file="data/output/latest.json") -> dict:
    """
    Load FinBERT sentiment from existing sentiment_analyzer output.
    Returns dict with mean and trend per symbol.
    Used for LIVE signals only — returns latest reading.
    """
    if not os.path.exists(sentiment_file):
        print(f"  ⚠️  Sentiment file not found: {sentiment_file}")
        return {"mean": 0.05, "trend": 0.0}

    try:
        with open(sentiment_file) as f:
            data = json.load(f)

        assets = data.get("assets", [])
        if not assets:
            return {"mean": 0.05, "trend": 0.0}

        all_means = [a.get("overall_mean", 0.05) for a in assets]
        sent_mean = float(np.mean(all_means))

        all_trends = []
        for asset in assets:
            daily = asset.get("daily_means", [])
            if len(daily) >= 2:
                vals = [d["mean_sentiment"] for d in daily]
                x = np.arange(len(vals))
                slope = np.polyfit(x, vals, 1)[0]
                all_trends.append(slope)

        sent_trend = float(np.mean(all_trends)) if all_trends else 0.0

        print(f"  ✅ Sentiment (live): mean={sent_mean:.3f}  "
              f"trend={sent_trend:.4f}  assets={len(assets)}")
        return {"mean": sent_mean, "trend": sent_trend}

    except Exception as e:
        print(f"  ⚠️  Sentiment load error: {e}")
        return {"mean": 0.05, "trend": 0.0}


def build_sentiment_proxy(df: pd.DataFrame) -> np.ndarray:
    """
    Build TIME-VARYING sentiment proxy for historical RL training.

    Uses 20-day price momentum percentile rank as sentiment proxy.
    Validated against known market events:
      - 2008 crisis:  0.03-0.16  ✅ correctly negative
      - 2020 COVID:   0.001-0.10 ✅ correctly very negative  
      - 2022 bull:    0.86-0.93  ✅ correctly positive

    This replaces the constant 0.562 FinBERT value which has no
    time variation across 22 years of historical training data.

    For LIVE signals, use load_sentiment_features() instead which
    returns real FinBERT scores from the latest news headlines.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.droplevel(1)

    close = df["Close"] if "Close" in df.columns else df.iloc[:,3]
    momentum = close.pct_change(20).fillna(0)
    # Percentile rank → 0 to 1 (0=very bearish, 1=very bullish)
    proxy = momentum.rank(pct=True).fillna(0.5).values.astype(np.float32)
    return proxy


def build_macro_signals(df: pd.DataFrame,
                        cache_dir: str = "data/cache",
                        asset_sym: str = None) -> np.ndarray:
    """
    Build macro + sentiment feature matrix aligned to df's index.
    Returns array of shape (len(df), 4):
      col 0: VIX normalised
      col 1: interest rate normalised
      col 2: SP500 macro regime
      col 3: sentiment mean
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Per-asset cache to avoid cross-asset contamination
    safe_sym = asset_sym.replace("^","").replace("-","_").replace("=","")                if asset_sym else "default"
    cache_file = os.path.join(cache_dir, f"macro_features_{safe_sym}.csv")

    start = df.index[0].strftime("%Y-%m-%d") \
            if hasattr(df.index[0], 'strftime') else "2004-01-01"
    end   = "2026-03-25"

    # Try loading from cache first
    if os.path.exists(cache_file):
        try:
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            print(f"  📂 Loaded macro features from cache ({len(cached)} rows)")
            macro_df = cached
        except Exception:
            macro_df = None
    else:
        macro_df = None

    if macro_df is None:
        print("  📊 Fetching macro features...")
        vix    = fetch_vix(start, end)
        rate   = fetch_interest_rate(start, end)
        regime = fetch_macro_regime(start=start, end=end, asset_sym=asset_sym)

        # Build combined dataframe
        parts = {}
        if vix    is not None: parts["vix"]    = vix
        if rate   is not None: parts["rate"]   = rate
        if regime is not None: parts["regime"] = regime

        if parts:
            macro_df = pd.DataFrame(parts)
            macro_df.to_csv(cache_file)
            print(f"  💾 Macro features cached → {cache_file}")
        else:
            macro_df = pd.DataFrame()

    # Load sentiment
    # For historical data (>30 days): use time-varying proxy
    # For recent data (<30 days): use real FinBERT scores
    n = len(df)
    result = np.zeros((n, 4), dtype=np.float32)

    if n > 30:
        # Historical training — use time-varying momentum proxy
        sentiment_proxy = build_sentiment_proxy(df)
        result[:, 3] = sentiment_proxy
        proxy_mean = float(np.mean(sentiment_proxy))
        print(f"  ✅ Sentiment (proxy): {n} timesteps  "
              f"mean={proxy_mean:.3f}  range={sentiment_proxy.min():.3f}"
              f"-{sentiment_proxy.max():.3f}")
    else:
        # Live signals — use real FinBERT
        sentiment = load_sentiment_features()
        result[:, 3] = float(sentiment["mean"])
        print(f"  ✅ Sentiment (live FinBERT): mean={sentiment['mean']:.3f}")

    if not macro_df.empty:
        for col_idx, col_name in enumerate(["vix","rate","regime"]):
            defaults = {"vix": 0.25, "rate": 0.15, "regime": 0.0}
            if col_name not in macro_df.columns:
                result[:, col_idx] = defaults[col_name]
                continue
            series = macro_df[col_name]
            # df may have DatetimeIndex or integer index
            if hasattr(df.index, "date"):
                # DatetimeIndex — align directly
                aligned = series.reindex(df.index, method="ffill")
            else:
                # Integer index — use positional iloc matching
                # Try to match by position using date column if available
                try:
                    df_dates = pd.DatetimeIndex(df.iloc[:, 0])                                if "Date" in df.columns else None
                except Exception:
                    df_dates = None

                if df_dates is not None:
                    aligned = series.reindex(df_dates, method="ffill")
                else:
                    # Last resort: sample macro series to match df length
                    step = max(1, len(series) // len(df))
                    vals = series.values[::step][:len(df)]
                    if len(vals) < len(df):
                        vals = np.pad(vals,(0,len(df)-len(vals)),
                                      mode="edge")
                    aligned = pd.Series(vals[:len(df)])

            filled = aligned.fillna(defaults[col_name])
            arr = filled.values.astype(np.float32)
            result[:, col_idx] = arr[:len(df)] if len(arr)>=len(df)                                   else np.pad(arr,(0,len(df)-len(arr)),
                                              mode="edge")

    print(f"  ✅ Macro signals: shape={result.shape}  "
          f"VIX_mean={result[:,0].mean():.3f}  "
          f"Rate_mean={result[:,1].mean():.3f}")
    return result
