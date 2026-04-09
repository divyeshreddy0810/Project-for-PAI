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
import os, json, time, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

_TODAY = datetime.now().strftime("%Y-%m-%d")


def fetch_vix(start="2004-01-01", end=None) -> pd.Series:
    if end is None: end = _TODAY
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


def fetch_interest_rate(start="2004-01-01", end=None) -> pd.Series:
    if end is None: end = _TODAY
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


# Map assets to appropriate macro benchmark (200-day SMA regime)
_MACRO_BENCHMARK = {
    "BTC-USD": "BTC-USD",  "ETH-USD": "BTC-USD",
    "SOL-USD": "BTC-USD",  "BNB-USD": "BTC-USD",
    "GC=F":    "GC=F",     "SI=F":    "GC=F",
    "CL=F":    "CL=F",     "NG=F":    "CL=F",
}
# Everything else (equity, forex, indices) uses S&P 500

# Map assets to their sector peer leader (for cross-asset momentum feature).
# When NVDA rips 8% in a day, SMCI/ARM/TSM follow — this captures contagion
# that per-asset models are completely blind to.
_PEER_BENCHMARK = {
    # Semiconductor / AI supply chain → NVDA leads
    "CRWV": "NVDA", "NBIS": "NVDA", "IREN": "NVDA",
    "MU":   "NVDA", "SMCI": "NVDA", "ARM":  "NVDA",
    "TSM":  "NVDA", "VRT":  "NVDA", "MRVL": "NVDA",
    "NVDA": "SOXX",      # NVDA's own peer is the semiconductor ETF
    # Crypto → BTC leads
    "ETH-USD": "BTC-USD", "SOL-USD": "BTC-USD",
    "BNB-USD": "BTC-USD", "XRP-USD": "BTC-USD",
    # Commodities
    "SI=F": "GC=F",  "HG=F": "GC=F",   # silver & copper follow gold
    "NG=F": "CL=F",  "ZW=F": "CL=F",   # nat gas / grains follow oil cycle
    "ZC=F": "CL=F",
    # Forex → EUR/USD is the risk-sentiment proxy
    "GBPUSD=X": "EURUSD=X", "AUDUSD=X": "EURUSD=X",
    "NZDUSD=X": "EURUSD=X", "USDCAD=X": "EURUSD=X",
    "USDCHF=X": "EURUSD=X", "EURJPY=X": "EURUSD=X",
    "GBPJPY=X": "EURUSD=X", "EURGBP=X": "EURUSD=X",
    "AUDNZD=X": "EURUSD=X",
    # African / EM forex → USD/ZAR is the EM risk proxy
    "USDNGN=X": "USDZAR=X", "EURNGN=X": "USDZAR=X",
    "USDKES=X": "USDZAR=X", "USDGHS=X": "USDZAR=X",
}


def _is_equity_sym(sym: str) -> bool:
    """True if symbol looks like an individual equity (no yfinance suffix)."""
    if not sym:
        return False
    return not any(sym.endswith(s) for s in ["=X", "-USD", "=F"])


def fetch_peer_momentum(asset_sym: str, start: str, end: str):
    """
    Fetch the 5-day return of the sector peer leader.
    Normalised to [-1, 1] by clipping at ±15%.
    Returns pd.Series indexed by date, or None on failure.
    """
    peer = _PEER_BENCHMARK.get(asset_sym)
    if peer is None or peer == asset_sym:
        return None
    try:
        df = yf.download(peer, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.dropna()
        if len(df) < 10:
            return None
        ret_5d     = df["Close"].pct_change(5).fillna(0)
        normalized = (ret_5d / 0.15).clip(-1.0, 1.0)
        print(f"  ✅ Peer momentum ({peer}→{asset_sym}): "
              f"{len(normalized)} rows  "
              f"mean={normalized.mean():.3f}")
        return normalized
    except Exception as e:
        print(f"  ⚠️  Peer momentum ({peer}) failed: {e} — using 0")
        return None


def fetch_earnings_proximity(asset_sym: str, n: int,
                              is_live: bool = False) -> np.ndarray:
    """
    Earnings proximity feature: how close is the next earnings date?
    Returns array of shape (n,):
      Training (is_live=False): all zeros (no historical earnings data)
      Live inference (is_live=True):
        1.0 = earnings tomorrow, 0.0 = 14+ days away (linear decay)

    Why zero during training: yfinance only has ~2 years of earnings history
    vs 20+ years of training data. Rather than partial data, we set the
    baseline at 0 — the agent learns the normal state. At inference, a
    non-zero value is a meaningful safety warning to the user.

    Only applies to equity symbols. Always 0 for forex/crypto/commodities.
    """
    result = np.zeros(n, dtype=np.float32)

    if not is_live or not _is_equity_sym(asset_sym):
        return result

    try:
        ticker = yf.Ticker(asset_sym)
        cal    = ticker.calendar

        earnings_date = None
        if cal is None:
            return result

        if isinstance(cal, dict):
            for key in ("Earnings Date", "earnings_date", "earningsDate"):
                val = cal.get(key)
                if val is not None:
                    if hasattr(val, "__iter__") and not isinstance(val, str):
                        earnings_date = pd.Timestamp(list(val)[0])
                    else:
                        earnings_date = pd.Timestamp(val)
                    break
        elif hasattr(cal, "columns"):
            for col in cal.columns:
                if "earnings" in str(col).lower():
                    earnings_date = pd.Timestamp(cal[col].iloc[0])
                    break

        if earnings_date is None:
            return result

        days_away = max(0, (earnings_date.tz_localize(None) -
                            pd.Timestamp.now()).days)
        proximity  = float(max(0.0, 1.0 - days_away / 14.0))
        result[:]  = proximity

        if proximity > 0.0:
            print(f"  ⚠️  {asset_sym} earnings in {days_away}d → "
                  f"proximity={proximity:.2f} (reduce confidence)")
    except Exception as e:
        print(f"  ⚠️  Earnings calendar {asset_sym}: {e}")

    return result

def fetch_macro_regime(start="2004-01-01", end=None,
                       asset_sym=None) -> pd.Series:
    if end is None: end = _TODAY
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

def fetch_sp500_regime(start="2004-01-01", end=None) -> pd.Series:
    if end is None: end = _TODAY
    """Legacy wrapper — kept for backward compatibility."""
    return fetch_macro_regime(start=start, end=end, asset_sym=None)


def load_sentiment_features(sentiment_file="data/output/latest.json",
                            asset_sym=None) -> dict:
    """
    Load FinBERT sentiment from existing sentiment_analyzer output.
    Returns dict with mean and trend per symbol.
    Used for LIVE signals only — returns latest reading.
    If asset_sym provided, returns that asset's sentiment specifically.
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

        # Per-asset sentiment lookup
        if asset_sym:
            for a in assets:
                sym = a.get("symbol","")
                if sym == asset_sym or sym.replace("=X","") == asset_sym.replace("=X",""):
                    mean_val = a.get("overall_mean",
                               a.get("sentiment_mean", 0.05))
                    try:
                        mean_val = float(mean_val)
                    except (TypeError, ValueError):
                        mean_val = 0.05
                    return {"mean": mean_val, "trend": 0.0,
                            "source": "finbert"}
            # Asset not found in sentiment file — use neutral
            print(f"  ⚠️  No sentiment for {asset_sym} — using neutral 0.05")
            return {"mean": 0.05, "trend": 0.0, "source": "stub"}

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

    Uses Bollinger Band positioning + volume confirmation divergence.
    This captures MARKET POSITIONING (how stretched/extreme the move is)
    rather than raw momentum, which is already present in LightGBM features
    (ret_roll_mean20, price_vs_ma20, ret_lag10) and would be redundant.

    BB position: where price sits within its ±2σ band → market greed/fear
    Volume divergence: price moving without volume = weak conviction
    Both are orthogonal signals to the momentum features already in the model.

    Validated against known market events:
      - 2008 crisis:  low BB position + vol spike = extreme fear  ✅
      - 2020 COVID:   crash below lower band = extreme fear        ✅
      - 2021 bull:    sustained upper-band hugging = extreme greed ✅

    For LIVE signals, use load_sentiment_features() instead which
    returns real FinBERT scores from the latest news headlines.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.droplevel(1)

    close = df["Close"] if "Close" in df.columns else df.iloc[:, 3]

    # ── Bollinger Band position ─────────────────────────────────
    # Captures how "stretched" the market is: fear (below band) ↔ greed (above band)
    roll_mean = close.rolling(20, min_periods=5).mean()
    roll_std  = close.rolling(20, min_periods=5).std().replace(0, np.nan)
    bb_pos    = ((close - roll_mean) / (2 * roll_std)).clip(-1.5, 1.5).fillna(0)
    # Rescale to [0, 1]: -1.5 → 0.0 (extreme fear), +1.5 → 1.0 (extreme greed)
    bb_norm   = ((bb_pos + 1.5) / 3.0).fillna(0.5)

    # ── Volume confirmation divergence ─────────────────────────
    # Price up + volume up   = genuine conviction → confirm signal
    # Price up + volume down = weak hands moving price → fade signal
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        price_dir = np.sign(close.pct_change(5).fillna(0))
        vol_chg   = df["Volume"].pct_change(5).fillna(0)
        vol_rank  = pd.Series(vol_chg).rank(pct=True).fillna(0.5)
        # When price direction and volume agree: vol_rank pulls toward extreme
        # When they disagree: vol_rank pulls toward neutral (0.5)
        confirmation = price_dir.values * (vol_rank.values - 0.5) + 0.5
        confirmation = pd.Series(confirmation).clip(0, 1).fillna(0.5)
        proxy = (bb_norm.values * 0.65 + confirmation.values * 0.35).astype(np.float32)
    else:
        proxy = bb_norm.values.astype(np.float32)

    return proxy


def build_macro_signals(df: pd.DataFrame,
                        cache_dir: str = "data/cache",
                        asset_sym: str = None) -> np.ndarray:
    """
    Build macro + sentiment feature matrix aligned to df's index.
    Returns array of shape (len(df), 6):
      col 0: VIX normalised
      col 1: interest rate normalised
      col 2: macro regime (200-day SMA of benchmark)
      col 3: sentiment mean (BB proxy for training, FinBERT for live)
      col 4: peer group momentum (5-day return of sector leader, [-1,1])
      col 5: earnings proximity (0=no earnings, 1=earnings tomorrow; equity only)
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_sym   = (asset_sym.replace("^","").replace("-","_").replace("=","")
                  if asset_sym else "default")
    cache_file = os.path.join(cache_dir, f"macro_features_{safe_sym}.csv")

    start = (df.index[0].strftime("%Y-%m-%d")
             if hasattr(df.index[0], 'strftime') else "2004-01-01")
    end   = _TODAY

    # ── Load or build cached macro data (VIX, rate, regime, peer_momentum) ─
    CACHE_MAX_AGE_DAYS = 7
    macro_df = None
    if os.path.exists(cache_file):
        cache_age_days = (time.time() - os.path.getmtime(cache_file)) / 86400
        if cache_age_days > CACHE_MAX_AGE_DAYS:
            print(f"  ♻️  Macro cache stale ({cache_age_days:.1f}d) — refreshing")
        else:
            try:
                cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Force refresh if cache predates peer_momentum column
                if "peer_momentum" not in cached.columns:
                    print(f"  ♻️  Macro cache missing peer_momentum — refreshing")
                else:
                    print(f"  📂 Loaded macro features from cache "
                          f"({len(cached)} rows, {cache_age_days:.1f}d old)")
                    macro_df = cached
            except Exception:
                pass

    if macro_df is None:
        print("  📊 Fetching macro features...")
        vix    = fetch_vix(start, end)
        rate   = fetch_interest_rate(start, end)
        regime = fetch_macro_regime(start=start, end=end, asset_sym=asset_sym)
        peer   = fetch_peer_momentum(asset_sym, start, end)

        parts = {}
        if vix    is not None: parts["vix"]           = vix
        if rate   is not None: parts["rate"]          = rate
        if regime is not None: parts["regime"]        = regime
        if peer   is not None: parts["peer_momentum"] = peer

        if parts:
            macro_df = pd.DataFrame(parts)
            macro_df.to_csv(cache_file)
            print(f"  💾 Macro features cached → {cache_file}")
        else:
            macro_df = pd.DataFrame()

    # ── Build result matrix (n, 6) ────────────────────────────────────────
    n      = len(df)
    is_live = (n <= 200)
    result = np.zeros((n, 6), dtype=np.float32)

    # col 3: sentiment
    if not is_live:
        sentiment_proxy = build_sentiment_proxy(df)
        result[:, 3]    = sentiment_proxy
        print(f"  ✅ Sentiment (BB proxy): {n} rows  "
              f"mean={sentiment_proxy.mean():.3f}  "
              f"range=[{sentiment_proxy.min():.3f},{sentiment_proxy.max():.3f}]")
    else:
        sentiment      = load_sentiment_features(asset_sym=asset_sym)
        result[:, 3]   = float(sentiment["mean"])
        src_tag        = sentiment.get("source", "finbert")
        print(f"  ✅ Sentiment ({src_tag}): {asset_sym} mean={sentiment['mean']:.3f}")

    # col 5: earnings proximity (live equity only; 0 for training/non-equity)
    result[:, 5] = fetch_earnings_proximity(asset_sym, n, is_live=is_live)

    # cols 0-2 + col 4 from cached macro dataframe
    if not macro_df.empty:
        col_map = {
            0: ("vix",           0.25),
            1: ("rate",          0.15),
            2: ("regime",        0.0),
            4: ("peer_momentum", 0.0),
        }
        for col_idx, (col_name, default) in col_map.items():
            if col_name not in macro_df.columns:
                result[:, col_idx] = default
                continue
            series = macro_df[col_name]

            if hasattr(df.index, "date"):
                aligned = series.reindex(df.index, method="ffill")
            else:
                try:
                    df_dates = (pd.DatetimeIndex(df.iloc[:, 0])
                                if "Date" in df.columns else None)
                except Exception:
                    df_dates = None

                if df_dates is not None:
                    aligned = series.reindex(df_dates, method="ffill")
                else:
                    step = max(1, len(series) // n)
                    vals = series.values[::step][:n]
                    if len(vals) < n:
                        vals = np.pad(vals, (0, n - len(vals)), mode="edge")
                    aligned = pd.Series(vals[:n])

            filled = aligned.fillna(default)
            arr    = filled.values.astype(np.float32)
            result[:, col_idx] = (arr[:n] if len(arr) >= n
                                  else np.pad(arr, (0, n - len(arr)), mode="edge"))

    print(f"  ✅ Macro signals: shape={result.shape}  "
          f"VIX={result[:,0].mean():.3f}  "
          f"Rate={result[:,1].mean():.3f}  "
          f"Peer={result[:,4].mean():.3f}")
    return result
