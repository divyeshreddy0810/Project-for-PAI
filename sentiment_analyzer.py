#!/usr/bin/env python3
"""
Global Asset News Sentiment Analyzer
------------------------------------
A single-file program that lets users select assets, choose a time window,
fetches news headlines, displays them, runs FinBERT sentiment analysis,
computes daily and overall statistics, and saves structured output for other agents.
"""

import requests
from datetime import datetime, timedelta
import time
import sys
import os
import json
import csv
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

# FinBERT imports
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not installed. Sentiment analysis will be skipped.")
    print("    Install with: pip install transformers torch")

# ========================== CONFIGURATION ==========================
FINNHUB_KEY = "d6k6089r01qko8c3hrf0d6k6089r01qko8c3hrfg"
GNEWS_KEY = "74c1f79e81a5d2c62494062adaf232ca"

# Predefined assets (11 global assets)
ALL_ASSETS = [
    {"index": 1, "symbol": "^GSPC", "name": "S&P 500", "type": "index", "source": "finnhub_company", "keywords": ["S&P 500", "SPX"]},
    {"index": 2, "symbol": "^IXIC", "name": "NASDAQ Composite", "type": "index", "source": "finnhub_company", "keywords": ["NASDAQ", "Nasdaq"]},
    {"index": 3, "symbol": "^STOXX50E", "name": "Euro Stoxx 50", "type": "index", "source": "gnews_keywords", "keywords": ["Euro Stoxx 50", "STOXX 50"]},
    {"index": 4, "symbol": "^N225", "name": "Nikkei 225", "type": "index", "source": "gnews_keywords", "keywords": ["Nikkei 225", "Nikkei index"]},
    {"index": 5, "symbol": "BTC-USD", "name": "Bitcoin", "type": "crypto", "source": "finnhub_market_crypto", "keywords": ["bitcoin", "BTC"]},
    {"index": 6, "symbol": "ETH-USD", "name": "Ethereum", "type": "crypto", "source": "finnhub_market_crypto", "keywords": ["ethereum", "ETH"]},
    {"index": 7, "symbol": "SOL-USD", "name": "Solana", "type": "crypto", "source": "finnhub_market_crypto", "keywords": ["solana", "SOL"]},
    {"index": 8, "symbol": "EURUSD=X", "name": "EUR/USD", "type": "forex", "source": "finnhub_market_forex", "keywords": ["EUR/USD", "euro dollar"]},
    {"index": 9, "symbol": "GBPUSD=X", "name": "GBP/USD", "type": "forex", "source": "finnhub_market_forex", "keywords": ["GBP/USD", "pound dollar"]},
    {"index": 10, "symbol": "GC=F", "name": "Gold", "type": "commodity", "source": "gnews_keywords", "keywords": ["gold", "gold price"]},
    {"index": 11, "symbol": "CL=F", "name": "Crude Oil (WTI)", "type": "commodity", "source": "gnews_keywords", "keywords": ["crude oil", "WTI", "oil price"]},
]

# Time windows supported
WINDOW_OPTIONS = [
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"
]

# ========================== HELPER FUNCTIONS ==========================
def window_to_dates(window_str: str) -> Tuple[datetime, datetime]:
    """Convert window string (e.g., '1d', '1mo') to (from_date, to_date)."""
    now = datetime.now()
    if window_str.endswith('d'):
        days = int(window_str[:-1])
        return now - timedelta(days=days), now
    elif window_str.endswith('mo'):
        months = int(window_str[:-2])
        # approximate month as 30 days
        return now - timedelta(days=months*30), now
    elif window_str.endswith('y'):
        years = int(window_str[:-1])
        return now - timedelta(days=years*365), now
    else:
        raise ValueError(f"Unsupported window: {window_str}")

def parse_asset_selection(input_str: str) -> List[int]:
    """Parse user input like '1,3,5-7' into list of asset indices."""
    if input_str.strip().lower() == 'all':
        return [a['index'] for a in ALL_ASSETS]
    
    indices = set()
    parts = input_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            try:
                start_idx = int(start)
                end_idx = int(end)
                if start_idx <= end_idx:
                    indices.update(range(start_idx, end_idx+1))
                else:
                    indices.update(range(end_idx, start_idx+1))
            except ValueError:
                continue
        else:
            try:
                indices.add(int(part))
            except ValueError:
                continue
    # Validate indices
    valid_indices = [a['index'] for a in ALL_ASSETS]
    selected = [i for i in indices if i in valid_indices]
    return sorted(selected)

# ========================== FETCHER FUNCTIONS ==========================
def fetch_finnhub_company_news(symbol: str, from_date: str, to_date: str) -> List[Dict]:
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": symbol, "from": from_date, "to": to_date, "token": FINNHUB_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else []
    except Exception as e:
        print(f"  ❌ Finnhub company news error: {e}")
        return []

def fetch_finnhub_market_news(category: str) -> List[Dict]:
    url = "https://finnhub.io/api/v1/news"
    params = {"category": category, "token": FINNHUB_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else []
    except Exception as e:
        print(f"  ❌ Finnhub market news error: {e}")
        return []

def fetch_gnews_keywords(keywords: List[str], from_date: str, to_date: str, max_results: int = 100) -> List[Dict]:
    # Build query: keywords joined by OR, wrap phrases in quotes
    query_parts = []
    for kw in keywords:
        if " " in kw:
            query_parts.append(f'"{kw}"')
        else:
            query_parts.append(kw)
    query = " OR ".join(query_parts)
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "from": from_date,
        "to": to_date,
        "max": max_results,
        "apikey": GNEWS_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("articles", [])
    except Exception as e:
        print(f"  ❌ GNews error: {e}")
        return []

# ========================== FINBERT SENTIMENT ANALYZER ==========================
class FinBERTAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self.loaded = False

    def load(self):
        if not TRANSFORMERS_AVAILABLE:
            print("  ⚠️  Transformers not installed. Cannot load model.")
            return
        if self.loaded:
            return
        print("  📦 Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            print(f"  ✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("  ✅ Using CPU")
        self.model.eval()
        self.loaded = True

    def analyze(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        if not self.loaded:
            self.load()
        if not texts:
            return []
        results = []
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # FinBERT classes: 0=positive, 1=negative, 2=neutral
            for prob in probs:
                results.append({
                    "positive": float(prob[0]),
                    "negative": float(prob[1]),
                    "neutral": float(prob[2]),
                    "confidence": float(max(prob)),
                    "label": ["positive", "negative", "neutral"][prob.argmax()]
                })
        return results

# ========================== MAIN PROGRAM ==========================
def main():
    print("="*70)
    print("🌍 GLOBAL ASSET NEWS SENTIMENT ANALYZER")
    print("="*70)
    
    # Display asset list
    print("\n📊 AVAILABLE ASSETS:")
    for a in ALL_ASSETS:
        print(f"  {a['index']}. {a['name']} ({a['symbol']})")
    
    # Asset selection
    while True:
        inp = input("\n📝 Enter asset numbers (e.g., '1,3,5-7' or 'all'): ").strip()
        if not inp:
            print("Please enter something.")
            continue
        selected_indices = parse_asset_selection(inp)
        if not selected_indices:
            print("No valid assets selected. Try again.")
            continue
        selected_assets = [a for a in ALL_ASSETS if a['index'] in selected_indices]
        print("\n✅ Selected assets:")
        for a in selected_assets:
            print(f"   - {a['name']} ({a['symbol']})")
        break
    
    # Time window selection
    print("\n⏱️  TIME WINDOW OPTIONS:")
    for opt in WINDOW_OPTIONS:
        print(f"   {opt}")
    while True:
        window = input("\nEnter time window (default 1d): ").strip()
        if not window:
            window = "1d"
        if window in WINDOW_OPTIONS:
            break
        else:
            print("Invalid window. Choose from:", ", ".join(WINDOW_OPTIONS))
    
    from_date, to_date = window_to_dates(window)
    from_str = from_date.strftime("%Y-%m-%d")
    to_str = to_date.strftime("%Y-%m-%d")
    print(f"\n⏱️  Fetching news from {from_str} to {to_str}")
    
    # ========== FETCH HEADLINES ==========
    all_headlines = []  # list of dicts with asset info
    print("\n📡 FETCHING HEADLINES...")
    for asset in selected_assets:
        print(f"\n🔍 {asset['name']} ({asset['symbol']})...")
        headlines = []
        source = asset['source']
        if source == "finnhub_company":
            headlines = fetch_finnhub_company_news(asset['symbol'], from_str, to_str)
        elif source == "finnhub_market_crypto":
            headlines = fetch_finnhub_market_news("crypto")
        elif source == "finnhub_market_forex":
            headlines = fetch_finnhub_market_news("forex")
        elif source == "gnews_keywords":
            headlines = fetch_gnews_keywords(asset['keywords'], from_str, to_str, max_results=20)
        else:
            print(f"  ❌ Unknown source: {source}")
            continue
        
        # For market news, we may need to filter relevance using keywords
        if source.startswith("finnhub_market") and asset.get("keywords"):
            filtered = []
            for h in headlines:
                headline_text = h.get('headline', '').lower()
                if any(kw.lower() in headline_text for kw in asset['keywords']):
                    filtered.append(h)
            headlines = filtered
            print(f"  🔍 Filtered to {len(headlines)} relevant headlines.")
        
        print(f"  ✅ {len(headlines)} headlines fetched.")
        
        # Store with asset metadata
        for h in headlines:
            all_headlines.append({
                "asset": asset,
                "headline_obj": h
            })
        
        # Display headlines for this asset
        if headlines:
            print(f"\n📰 Headlines for {asset['name']}:")
            for idx, h in enumerate(headlines[:5], 1):  # show first 5
                if source.startswith('finnhub'):
                    headline = h.get('headline', 'N/A')
                    date = h.get('datetime', '')
                    if date:
                        try:
                            date = datetime.fromtimestamp(int(date)).strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            date = str(date)
                    source_name = h.get('source', 'N/A')
                else:  # GNews
                    headline = h.get('title', 'N/A')
                    date = h.get('publishedAt', '')[:19].replace('T', ' ')
                    source_name = h.get('source', {}).get('name', 'N/A')
                print(f"  [{idx}] {headline[:100]}...")
                print(f"       {date} - {source_name}")
            if len(headlines) > 5:
                print(f"       ... and {len(headlines)-5} more")
        else:
            print("  (No headlines to display)")
        
        # Small delay to respect rate limits
        time.sleep(0.5)
    
    if not all_headlines:
        print("\n❌ No headlines found for any selected asset. Exiting.")
        return
    
    # ========== SENTIMENT ANALYSIS ==========
    print("\n🧠 SENTIMENT ANALYSIS WITH FinBERT")
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers not installed. Cannot proceed.")
        return
    
    analyzer = FinBERTAnalyzer()
    analyzer.load()
    
    # Prepare texts
    texts = []
    headline_refs = []  # to map back
    for item in all_headlines:
        h = item['headline_obj']
        source_type = item['asset']['source']
        if source_type.startswith('finnhub'):
            text = h.get('headline', '')
        else:
            text = h.get('title', '')
        if text:
            texts.append(text)
            headline_refs.append(item)
    
    # Process in batches with progress
    print(f"\n⚙️  Analyzing {len(texts)} headlines...")
    batch_size = 16
    sentiment_results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_results = analyzer.analyze(batch_texts, batch_size=batch_size)
        sentiment_results.extend(batch_results)
        print(f"   Processed {min(i+batch_size, len(texts))}/{len(texts)} headlines")
    
    # Attach sentiment to headline objects
    for item, sent in zip(headline_refs, sentiment_results):
        item['sentiment'] = sent
    
    # ========== DISPLAY PER-HEADLINE SENTIMENT ==========
    print("\n" + "="*80)
    print("📰 HEADLINE-LEVEL SENTIMENT SCORES")
    print("="*80)
    for idx, item in enumerate(all_headlines, 1):
        asset_name = item['asset']['name']
        sent = item['sentiment']
        h = item['headline_obj']
        if item['asset']['source'].startswith('finnhub'):
            headline = h.get('headline', 'N/A')[:80]
        else:
            headline = h.get('title', 'N/A')[:80]
        print(f"\n[{idx}] {headline}...")
        print(f"      Positive: {sent['positive']:.3f} | Negative: {sent['negative']:.3f} | Neutral: {sent['neutral']:.3f}")
        print(f"      Label: {sent['label']} (confidence: {sent['confidence']:.3f})")
    
    # ========== AGGREGATION ==========
    print("\n📊 AGGREGATING RESULTS...")
    
    # Group by asset
    asset_groups = defaultdict(list)
    for item in all_headlines:
        asset_groups[item['asset']['symbol']].append(item)
    
    aggregated_results = []
    for asset in selected_assets:
        symbol = asset['symbol']
        items = asset_groups.get(symbol, [])
        if not items:
            continue
        
        # Group by day (using published date)
        day_groups = defaultdict(list)
        for item in items:
            h = item['headline_obj']
            if asset['source'].startswith('finnhub'):
                ts = h.get('datetime')
                if ts:
                    try:
                        dt = datetime.fromtimestamp(int(ts))
                    except:
                        dt = None
                else:
                    dt = None
            else:
                date_str = h.get('publishedAt', '')
                if date_str:
                    try:
                        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except:
                        dt = None
                else:
                    dt = None
            if dt:
                day_key = dt.date()
                day_groups[day_key].append(item['sentiment']['positive'])
        
        # Compute daily means
        daily_means = []
        for day, scores in sorted(day_groups.items()):
            mean_score = sum(scores) / len(scores)
            daily_means.append({
                "date": day.isoformat(),
                "mean_sentiment": mean_score,
                "headline_count": len(scores)
            })
        
        # Overall stats across days
        if len(daily_means) > 1:
            means = [d['mean_sentiment'] for d in daily_means]
            overall_mean = sum(means) / len(means)
            overall_median = float(np.median(means))
            overall_std = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
        elif len(daily_means) == 1:
            overall_mean = daily_means[0]['mean_sentiment']
            overall_median = None
            overall_std = None
        else:
            # No daily data (should not happen because we have items)
            continue
        
        # Top headlines (by confidence)
        top_headlines = []
        for item in sorted(items, key=lambda x: x['sentiment']['confidence'], reverse=True)[:3]:
            h = item['headline_obj']
            if asset['source'].startswith('finnhub'):
                headline = h.get('headline', '')
                date = h.get('datetime', '')
                if date:
                    try:
                        date = datetime.fromtimestamp(int(date)).isoformat()
                    except:
                        date = str(date)
                url = h.get('url', '')
            else:
                headline = h.get('title', '')
                date = h.get('publishedAt', '')
                url = h.get('url', '')
            top_headlines.append({
                "headline": headline,
                "sentiment": item['sentiment']['positive'],
                "confidence": item['sentiment']['confidence'],
                "date": date,
                "url": url
            })
        
        aggregated_results.append({
            "symbol": symbol,
            "name": asset['name'],
            "total_headlines": len(items),
            "days_count": len(daily_means),
            "daily_means": daily_means,
            "overall_mean": overall_mean,
            "overall_median": overall_median,
            "overall_std": overall_std,
            "top_headlines": top_headlines
        })
    
    # ========== DISPLAY RESULTS ==========
    print("\n" + "="*80)
    print("📊 FINAL SENTIMENT RESULTS")
    print("="*80)
    
    # Table header
    print(f"\n{'Asset':<25} {'Days':<6} {'Headlines':<10} {'Mean':<8} {'Median':<8} {'Std':<8} {'Signal':<6}")
    print("-"*80)
    
    for res in aggregated_results:
        # Determine signal based on overall_mean
        mean = res['overall_mean']
        if mean is not None:
            if mean > 0.6:
                signal = "🟢 BUY"
            elif mean < 0.4:
                signal = "🔴 SELL"
            else:
                signal = "🟡 HOLD"
        else:
            signal = "N/A"
        
        median_str = f"{res['overall_median']:.3f}" if res['overall_median'] is not None else "N/A"
        std_str = f"{res['overall_std']:.3f}" if res['overall_std'] is not None else "N/A"
        mean_str = f"{res['overall_mean']:.3f}" if res['overall_mean'] is not None else "N/A"
        
        print(f"{res['name'][:24]:<25} {res['days_count']:<6} {res['total_headlines']:<10} {mean_str:<8} {median_str:<8} {std_str:<8} {signal:<6}")
    
    print("\n📘 EXPLANATION:")
    print("  - Headline-level scores (positive probability) are shown above.")
    print("  - Daily Mean: average sentiment for each day (positive probability).")
    print("  - Overall Mean: average of daily means across the selected period.")
    print("  - Median: middle value of daily means (less sensitive to extremes).")
    print("  - Std: standard deviation of daily means (high = sentiment varies a lot).")
    print("  - Signal: BUY if mean > 0.6, SELL if mean < 0.4, else HOLD.")
    print(f"  - For this run, overall mean = {aggregated_results[0]['overall_mean']:.3f} (from {aggregated_results[0]['total_headlines']} headlines).")
    
# ========== SAVE OUTPUT ==========
    print("\n💾 SAVING RESULTS...")
    
    # Create output directory
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(output_dir, f"sentiment_{timestamp}.json")
    csv_file = os.path.join(output_dir, f"sentiment_{timestamp}.csv")
    latest_file = os.path.join(output_dir, "latest.json")
    
    # Prepare JSON data (include per-headline details)
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "window": window,
        "from_date": from_str,
        "to_date": to_str,
        "assets": aggregated_results,
        "headlines": [
            {
                "asset": item['asset']['name'],
                "headline": (item['headline_obj'].get('headline') or item['headline_obj'].get('title', '')),
                "positive": item['sentiment']['positive'],
                "negative": item['sentiment']['negative'],
                "neutral": item['sentiment']['neutral'],
                "confidence": item['sentiment']['confidence'],
                "label": item['sentiment']['label']
            }
            for item in all_headlines if 'sentiment' in item
        ]
    }
    
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"  ✅ JSON saved: {json_file}")
    
    # Write CSV (summary only)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Asset", "Symbol", "Days", "Headlines", "Mean", "Median", "Std", "Signal"])
        for res in aggregated_results:
            signal = "BUY" if res['overall_mean'] and res['overall_mean'] > 0.6 else "SELL" if res['overall_mean'] and res['overall_mean'] < 0.4 else "HOLD"
            writer.writerow([
                res['name'],
                res['symbol'],
                res['days_count'],
                res['total_headlines'],
                f"{res['overall_mean']:.3f}" if res['overall_mean'] else "N/A",
                f"{res['overall_median']:.3f}" if res['overall_median'] else "N/A",
                f"{res['overall_std']:.3f}" if res['overall_std'] else "N/A",
                signal
            ])
    print(f"  ✅ CSV saved: {csv_file}")
    
    # Save as latest.json (overwrite if exists)
    with open(latest_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"  🔗 Latest saved: {latest_file}")
    
    print("\n✅ All done! Results ready for other agents.")
    print("   Files are in 'data/output/' directory.")
    print("   Headline-level sentiment is also saved in the JSON.")

if __name__ == "__main__":
    main()
