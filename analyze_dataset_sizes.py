#!/usr/bin/env python3
"""Analyze dataset sizes and attributes used in the project"""

import json
import os
import pandas as pd

def analyze_datasets():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 70)
    print("DATASET SIZES AND ATTRIBUTES ANALYSIS")
    print("=" * 70)
    
    # 1. SENTIMENT ANALYSIS INPUT
    print("\n1. SENTIMENT ANALYSIS INPUT")
    print("-" * 70)
    
    with open('data/output/latest.json') as f:
        sentiment = json.load(f)
    
    num_assets = len(sentiment['assets'])
    total_headlines = sentiment['headlines']['total_count']
    
    print(f"   Assets Analyzed: {num_assets}")
    print(f"   Total Headlines Collected: {total_headlines}")
    print(f"   Analysis Window: {sentiment['window']}")
    print(f"   Date Range: {sentiment['from_date']} to {sentiment['to_date']}")
    
    if sentiment['assets']:
        asset = sentiment['assets'][0]
        print(f"\n   Example Asset: {asset['symbol']} ({asset['name']})")
        print(f"   - Headlines per asset: {asset['total_headlines']}")
        print(f"   - Days covered: {asset['days_count']}")
        print(f"   - Sentiment Attributes: {3}")  # mean, median, std
        print(f"   - Daily sentiment records: {len(asset['daily_means'])}")
    
    print(f"\n   Per-Asset Attributes Generated:")
    print(f"   - overall_mean, overall_median, overall_std (3)")
    print(f"   - total_headlines, days_count (2)")
    print(f"   - daily_means: list of {len(asset['daily_means'])} daily records")
    print(f"   - top_headlines: list of top articles")
    print(f"\n   Total Rows: {num_assets}")
    print(f"   Total Attributes: 5+ per asset + N daily records")
    
    # 2. PRICE DATA INPUT
    print("\n\n2. PRICE DATA INPUT (from yfinance)")
    print("-" * 70)
    print(f"   Assets: {num_assets} (same as sentiment analysis)")
    print(f"   Data Collection Period: 3-6 months (typical)")
    print(f"   Minimum Data Points: 20 daily records required")
    print(f"   Maximum Data Points: ~252 trading days per year")
    print(f"\n   OHLCV Attributes per asset:")
    print(f"   - Open, High, Low, Close, Volume (5 attributes)")
    print(f"   - Frequency: Daily")
    print(f"\n   Typical Size:")
    print(f"   - 3 months: ~63 trading days × 5 attributes")
    print(f"   - 6 months: ~126 trading days × 5 attributes")
    print(f"   - 1 year: ~252 trading days × 5 attributes")
    
    # 3. TECHNICAL INDICATORS OUTPUT
    print("\n\n3. TECHNICAL INDICATORS OUTPUT")
    print("-" * 70)
    print(f"   Technical Features Calculated:")
    features = [
        "Close", "Open", "High", "Low", "Volume",  # 5 OHLCV
        "Returns", "Price_MA5", "Price_MA20", "Price_MA50",  # 4
        "Volatility", "RSI", "MACD", "Signal",  # 4
        "Volume_MA", "Volume_Trend",  # 2
        "High_Low_Range", "Open_Close_Range"  # 2
    ]
    print(f"   Total Features: {len(features)}")
    for i, feat in enumerate(features, 1):
        print(f"   {i:2d}. {feat}")
    
    # 4. MARKET REGIME OUTPUT
    print("\n\n4. MARKET REGIME CLASSIFICATION OUTPUT")
    print("-" * 70)
    regime_files = [f for f in os.listdir('data/output') 
                    if 'market_regime' in f and f.endswith('.json')]
    if regime_files:
        latest = sorted(regime_files)[-1]
        with open(f'data/output/{latest}') as f:
            regime = json.load(f)
        
        print(f"   Output File: {latest}")
        if 'assets' in regime and regime['assets']:
            asset = regime['assets'][0]
            attrs = list(asset.keys())
            print(f"   Attributes per Asset: {len(attrs)}")
            print(f"   {attrs}")
            print(f"   Total Assets in Output: {len(regime['assets'])}")
    
    # 5. PRICE FORECAST OUTPUT
    print("\n\n5. PRICE FORECASTING OUTPUT")
    print("-" * 70)
    print(f"   Forecast Horizons: 5, 10, 15, 20 days")
    print(f"   Prediction Features:")
    print(f"   - Symbol, Current Price (2)")
    print(f"   - Day5_Forecast, Day10_Forecast, Day15_Forecast, Day20_Forecast (4)")
    print(f"   - Confidence scores for each prediction (4)")
    print(f"   - Model type used (1)")
    print(f"   Total: ~11-12 attributes per asset")
    
    # 6. SUMMARY TABLE
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLE: DATASET INPUTS")
    print("=" * 70)
    
    summary_data = {
        'Dataset': [
            'Sentiment (Headlines)',
            'Price Data (OHLCV)',
            'Technical Indicators',
            'Market Regime',
            'Price Forecasts'
        ],
        'Input Rows/Records': [
            f'{num_assets} assets × avg. {int(total_headlines/num_assets)} headlines/asset',
            f'{num_assets} assets × ~126 days',
            f'{num_assets} assets × ~126 days',
            f'{num_assets} assets (aggregated)',
            f'{num_assets} assets'
        ],
        'Attributes': [
            '5-7 per asset + daily breakdown',
            '5 (OHLCV)',
            '18 features',
            '8-12 per asset',
            '11-12 per asset'
        ],
        'Total Size': [
            f'~{total_headlines} headlines + metadata',
            f'~{num_assets * 126} rows (6mo) × 5 cols',
            f'~{num_assets * 126} rows × 18 cols',
            f'{num_assets} assets × 10-12 attrs',
            f'{num_assets} assets × 12 attrs'
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    analyze_datasets()
