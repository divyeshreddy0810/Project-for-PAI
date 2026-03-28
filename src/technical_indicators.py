#!/usr/bin/env python3
"""
Stock Regime Predictor with Technical Indicators & Sentiment
------------------------------------------------------------
Combines sentiment analysis output with technical indicators
to predict stock market regimes (BULL, BEAR, SIDEWAYS).
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
import csv
import argparse

# Try to import yfinance for real market data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not installed. Install with: pip install yfinance")


# ========================== CONFIGURATION ==========================
SENTIMENT_FILE = "data/output/latest.json"
OUTPUT_DIR = "data/output"

# Technical indicator thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_NEUTRAL_LOWER = 40
RSI_NEUTRAL_UPPER = 60

# Regime scoring parameters
SCORE_WEIGHTS = {
    'price_trend': 1.5,      # Price above/below SMAs
    'rsi': 1.0,              # RSI strength
    'macd': 1.0,             # MACD momentum
    'sentiment_level': 1.5,  # Overall sentiment value
    'sentiment_trend': 1.0,  # Sentiment direction
    'headline_volume': 0.5   # Headline engagement
}


class TechnicalIndicatorCalculator:
    """Calculate technical indicators from price data
    
    Supports daily and intraday data (hourly, 15-min, 5-min) with automatic
    period adjustment for consistent signal behavior across all timeframes.
    """
    
    @staticmethod
    def detect_frequency(df: pd.DataFrame) -> str:
        """Detect data frequency (daily, hourly, 15-min, 5-min)"""
        if len(df) < 2:
            return 'unknown'
        
        # Get time deltas between first few bars
        time_diffs = df.index.to_series().diff().head(10)
        median_diff = time_diffs.median()
        
        # Convert to hours for comparison
        if hasattr(median_diff, 'total_seconds'):
            hours = median_diff.total_seconds() / 3600
        else:
            hours = median_diff / pd.Timedelta(hours=1)
        
        if hours >= 20:  # Daily or more
            return 'daily'
        elif 0.9 <= hours <= 1.1:  # Hourly
            return 'hourly'
        elif 0.2 <= hours <= 0.3:  # 15-min
            return '15min'
        elif 0.05 <= hours <= 0.1:  # 5-min
            return '5min'
        else:
            return 'unknown'
    
    @staticmethod
    def get_adjusted_periods(frequency: str) -> Dict[str, int]:
        """Get indicator periods adjusted for data frequency
        
        Ensures that technical indicators produce consistent signals
        regardless of whether data is daily, hourly, or intraday.
        """
        periods = {
            'daily': {'sma5': 5, 'sma20': 20, 'sma50': 50, 'rsi': 14, 'macd_fast': 12, 'macd_slow': 26, 'bb': 20},
            'hourly': {'sma5': 5, 'sma20': 20, 'sma50': 50, 'rsi': 14, 'macd_fast': 12, 'macd_slow': 26, 'bb': 20},
            '15min': {'sma5': 4, 'sma20': 16, 'sma50': 40, 'rsi': 14, 'macd_fast': 12, 'macd_slow': 24, 'bb': 16},
            '5min': {'sma5': 6, 'sma20': 12, 'sma50': 30, 'rsi': 14, 'macd_fast': 10, 'macd_slow': 20, 'bb': 12},
        }
        return periods.get(frequency, periods['daily'])
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (0-100)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                                  num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 14) -> pd.Series:
        """Average True Range (volatility)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr


class SentimentAnalyzer:
    """Load and extract features from sentiment analysis output"""
    
    def __init__(self, filepath: str):
        """Load sentiment data from JSON"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Sentiment file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            self.data = json.load(f)
    
    def get_asset_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Extract sentiment metrics for a specific symbol"""
        for asset in self.data.get('assets', []):
            if asset['symbol'] == symbol:
                return {
                    'symbol': symbol,
                    'name': asset['name'],
                    'overall_mean': asset['overall_mean'],
                    'overall_median': asset['overall_median'],
                    'overall_std': asset['overall_std'],
                    'total_headlines': asset['total_headlines'],
                    'days_count': asset['days_count'],
                    'daily_means': asset['daily_means'],
                    'top_headlines': asset.get('top_headlines', []),
                    'sentiment_trend': self._calculate_trend(asset['daily_means'])
                }
        return None
    
    @staticmethod
    def _calculate_trend(daily_means: List[Dict]) -> float:
        """Calculate sentiment trend (slope over time)"""
        if len(daily_means) < 2:
            return 0.0
        
        sentiments = [d['mean_sentiment'] for d in daily_means]
        # Simple linear regression slope
        x = np.arange(len(sentiments))
        slope = np.polyfit(x, sentiments, 1)[0]
        return slope
    
    def get_analysis_metadata(self) -> Dict[str, Any]:
        """Get metadata about the analysis"""
        return {
            'timestamp': self.data.get('timestamp'),
            'window': self.data.get('window'),
            'from_date': self.data.get('from_date'),
            'to_date': self.data.get('to_date')
        }


class RegimePredictor:
    """Predict stock regime using sentiment + technical indicators"""
    
    def __init__(self, sentiment_file: str):
        """Initialize with sentiment analysis output"""
        self.sentiment = SentimentAnalyzer(sentiment_file)
        self.calculator = TechnicalIndicatorCalculator()
        self.metadata = self.sentiment.get_analysis_metadata()
    
    def fetch_price_data(self, symbol: str, period: str = '1mo') -> Optional[pd.DataFrame]:
        """Fetch historical price data from yfinance"""
        if not YFINANCE_AVAILABLE:
            print(f"❌ yfinance not available. Cannot fetch price data for {symbol}")
            return None
        
        try:
            # Convert sentiment window to yfinance period
            period_map = {
                '1d': '5d', '5d': '1mo', '1mo': '1mo', '3mo': '3mo',
                '6mo': '6mo', '1y': '1y', '2y': '2y', '5y': '5y', '10y': '10y'
            }
            yf_period = period_map.get(period, '1mo')
            
            print(f"   📊 Fetching price data for {symbol}...")
            df = yf.download(symbol, period=yf_period, progress=False)
            
            if df.empty:
                print(f"   ⚠️  No price data available for {symbol}")
                return None
            
            return df
        except Exception as e:
            print(f"   ❌ Error fetching data: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = self.calculator.calculate_sma(df['Close'], 20)
        indicators['sma_50'] = self.calculator.calculate_sma(df['Close'], 50)
        indicators['sma_200'] = self.calculator.calculate_sma(df['Close'], 200)
        
        # Momentum indicators
        indicators['rsi'] = self.calculator.calculate_rsi(df['Close'])
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = \
            self.calculator.calculate_macd(df['Close'])
        
        # Volatility
        indicators['upper_bb'], indicators['middle_bb'], indicators['lower_bb'] = \
            self.calculator.calculate_bollinger_bands(df['Close'])
        indicators['atr'] = self.calculator.calculate_atr(df['High'], df['Low'], df['Close'])
        
        return indicators
    
    def score_regime(self, sentiment: Dict, indicators: Dict, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate regime score combining sentiment and technical indicators"""
        scores = {}
        total_score = 0
        
        # Convert series to latest values (scalar floats)
        latest_idx = -1
        
        # Helper function to safely extract float values
        def safe_float(value):
            try:
                # Use .item() method for pandas/numpy scalar conversion
                if hasattr(value, 'item'):
                    v = value.item()
                else:
                    v = float(value)
                return v if not np.isnan(v) else None
            except (ValueError, TypeError):
                return None
        
        # 1. Price trend scoring
        close = safe_float(price_data['Close'].iloc[latest_idx])
        if close is None:
            close = float(price_data['Close'].iloc[latest_idx].item())
        
        sma_20 = safe_float(indicators['sma_20'].iloc[latest_idx])
        sma_50 = safe_float(indicators['sma_50'].iloc[latest_idx])
        sma_200 = safe_float(indicators['sma_200'].iloc[latest_idx])
        
        trend_score = 0
        if sma_20 is not None and sma_50 is not None and sma_200 is not None:
            if close > sma_20 and sma_20 > sma_50:
                trend_score = 3.0  # Strong uptrend
            elif close > sma_50 and sma_50 > sma_200:
                trend_score = 2.0  # Medium uptrend
            elif close > sma_20:
                trend_score = 1.0  # Weak uptrend
            elif close < sma_20 and sma_20 < sma_50:
                trend_score = -3.0  # Strong downtrend
            elif close < sma_50 and sma_50 < sma_200:
                trend_score = -2.0  # Medium downtrend
            elif close < sma_20:
                trend_score = -1.0  # Weak downtrend
        elif sma_20 is not None:
            if close > sma_20:
                trend_score = 1.0
            elif close < sma_20:
                trend_score = -1.0
        
        scores['price_trend'] = trend_score
        total_score += trend_score * SCORE_WEIGHTS['price_trend']
        
        # 2. RSI scoring
        rsi = safe_float(indicators['rsi'].iloc[latest_idx])
        
        if rsi is None:
            rsi_score = 0
        elif rsi > RSI_OVERBOUGHT:
            rsi_score = -2.0  # Overbought, bearish
        elif rsi > RSI_NEUTRAL_UPPER:
            rsi_score = 1.0   # Strong, bullish
        elif rsi > RSI_NEUTRAL_LOWER:
            rsi_score = 0.5   # Neutral
        elif rsi > RSI_OVERSOLD:
            rsi_score = -0.5  # Weak
        else:
            rsi_score = 2.0   # Oversold, potential bounce
        
        scores['rsi'] = rsi_score
        total_score += rsi_score * SCORE_WEIGHTS['rsi']
        
        # 3. MACD scoring
        macd = safe_float(indicators['macd'].iloc[latest_idx])
        signal = safe_float(indicators['macd_signal'].iloc[latest_idx])
        
        if macd is None or signal is None:
            macd_score = 0
        elif macd > signal and macd > 0:
            macd_score = 2.0   # Strong bullish
        elif macd > signal:
            macd_score = 1.0   # Bullish crossover
        elif macd < signal and macd < 0:
            macd_score = -2.0  # Strong bearish
        else:
            macd_score = -1.0  # Bearish crossover
        
        scores['macd'] = macd_score
        total_score += macd_score * SCORE_WEIGHTS['macd']
        
        # 4. Sentiment level scoring
        sentiment_level = sentiment['overall_mean']
        if sentiment_level > 0.15:
            sentiment_score = 2.0
        elif sentiment_level > 0.05:
            sentiment_score = 1.0
        elif sentiment_level > -0.05:
            sentiment_score = 0.0
        elif sentiment_level > -0.15:
            sentiment_score = -1.0
        else:
            sentiment_score = -2.0
        
        scores['sentiment_level'] = sentiment_score
        total_score += sentiment_score * SCORE_WEIGHTS['sentiment_level']
        
        # 5. Sentiment trend scoring
        sentiment_trend = sentiment['sentiment_trend']
        if sentiment_trend > 0.05:
            trend_score = 1.5   # Improving
        elif sentiment_trend > 0:
            trend_score = 0.5   # Slightly improving
        elif sentiment_trend > -0.05:
            trend_score = 0.0   # Stable
        else:
            trend_score = -1.0  # Worsening
        
        scores['sentiment_trend'] = trend_score
        total_score += trend_score * SCORE_WEIGHTS['sentiment_trend']
        
        # 6. Headline volume scoring (engagement)
        headline_count = sentiment['total_headlines']
        if headline_count > 50:
            volume_score = 1.0  # High engagement
        elif headline_count > 20:
            volume_score = 0.5
        else:
            volume_score = -0.5  # Low engagement
        
        scores['headline_volume'] = volume_score
        total_score += volume_score * SCORE_WEIGHTS['headline_volume']
        
        return {
            'total_score': total_score,
            'component_scores': scores,
            'rsi': rsi if rsi is not None else 0.0,
            'macd': macd if macd is not None else 0.0,
            'macd_signal': signal if signal is not None else 0.0,
            'sma_20': sma_20 if sma_20 is not None else close,
            'sma_50': sma_50 if sma_50 is not None else close,
            'sma_200': sma_200 if sma_200 is not None else close,
            'close': close
        }
    
    def determine_regime(self, total_score: float) -> Tuple[str, float]:
        """Determine regime and confidence from score"""
        # Score range: typically -10 to +10
        normalized_score = (total_score + 10) / 20  # 0 to 1
        confidence = min(abs(total_score) / 10, 0.99)
        
        if total_score >= 3.0:
            return "🔼 BULL", confidence
        elif total_score <= -3.0:
            return "🔽 BEAR", confidence
        else:
            return "↔️  SIDEWAYS", confidence
    
    def predict(self, symbol: str, period: Optional[str] = None) -> Dict[str, Any]:
        """Full prediction pipeline for a symbol"""
        print(f"\n{'='*60}")
        print(f"🎯 Analyzing {symbol}")
        print(f"{'='*60}")
        
        # Get sentiment
        sentiment = self.sentiment.get_asset_sentiment(symbol)
        if not sentiment:
            print(f"❌ No sentiment data found for {symbol}")
            return None
        
        print(f"✓ Sentiment loaded (Mean: {sentiment['overall_mean']:.3f})")
        
        # Use metadata window if not specified
        if period is None:
            period = self.metadata.get('window', '1mo')
        
        # Fetch price data
        price_data = self.fetch_price_data(symbol, period)
        if price_data is None or price_data.empty:
            print(f"⚠️  Using sentiment-only prediction (limited accuracy)")
            regime, confidence = self.determine_regime(sentiment['overall_mean'] * 5)
            return {
                'symbol': symbol,
                'name': sentiment['name'],
                'regime': regime,
                'confidence': confidence,
                'sentiment': sentiment,
                'technical': None,
                'score': sentiment['overall_mean'] * 5,
                'metadata': self.metadata
            }
        
        # Calculate technical indicators
        print(f"✓ Price data loaded ({len(price_data)} days)")
        indicators = self.calculate_indicators(price_data)
        print(f"✓ Technical indicators calculated")
        
        # Score and predict
        score_result = self.score_regime(sentiment, indicators, price_data)
        regime, confidence = self.determine_regime(score_result['total_score'])
        
        return {
            'symbol': symbol,
            'name': sentiment['name'],
            'regime': regime,
            'confidence': confidence,
            'score': score_result['total_score'],
            'sentiment': sentiment,
            'technical': {
                'close': score_result['close'],
                'sma_20': score_result['sma_20'],
                'sma_50': score_result['sma_50'],
                'sma_200': score_result['sma_200'],
                'rsi': score_result['rsi'],
                'macd': score_result['macd'],
                'macd_signal': score_result['macd_signal']
            },
            'component_scores': score_result['component_scores'],
            'metadata': self.metadata
        }
    
    def predict_multiple(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """Predict regime for multiple symbols"""
        if symbols is None:
            # Predict all symbols in sentiment data
            symbols = [a['symbol'] for a in self.sentiment.data.get('assets', [])]
        
        results = []
        for symbol in symbols:
            result = self.predict(symbol)
            if result:
                results.append(result)
        
        return results


def format_result(result: Dict) -> str:
    """Format prediction result for display"""
    if not result:
        return ""
    
    def safe_format(value, fmt='f', precision=3):
        """Safely format numeric values that might be None"""
        if value is None:
            return 'N/A'
        try:
            if fmt == 'f':
                return f"{float(value):.{precision}f}"
            elif fmt == '%':
                return f"{float(value):.1%}"
            elif fmt == 'price':
                return f"${float(value):.2f}"
            else:
                return str(value)
        except (TypeError, ValueError):
            return 'N/A'
    
    output = f"\n{'='*60}\n"
    output += f"Symbol: {result['symbol']} ({result['name']})\n"
    output += f"Regime: {result['regime']}\n"
    output += f"Confidence: {safe_format(result.get('confidence', 0), '%')}\n"
    output += f"Score: {safe_format(result.get('score', 0), 'f', 2)}/10\n"
    output += f"\n📊 SENTIMENT FACTORS:\n"
    
    sentiment = result.get('sentiment', {})
    output += f"  • Mean Sentiment: {safe_format(sentiment.get('overall_mean'), 'f', 3)}\n"
    output += f"  • Median: {safe_format(sentiment.get('overall_median'), 'f', 3)}\n"
    output += f"  • Std Dev: {safe_format(sentiment.get('overall_std'), 'f', 3)}\n"
    output += f"  • Headlines: {sentiment.get('total_headlines', 0)}\n"
    output += f"  • Trend: {safe_format(sentiment.get('sentiment_trend', 0), 'f', 4)}\n"
    
    if result.get('technical'):
        output += f"\n📈 TECHNICAL FACTORS:\n"
        tech = result['technical']
        output += f"  • Price: {safe_format(tech.get('close'), 'price')}\n"
        output += f"  • SMA 20: {safe_format(tech.get('sma_20'), 'price')}\n"
        output += f"  • SMA 50: {safe_format(tech.get('sma_50'), 'price')}\n"
        output += f"  • RSI (14): {safe_format(tech.get('rsi'), 'f', 1)}\n"
        output += f"  • MACD: {safe_format(tech.get('macd'), 'f', 4)}\n"
        output += f"  • MACD Signal: {safe_format(tech.get('macd_signal'), 'f', 4)}\n"
    
    output += f"\n🎯 COMPONENT SCORES:\n"
    for component, score in result.get('component_scores', {}).items():
        score_val = score if score is not None else 0
        bar = "█" * int(abs(score_val)) + ("+" if score_val > 0 else "-" if score_val < 0 else "")
        output += f"  • {component:20s}: {safe_format(score, 'f', 1)} {bar}\n"
    
    metadata = result.get('metadata', {})
    output += f"\n📅 Analysis Period: {metadata.get('from_date', 'N/A')} to {metadata.get('to_date', 'N/A')}\n"
    output += f"{'='*60}\n"
    
    return output


def save_results_csv(results: List[Dict], filename: str):
    """Save results to CSV file"""
    with open(filename, 'w', newline='') as f:
        fieldnames = ['Symbol', 'Name', 'Regime', 'Confidence', 'Score', 
                     'Sentiment_Mean', 'Sentiment_Trend', 'RSI', 'MACD', 'Headlines']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'Symbol': result['symbol'],
                'Name': result['name'],
                'Regime': result['regime'],
                'Confidence': f"{result['confidence']:.1%}",
                'Score': f"{result['score']:.2f}",
                'Sentiment_Mean': f"{result['sentiment']['overall_mean']:.3f}",
                'Sentiment_Trend': f"{result['sentiment']['sentiment_trend']:+.4f}",
                'RSI': f"{result['technical']['rsi']:.1f}" if result['technical'] else "N/A",
                'MACD': f"{result['technical']['macd']:.4f}" if result['technical'] else "N/A",
                'Headlines': result['sentiment']['total_headlines']
            })


def main():
    """Main execution"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze technical indicators and market regime")
    parser.add_argument('--symbols', type=str, default=None, help='Comma-separated symbols (e.g., ^GSPC,BTC-USD)')
    parser.add_argument('--window', type=str, default='1mo', help='Time window for analysis')
    parser.add_argument('--non-interactive', action='store_true', help='Run without interactive prompts')
    args = parser.parse_args()
    
    print("\n🚀 Stock Regime Predictor with Technical Indicators")
    print("====================================================\n")
    
    # Load sentiment data
    try:
        predictor = RegimePredictor(SENTIMENT_FILE)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"   Make sure sentiment_analyzer.py has been run first.")
        sys.exit(1)
    
    available_symbols = [a['symbol'] for a in predictor.sentiment.data.get('assets', [])]
    
    # Determine selected symbols
    if args.symbols and args.non_interactive:
        # CLI mode: use provided symbols
        print(f"📍 Using CLI arguments: symbols={args.symbols}, window={args.window}")
        selected_symbols = [s.strip() for s in args.symbols.split(',')]
        
        # Filter to only available symbols (those in sentiment data)
        selected_symbols = [s for s in selected_symbols if s in available_symbols]
        
        if not selected_symbols:
            print(f"⚠️  No valid symbols provided. Using all available: {', '.join(available_symbols)}")
            selected_symbols = available_symbols
    else:
        # Interactive mode
        print("Available symbols from sentiment analysis:")
        symbol_map = {}  # Map index to symbol
        for i, sym in enumerate(available_symbols, 1):
            names = [a['name'] for a in predictor.sentiment.data['assets'] if a['symbol'] == sym]
            print(f"  {i}. {sym} - {names[0] if names else 'Unknown'}")
            symbol_map[str(i)] = sym
        
        print("\nOptions:")
        print("  • Enter numbers: 1,2,3 or 1-3 (for index range)")
        print("  • Enter symbols: ^GSPC,^IXIC,BTC-USD")
        print("  • Type 'all' for all symbols")
        print("  • Type 'q' to quit")
        
        user_input = input("\nEnter symbols or indices: ").strip()
        
        if user_input.lower() == 'q':
            print("Exiting...")
            sys.exit(0)
        
        if user_input.lower() == 'all':
            selected_symbols = available_symbols
        else:
            # Parse input: could be numbers (1,2,3 or 1-3) or symbols
            selected_symbols = []
            parts = user_input.split(',')
            
            for part in parts:
                part = part.strip()
                
                # Check if it's a range like "1-3"
                if '-' in part:
                    try:
                        start, end = part.split('-')
                        start_idx = int(start.strip())
                        end_idx = int(end.strip())
                        for i in range(start_idx, end_idx + 1):
                            if str(i) in symbol_map:
                                selected_symbols.append(symbol_map[str(i)])
                    except ValueError:
                        # Not a number range, treat as symbol
                        selected_symbols.append(part)
                # Check if it's a single number
                elif part.isdigit():
                    if part in symbol_map:
                        selected_symbols.append(symbol_map[part])
                    else:
                        print(f"⚠️  Invalid index: {part}")
                # Otherwise treat as symbol
                else:
                    selected_symbols.append(part)
    
    # Run predictions
    results = []
    for symbol in selected_symbols:
        result = predictor.predict(symbol)
        if result:
            results.append(result)
            print(format_result(result))
    
    # Save to CSV
    if results:
        csv_file = os.path.join(OUTPUT_DIR, 
                               f"regime_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        save_results_csv(results, csv_file)
        print(f"\n✓ Results saved to {csv_file}")


if __name__ == "__main__":
    main()
