#!/usr/bin/env python3
"""
Market Regime Model
-------------------
Classifies market regime as BULL, BEAR, or VOLATILE based on:
- Sentiment analysis (from sentiment_analyzer.py output)
- Technical indicators (from technical_indicators.py output)
- Historical price volatility

Output: Regime classification with confidence scores for each regime type.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import csv
from typing import Dict, List, Any, Tuple, Optional

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ========================== CONFIGURATION ==========================
REGIME_THRESHOLDS = {
    'bull_threshold': 0.5,      # Bull probability > 50%
    'bear_threshold': 0.5,      # Bear probability > 50%
    'volatility_threshold': 0.4 # Volatility > 40%
}

OUTPUT_DIR = "data/output"


class RegimeAnalyzer:
    """Advanced market regime classification using multiple factors"""
    
    def __init__(self):
        """Initialize regime analyzer"""
        self.regime_history = []
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, period: int = 14) -> float:
        """Calculate historical volatility (standard deviation of returns)"""
        vals = prices.values.flatten().astype(float)
        returns = np.diff(vals) / vals[:-1]
        vol = float(np.nanstd(returns) * np.sqrt(252))  # Annualized
        return vol if not np.isnan(vol) else 0.0
    
    @staticmethod
    def calculate_trend_strength(prices: pd.Series, period: int = 20) -> float:
        """Calculate how strong the current trend is (0 to 1)"""
        if len(prices) < period:
            return 0.5
        
        # Calculate linear regression slope
        x = np.arange(len(prices[-period:]))
        y = prices[-period:].values.flatten()
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize to 0-1 range
        price_range = float(np.max(y) - np.min(y))
        if price_range == 0:
            return 0.5
        
        normalized_slope = float(slope * period) / price_range
        return float(np.clip(np.tanh(normalized_slope), 0, 1))
    
    @staticmethod
    def calculate_mean_reversion_score(prices: pd.Series, period: int = 20) -> float:
        """Calculate mean reversion potential (how far price is from average)"""
        if len(prices) < period:
            return 0.5
        
        vals = prices[-period:].values.flatten()
        ma = float(np.mean(vals))
        current = float(vals[-1])
        std_val = float(np.std(vals))
        
        if std_val == 0:
            return 0.5
        
        z_score = (current - ma) / std_val
        # Convert to 0-1 scale: extreme deviations = high reversion potential
        return float(1 / (1 + np.exp(-z_score)))  # Sigmoid function
    
    def classify_regime(self, 
                       sentiment_data: Dict[str, Any],
                       technical_data: Dict[str, Any],
                       prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Classify market regime using multiple factors.
        
        Input:
            sentiment_data: From sentiment_analyzer output
            technical_data: From technical_indicators output
            prices: Historical price series (optional)
        
        Output:
            Dictionary with regime probabilities and classification
        """
        
        # Helper function to convert values to float
        def to_float(val, default=0.0):
            """Safely convert value to float, handling Series/arrays"""
            if val is None:
                return default
            try:
                if isinstance(val, (pd.Series, np.ndarray)):
                    if len(val) == 1:
                        return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val.flat[0])
                    return float(val.mean())
                if hasattr(val, 'item'):
                    return float(val.item())
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # Extract features (convert to scalars)
        sentiment_mean = to_float(sentiment_data.get('overall_mean', 0))
        sentiment_std = to_float(sentiment_data.get('overall_std', 0))
        sentiment_trend = to_float(sentiment_data.get('sentiment_trend', 0))
        headline_count = to_float(sentiment_data.get('total_headlines', 0))
        
        # Technical features (from technical_indicators output)
        tech_score = to_float(technical_data.get('score', 0))
        tech_rsi = to_float(technical_data.get('technical', {}).get('rsi', 50))
        tech_macd = to_float(technical_data.get('technical', {}).get('macd', 0))
        
        # Calculate volatility from prices if available
        volatility = 0.3  # Default
        if prices is not None and len(prices) > 14:
            volatility = float(self.calculate_volatility(prices))
            trend_strength = float(self.calculate_trend_strength(prices))
            mean_reversion = float(self.calculate_mean_reversion_score(prices))
        else:
            trend_strength = 0.5
            mean_reversion = 0.5
        
        # =====================================================
        # BULL PROBABILITY
        # =====================================================
        bull_score = 0.0
        bull_factors = {}
        
        # Factor 1: Positive sentiment (weight: 0.25)
        sentiment_factor = (sentiment_mean + 1) / 2  # Normalize -1,1 to 0,1
        bull_score += sentiment_factor * 0.25
        bull_factors['sentiment'] = sentiment_factor
        
        # Factor 2: Improving sentiment trend (weight: 0.2)
        trend_factor = float(np.clip(sentiment_trend * 10, 0, 1))  # Amplify and clip
        bull_score += trend_factor * 0.2
        bull_factors['trend'] = trend_factor
        
        # Factor 3: Technical score (weight: 0.25)
        tech_factor = (tech_score + 10) / 20  # Normalize -10,10 to 0,1
        bull_score += tech_factor * 0.25
        bull_factors['technical'] = tech_factor
        
        # Factor 4: RSI not overbought (weight: 0.15)
        rsi_factor = 1 - (max(0, (tech_rsi - 60) / 40))  # Higher RSI = lower bull score
        bull_score += rsi_factor * 0.15
        bull_factors['rsi'] = rsi_factor
        
        # Factor 5: Strong trend (weight: 0.15)
        bull_score += trend_strength * 0.15
        bull_factors['trend_strength'] = trend_strength
        
        bull_probability = float(np.clip(bull_score, 0, 1))
        
        # =====================================================
        # BEAR PROBABILITY
        # =====================================================
        bear_score = 0.0
        bear_factors = {}
        
        # Factor 1: Negative sentiment (weight: 0.25)
        bear_factors['sentiment'] = 1 - sentiment_factor
        bear_score += bear_factors['sentiment'] * 0.25
        
        # Factor 2: Worsening sentiment (weight: 0.2)
        bear_factors['trend'] = 1 - trend_factor
        bear_score += bear_factors['trend'] * 0.2
        
        # Factor 3: Negative technical score (weight: 0.25)
        bear_factors['technical'] = 1 - tech_factor
        bear_score += bear_factors['technical'] * 0.25
        
        # Factor 4: RSI oversold (weight: 0.15)
        bear_factors['rsi'] = max(0, (30 - tech_rsi) / 30)
        bear_score += bear_factors['rsi'] * 0.15
        
        # Factor 5: Mean reversion opportunity (weight: 0.15)
        bear_factors['mean_reversion'] = mean_reversion
        bear_score += (1 - mean_reversion) * 0.15
        
        bear_probability = float(np.clip(bear_score, 0, 1))
        
        # =====================================================
        # VOLATILE PROBABILITY
        # =====================================================
        volatile_factors = {}
        
        # Factor 1: High volatility (weight: 0.4)
        volatility_normalized = min(volatility / 0.5, 1.0)  # 50% vol = max volatile
        volatile_factors['volatility'] = volatility_normalized
        volatile_score = volatility_normalized * 0.4
        
        # Factor 2: High sentiment std dev (weight: 0.3)
        sentiment_uncertainty = min(sentiment_std / 0.15, 1.0)
        volatile_factors['sentiment_uncertainty'] = sentiment_uncertainty
        volatile_score += sentiment_uncertainty * 0.3
        
        # Factor 3: Low headline engagement (ambiguous market) (weight: 0.2)
        engagement_factor = max(0, 1 - (headline_count / 50))
        volatile_factors['engagement'] = engagement_factor
        volatile_score += engagement_factor * 0.2
        
        # Factor 4: Extreme RSI (overbought or oversold) (weight: 0.1)
        extreme_rsi = max((tech_rsi - 50) / 50, 0)  # How far from 50
        volatile_factors['extreme_rsi'] = abs(extreme_rsi)
        volatile_score += abs(extreme_rsi) * 0.1
        
        volatile_probability = float(np.clip(volatile_score, 0, 1))
        
        # =====================================================
        # NORMALIZE PROBABILITIES
        # =====================================================
        
        total_prob = bull_probability + bear_probability + volatile_probability
        if total_prob > 0:
            bull_probability /= total_prob
            bear_probability /= total_prob
            volatile_probability /= total_prob
        
        # =====================================================
        # DETERMINE PRIMARY REGIME
        # =====================================================
        probs = {
            'bull': bull_probability,
            'bear': bear_probability,
            'volatile': volatile_probability
        }
        
        primary_regime = max(probs, key=probs.get)
        confidence = probs[primary_regime]
        
        # Map to emoji
        regime_emoji = {
            'bull': '🔼 BULL',
            'bear': '🔽 BEAR',
            'volatile': '🌪️  VOLATILE'
        }
        
        return {
            'symbol': technical_data.get('symbol', 'UNKNOWN'),
            'name': technical_data.get('name', 'Unknown'),
            'primary_regime': regime_emoji[primary_regime],
            'regime_type': primary_regime,
            'confidence': confidence,
            'probabilities': probs,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'bull_factors': bull_factors,
            'bear_factors': bear_factors,
            'volatile_factors': volatile_factors,
            'sentiment_data': sentiment_data,
            'technical_data': technical_data,
            'timestamp': datetime.now().isoformat()
        }


def load_technical_results(filepath: str) -> Optional[Dict]:
    """Load results from technical_indicators.py output"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in {filepath}")
        return None


def format_regime_output(result: Dict) -> str:
    """Format regime analysis for display"""
    output = f"\n{'='*70}\n"
    output += f"MARKET REGIME ANALYSIS: {result['symbol']} ({result['name']})\n"
    output += f"{'='*70}\n"
    output += f"\n🎯 PRIMARY REGIME: {result['primary_regime']}\n"
    output += f"   Confidence: {result['confidence']:.1%}\n"
    output += f"   Volatility: {result['volatility']:.1%}\n"
    
    output += f"\n📊 REGIME PROBABILITIES:\n"
    output += f"   🔼 Bull:     {result['probabilities']['bull']:.1%}\n"
    output += f"   🔽 Bear:     {result['probabilities']['bear']:.1%}\n"
    output += f"   🌪️  Volatile: {result['probabilities']['volatile']:.1%}\n"
    
    output += f"\n📈 BULL FACTORS:\n"
    for factor, score in result['bull_factors'].items():
        bar = "█" * int(score * 10)
        output += f"   • {factor:15s}: {score:.2f} {bar}\n"
    
    output += f"\n📉 BEAR FACTORS:\n"
    for factor, score in result['bear_factors'].items():
        bar = "█" * int(score * 10)
        output += f"   • {factor:15s}: {score:.2f} {bar}\n"
    
    output += f"\n⚡ VOLATILE FACTORS:\n"
    for factor, score in result['volatile_factors'].items():
        bar = "█" * int(score * 10)
        output += f"   • {factor:15s}: {score:.2f} {bar}\n"
    
    output += f"\n{'='*70}\n"
    return output


def save_regime_results(results: List[Dict], filename: str):
    """Save regime analysis to CSV"""
    with open(filename, 'w', newline='') as f:
        fieldnames = ['Symbol', 'Name', 'Regime', 'Confidence', 
                     'Bull_Prob', 'Bear_Prob', 'Volatile_Prob', 'Volatility']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'Symbol': result['symbol'],
                'Name': result['name'],
                'Regime': result['regime_type'].upper(),
                'Confidence': f"{result['confidence']:.1%}",
                'Bull_Prob': f"{result['probabilities']['bull']:.1%}",
                'Bear_Prob': f"{result['probabilities']['bear']:.1%}",
                'Volatile_Prob': f"{result['probabilities']['volatile']:.1%}",
                'Volatility': f"{result['volatility']:.1%}"
            })


def convert_types_for_json(obj):
    """Convert numpy and pandas types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_types_for_json(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [convert_types_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def main():
    """Main execution"""
    print("\n🎯 MARKET REGIME MODEL")
    print("=" * 70)
    print("Analyzes market regime (BULL / BEAR / VOLATILE) using")
    print("sentiment + technical indicators\n")
    
    # Load technical results
    tech_file = "data/output/latest.json"
    if not os.path.exists(tech_file):
        print(f"❌ Please run sentiment_analyzer.py first")
        sys.exit(1)
    
    # Load sentiment data for reference
    try:
        with open(tech_file, 'r') as f:
            sentiment_json = json.load(f)
    except Exception as e:
        print(f"❌ Error loading sentiment data: {e}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = RegimeAnalyzer()
    results = []
    
    # Analyze each asset
    for asset in sentiment_json.get('assets', []):
        symbol = asset['symbol']
        print(f"\n📊 Analyzing {symbol}...")
        
        # Prepare sentiment data
        sentiment_data = {
            'overall_mean': asset['overall_mean'],
            'overall_median': asset['overall_median'],
            'overall_std': asset['overall_std'],
            'total_headlines': asset['total_headlines'],
            'sentiment_trend': 0.0  # Will calculate from daily_means
        }
        
        # Calculate sentiment trend
        daily_means = asset.get('daily_means', [])
        if len(daily_means) >= 2:
            sentiments = [d['mean_sentiment'] for d in daily_means]
            x = np.arange(len(sentiments))
            sentiment_data['sentiment_trend'] = np.polyfit(x, sentiments, 1)[0]
        
        # Prepare technical data (simplified for now)
        technical_data = {
            'symbol': symbol,
            'name': asset['name'],
            'score': 0.0,  # Placeholder
            'technical': {
                'rsi': 50.0,
                'macd': 0.0
            }
        }
        
        # Get price data for volatility calculation
        prices = None
        if YFINANCE_AVAILABLE:
            try:
                df = yf.download(symbol, period='1mo', progress=False)
                close = df['Close']
                # Flatten DataFrame to Series (yfinance 2.x returns multi-level columns)
                if isinstance(close, pd.DataFrame):
                    prices = close.iloc[:, 0]
                else:
                    prices = close
            except:
                prices = None
        
        # Classify regime
        regime_result = analyzer.classify_regime(sentiment_data, technical_data, prices)
        results.append(regime_result)
        
        print(format_regime_output(regime_result))
    
    # Save results
    if results:
        output_file = os.path.join(OUTPUT_DIR, 
                                  f"market_regime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        save_regime_results(results, output_file)
        print(f"✅ Results saved to {output_file}\n")
        
        # Also save as JSON for downstream processing
        json_file = os.path.join(OUTPUT_DIR, 
                                f"market_regime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_file, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            json_data = convert_types_for_json(results)
            json.dump(json_data, f, indent=2)
        print(f"✅ JSON results saved to {json_file}")


if __name__ == "__main__":
    main()
