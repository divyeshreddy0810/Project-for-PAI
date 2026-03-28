#!/usr/bin/env python3
"""
Price Forecaster with LSTM
--------------------------
Predicts future price movements using:
- Historical OHLCV data
- Technical indicators
- Market regime classification
- Sentiment trends

Output: Price prediction for next 5, 10, 15, 20 days
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import csv
from typing import Dict, List, Any, Tuple, Optional

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not installed. Install with: pip install yfinance")

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not installed. Install with: pip install scikit-learn")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not installed. Using fallback methods.")


# ========================== CONFIGURATION ==========================
OUTPUT_DIR = "data/output"

# DATA COLLECTION SETTINGS (for large-scale training 1M+ records)
HISTORICAL_PERIOD = '5y'  # '1y', '2y', '5y', '10y' for extended training data
INTERVAL = '1d'  # '1d' (daily), '1h' (hourly), '15m' (15-min), '5m' (5-min) for intraday
# Intraday data multiplier:
#   - '1d': ~252 records/year per asset
#   - '1h': ~1,512 records/year (6.5h/day × 252 days)
#   - '15m': ~5,616 records/year (26 intervals/day × 252 days)
#   - '5m': ~16,848 records/year (78 intervals/day × 252 days)
# With 100+ assets at hourly intervals over 5 years > 1M training samples

LOOKBACK_PERIOD = 20  # Historical bars to use for prediction (adaptive to interval)
MIN_DATA_POINTS = 50  # Minimum data points needed for any prediction
PREDICTION_HORIZONS = [5, 10, 15, 20]  # Predict 5, 10, 15, 20 periods ahead (days/hours/etc)


class PriceForecaster:
    """Forecasts future prices using multiple methods"""
    
    def __init__(self, symbol: str, regime_data: Optional[Dict] = None,
                 period: str = HISTORICAL_PERIOD, interval: str = INTERVAL):
        """
        Initialize forecaster
        
        Args:
            symbol: Stock symbol (e.g., '^GSPC', 'BTC-USD')
            regime_data: Market regime classification from market_regime_model.py
            period: Historical period ('1y', '5y', '10y', etc) for training data volume
            interval: Data interval ('1d', '1h', '15m', '5m') for intraday high-frequency data
        """
        self.symbol = symbol
        self.regime_data = regime_data
        self.period = period  # Historical lookback window
        self.interval = interval  # Data frequency (daily/hourly/15-min/etc)
        self.scaler = MinMaxScaler() if SKLEARN_AVAILABLE else None
        self.model = None
    
    def fetch_price_data(self, period: Optional[str] = None, interval: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch historical price data with support for extended periods and intraday intervals"""
        if not YFINANCE_AVAILABLE:
            print(f"❌ yfinance not available")
            return None
        
        # Use instance parameters if not provided
        period = period or self.period
        interval = interval or self.interval
        
        # For extended historical data (5-10y) with intraday intervals:
        # Can generate 750K-2.7M+ data points per asset
        periods_to_try = [period, '5y', '3y', '1y']  # Try requested period first
        df = None
        
        for p in periods_to_try:
            try:
                print(f"   📥 Fetching {p} of price data (interval={interval}) for {self.symbol}...")
                df = yf.download(self.symbol, period=p, interval=interval, progress=False)
                
                if df is not None and len(df) >= MIN_DATA_POINTS:
                    print(f"   ✅ Got {len(df)} records of {interval} data (period: {p})")
                    return df
            except Exception as e:
                print(f"   ⚠️  {p} with interval={interval} failed: {e}")
                # Fallback to daily if intraday fails
                if interval != '1d':
                    try:
                        print(f"   📥 Falling back to daily data for {self.symbol}...")
                        df = yf.download(self.symbol, period=p, progress=False)
                        if df is not None and len(df) >= MIN_DATA_POINTS:
                            print(f"   ✅ Got {len(df)} days of data")
                            return df
                    except:
                        continue
                continue
        
        # If we got here but have some data, return it
        if df is not None and len(df) > 0:
            print(f"   ⚠️  Got {len(df)} records (less than ideal but usable)")
            return df
        
        return None
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for price prediction"""
        df = df.copy()
        
        # Flatten multi-level columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Price_MA5'] = df['Close'].rolling(5, min_periods=2).mean()
        df['Price_MA20'] = df['Close'].rolling(20, min_periods=5).mean()
        # Use adaptive MA50 - if < 50 days available, use what we have
        ma50_period = min(50, max(10, len(df) // 3))
        df['Price_MA50'] = df['Close'].rolling(ma50_period, min_periods=3).mean()
        
        # Volatility - use adaptive period
        vol_period = min(14, max(5, len(df) // 8))
        df['Volatility'] = df['Returns'].rolling(vol_period, min_periods=2).std()
        df['Volatility'] = df['Volatility'].fillna(df['Returns'].std())  # Fill with overall std
        
        # Momentum
        df['RSI'] = self._calculate_rsi(df['Close'])
        macd_result = self._calculate_macd(df['Close'])
        if isinstance(macd_result, tuple):
            df['MACD'], df['Signal'] = macd_result
        else:
            df['MACD'] = macd_result
            df['Signal'] = df['MACD'].rolling(9, min_periods=2).mean()
        
        # Fill NaN values in momentum indicators
        df['RSI'] = df['RSI'].fillna(50)  # Neutral RSI
        df['MACD'] = df['MACD'].fillna(0)
        df['Signal'] = df['Signal'].fillna(0)
        
        # Volume trend (with proper NaN handling)
        df['Volume_MA'] = df['Volume'].rolling(20, min_periods=5).mean()
        
        # Safe volume trend calculation
        volume_trend = []
        for i in range(len(df)):
            vol = df['Volume'].iloc[i]
            vol_ma = df['Volume_MA'].iloc[i]
            if pd.notna(vol_ma) and vol_ma != 0:
                vol_val = vol.item() if hasattr(vol, 'item') else float(vol)
                vol_ma_val = vol_ma.item() if hasattr(vol_ma, 'item') else float(vol_ma)
                trend = vol_val / vol_ma_val
            else:
                trend = 1.0
            volume_trend.append(trend)
        df['Volume_Trend'] = pd.Series(volume_trend, index=df.index)
        df['Volume_Trend'] = df['Volume_Trend'].fillna(1.0)
        
        # Range
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_Range'] = (df['Close'] - df['Open']) / df['Open'].replace(0, 1.0)
        df['High_Low_Range'] = df['High_Low_Range'].fillna(0)
        df['Open_Close_Range'] = df['Open_Close_Range'].fillna(0)
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def prepare_training_data(self, df: pd.DataFrame, 
                             lookback: int = LOOKBACK_PERIOD) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare data for LSTM or regression model
        
        Returns:
            X: Feature sequences of shape (samples, lookback, features)
            y: Target values (next day returns)
        """
        # Drop NaN rows
        df = df.dropna()
        
        if len(df) < lookback + 1:
            print(f"   ⚠️  Not enough data (need {lookback} days)")
            return None
        
        # Select features
        feature_cols = ['Returns', 'Price_MA5', 'Price_MA20', 'Price_MA50',
                       'Volatility', 'RSI', 'MACD', 'Signal', 'Volume_Trend',
                       'High_Low_Range', 'Open_Close_Range']
        
        X_data = df[feature_cols].values
        y_data = df['Close'].values
        
        # Normalize if sklearn available
        if SKLEARN_AVAILABLE:
            X_data = self.scaler.fit_transform(X_data)
        else:
            X_data = (X_data - np.mean(X_data, axis=0)) / (np.std(X_data, axis=0) + 1e-8)
        
        # Create sequences
        X, y = [], []
        for i in range(len(X_data) - lookback):
            X.append(X_data[i:i+lookback])
            y.append(y_data[i+lookback])
        
        return np.array(X), np.array(y)
    
    def train_simple_model(self, df: pd.DataFrame) -> bool:
        """Train simple linear regression model for fast predictions"""
        try:
            if not SKLEARN_AVAILABLE:
                print("   ⚠️  scikit-learn not available for training")
                return False
            
            print(f"   🤖 Training prediction model...")
            
            # Prepare data
            training_data = self.prepare_training_data(df)
            if training_data is None:
                return False
            
            X, y = training_data
            
            # Flatten X for linear regression
            X_flat = X.reshape(X.shape[0], -1)
            
            # Train model
            self.model = LinearRegression()
            self.model.fit(X_flat, y)
            
            # Calculate R² score
            train_score = self.model.score(X_flat, y)
            print(f"   ✅ Model trained (R² = {train_score:.3f})")
            
            return True
        
        except Exception as e:
            print(f"   ⚠️  Training error: {e}")
            return False
    
    def predict_prices(self, df: pd.DataFrame, 
                      horizons: List[int] = PREDICTION_HORIZONS) -> Dict[str, Any]:
        """
        Predict future prices using momentum-based approach
        
        Args:
            df: Historical price dataframe with calculated features
            horizons: Number of days ahead to predict [5, 10, 15, 20]
        
        Returns:
            Dictionary with predictions for each horizon
        """
        # Only drop rows with missing Close prices, not all NaN values
        df = df[df['Close'].notna()].copy()
        
        # Fill remaining NaN values in features with forward fill then backward fill
        df = df.ffill().bfill().fillna(0)
        
        # Check for minimum data requirement
        if len(df) < MIN_DATA_POINTS:
            print(f"   ⚠️  Not enough data ({len(df)} < {MIN_DATA_POINTS})")
            if len(df) > 0:
                last_close = df['Close'].iloc[-1]
                last_date = df.index[-1]
                last_close_val = last_close.item() if hasattr(last_close, 'item') else float(last_close)
                return {
                    'current_price': last_close_val,
                    'prediction_date': last_date.strftime('%Y-%m-%d'),
                    'horizons': {},
                    'error': f'Insufficient data ({len(df)} points < {MIN_DATA_POINTS} required)'
                }
            else:
                return {'error': 'No valid price data available'}
        
        # We have enough data, proceed with predictions
        print(f"   ✅ Making predictions with {len(df)} days of data")
        
        last_close = df['Close'].iloc[-1]
        last_date = df.index[-1]
        last_close_val = last_close.item() if hasattr(last_close, 'item') else float(last_close)
        
        predictions = {
            'current_price': last_close_val,
            'prediction_date': last_date.strftime('%Y-%m-%d'),
            'horizons': {},
            'data_points': len(df),
            'confidence_modifier': max(0.5, min(1.0, len(df) / 30))  # Lower confidence with less data
        }
        
        # Method 1: Simple momentum-based prediction
        # Use available history (at least 5 days, up to 20)
        lookback_days = min(20, max(5, len(df) - 1))
        recent_returns = df['Returns'].iloc[-lookback_days:].mean()
        volatility = df['Volatility'].iloc[-1]
        
        # Handle NaN volatility
        if pd.isna(volatility) or volatility is None:
            # Calculate volatility from returns
            volatility = df['Returns'].iloc[-lookback_days:].std() or 0.01
        volatility = max(0.001, volatility)  # Ensure positive
        
        trend = (df['Close'].iloc[-1] - df['Close'].iloc[-lookback_days]) / df['Close'].iloc[-lookback_days]
        
        for horizon in horizons:
            # Base prediction: momentum extrapolation
            # Convert daily returns to horizon-periods returns
            recent_returns_safe = recent_returns if pd.notna(recent_returns) else 0.0
            expected_return = recent_returns_safe * horizon
            
            # Add trend component 
            trend_boost = trend / 10  # Scale trend contribution
            expected_return += trend_boost
            
            # Adjust for regime
            confidence = max(0.3, 1 - volatility * 2)
            if self.regime_data:
                regime_type = self.regime_data.get('regime_type', 'neutral')
                if regime_type == 'bull':
                    expected_return *= 1.1  # Boost bullish predictions
                    confidence *= 1.1
                elif regime_type == 'bear':
                    expected_return *= 0.9  # Reduce bearish predictions
                    confidence *= 0.9
            
            # Apply confidence modifier based on data points
            confidence *= predictions.get('confidence_modifier', 1.0)
            confidence = float(max(0.2, min(1.0, confidence)))  # Clamp to [0.2, 1.0]
            
            # Calculate predicted price
            predicted_price = last_close_val * (1 + expected_return)
            
            # Calculate price range (wider for longer horizons and higher volatility)
            std_move = last_close_val * volatility * np.sqrt(horizon / 252)
            price_range = (
                predicted_price - std_move,
                predicted_price + std_move
            )
            
            predictions['horizons'][horizon] = {
                'days_ahead': horizon,
                'predicted_price': float(predicted_price),
                'target_date': (last_date + timedelta(days=horizon)).strftime('%Y-%m-%d'),
                'expected_return': float(expected_return),
                'price_range': (float(price_range[0]), float(price_range[1])),
                'confidence': float(confidence),
                'reasoning': self._get_prediction_reasoning(trend, volatility, horizon)
            }
        
        return predictions
    
    @staticmethod
    def _get_prediction_reasoning(trend: float, volatility: float, horizon: int) -> str:
        """Generate explanation for prediction"""
        if abs(trend) > 0.1:
            direction = "uptrend" if trend > 0 else "downtrend"
            reasoning = f"Strong {direction} detected"
        elif volatility > 0.4:
            reasoning = "High volatility: expect wide price swings"
        elif volatility < 0.1:
            reasoning = "Low volatility: stable price expected"
        else:
            reasoning = "Neutral trend: mean reversion likely"
        
        reasoning += f" over {horizon}-day period"
        return reasoning


def load_regime_data(regime_file: str) -> Optional[Dict]:
    """Load regime classification from market_regime_model output"""
    if not os.path.exists(regime_file):
        return None
    
    try:
        with open(regime_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # Return first result
        return data
    except Exception as e:
        print(f"⚠️  Error loading regime data: {e}")
        return None


def format_prediction_output(result: Dict) -> str:
    """Format prediction results for display"""
    # Handle error cases
    if 'error' in result:
        return f"\n⚠️  Price Forecast: {result.get('error', 'Unknown error')}\n"
    
    if not result or 'current_price' not in result:
        return f"\n⚠️  No prediction results available\n"
    
    output = f"\n{'='*70}\n"
    output += f"PRICE FORECAST\n"
    output += f"{'='*70}\n"
    output += f"\nCurrent Price: ${result['current_price']:.2f}\n"
    output += f"Analysis Date: {result.get('prediction_date', 'N/A')}\n\n"
    
    # Handle empty horizons
    if not result.get('horizons'):
        output += "⚠️  No predictions available (insufficient historical data)\n"
        return output
    
    output += f"{'Days':<8} {'Target Date':<15} {'Predicted':<15} {'Range':<25} {'Return':<12} {'Conf':<8}\n"
    output += "-" * 83 + "\n"
    
    for horizon, pred in result['horizons'].items():
        price_low, price_high = pred['price_range']
        output += f"{horizon:<8} {pred['target_date']:<15} ${pred['predicted_price']:<14.2f} "
        output += f"${price_low:.2f}-${price_high:.2f} "
        output += f"{pred['expected_return']:>+10.2%} {pred['confidence']:<6.0%}\n"
    
    output += "-" * 83 + "\n"
    output += f"\n📌 Prediction Reasoning:\n"
    for horizon, pred in result['horizons'].items():
        output += f"   {horizon}D: {pred['reasoning']}\n"
    
    output += f"\n{'='*70}\n"
    return output


def save_predictions(results: List[Dict], filename: str):
    """Save predictions to CSV"""
    with open(filename, 'w', newline='') as f:
        fieldnames = ['Symbol', 'Current_Price', 'Days_Ahead', 'Target_Date', 
                     'Predicted_Price', 'Price_Range', 'Expected_Return', 'Confidence']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            symbol = result.get('symbol', 'UNKNOWN')
            current = result['current_price']
            
            for horizon, pred in result['horizons'].items():
                price_low, price_high = pred['price_range']
                writer.writerow({
                    'Symbol': symbol,
                    'Current_Price': f"${current:.2f}",
                    'Days_Ahead': horizon,
                    'Target_Date': pred['target_date'],
                    'Predicted_Price': f"${pred['predicted_price']:.2f}",
                    'Price_Range': f"${price_low:.2f}-${price_high:.2f}",
                    'Expected_Return': f"{pred['expected_return']:.2%}",
                    'Confidence': f"{pred['confidence']:.0%}"
                })


def main():
    """Main execution"""
    print("\n💰 PRICE FORECASTER WITH MULTIMODAL ANALYSIS")
    print("=" * 70)
    print("Predicts future prices using sentiment + technical + regime data\n")
    
    # Load sentiment data for symbols
    sentiment_file = "data/output/latest.json"
    if not os.path.exists(sentiment_file):
        print(f"❌ Please run sentiment_analyzer.py first")
        sys.exit(1)
    
    try:
        with open(sentiment_file, 'r') as f:
            sentiment_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading sentiment data: {e}")
        sys.exit(1)
    
    # Try to load latest regime data
    regime_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('market_regime_') and f.endswith('.json')]
    regime_data = None
    if regime_files:
        latest_regime_file = os.path.join(OUTPUT_DIR, sorted(regime_files)[-1])
        print(f"Using regime data from: {latest_regime_file}\n")
        regime_data = load_regime_data(latest_regime_file)
    
    results = []
    
    # Forecast prices for each asset
    for asset in sentiment_data.get('assets', []):
        symbol = asset['symbol']
        name = asset['name']
        
        print(f"📊 Forecasting {symbol} ({name})...")
        
        # Initialize forecaster
        forecaster = PriceForecaster(symbol, regime_data)
        
        # Fetch price data
        df = forecaster.fetch_price_data(period='3mo')
        if df is None or df.empty:
            print(f"   ⚠️  Skipping {symbol}")
            continue
        
        # Calculate features
        df = forecaster.calculate_features(df)
        
        # Train model
        forecaster.train_simple_model(df)
        
        # Make predictions
        prediction = forecaster.predict_prices(df)
        prediction['symbol'] = symbol
        prediction['name'] = name
        
        results.append(prediction)
        print(format_prediction_output(prediction))
    
    # Save results
    if results:
        # CSV output
        csv_file = os.path.join(OUTPUT_DIR, 
                               f"price_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        save_predictions(results, csv_file)
        print(f"✅ CSV results saved to {csv_file}")
        
        # JSON output
        json_file = os.path.join(OUTPUT_DIR, 
                                f"price_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ JSON results saved to {json_file}")


if __name__ == "__main__":
    main()
