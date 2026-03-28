#!/usr/bin/env python3
"""
Reinforcement Learning Trader
------------------------------
Makes trading decisions (BUY / SELL / HOLD) based on:
- Price predictions (from price_forecaster.py)
- Market regime (from market_regime_model.py)
- Risk management rules
- Portfolio optimization

Output: Trading signals with risk assessment and expected returns
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import csv
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum


# ========================== CONFIGURATION ==========================
OUTPUT_DIR = "data/output"

class TradeSignal(Enum):
    """Trading decision signals"""
    STRONG_BUY = 1.0
    BUY = 0.5
    HOLD = 0.0
    SELL = -0.5
    STRONG_SELL = -1.0


class RiskProfile(Enum):
    """Risk management profiles"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# Risk parameters by profile
RISK_PARAMS = {
    RiskProfile.CONSERVATIVE: {
        'max_position_size': 0.05,      # 5% of portfolio per trade
        'stop_loss': 0.05,              # 5% stop loss
        'take_profit': 0.10,            # 10% take profit
        'min_confidence': 0.60,         # Require 60% confidence
        'max_leverage': 1.0             # No leverage
    },
    RiskProfile.MODERATE: {
        'max_position_size': 0.10,      # 10% per trade
        'stop_loss': 0.08,              # 8% stop loss
        'take_profit': 0.15,            # 15% take profit
        'min_confidence': 0.55,         # Require 55% confidence
        'max_leverage': 2.0             # 2x leverage
    },
    RiskProfile.AGGRESSIVE: {
        'max_position_size': 0.20,      # 20% per trade
        'stop_loss': 0.10,              # 10% stop loss
        'take_profit': 0.25,            # 25% take profit
        'min_confidence': 0.50,         # Require 50% confidence
        'max_leverage': 3.0             # 3x leverage
    }
}


class TradeSignalGenerator:
    """Generates trading signals using multi-factor analysis"""
    
    def __init__(self, risk_profile: RiskProfile = RiskProfile.MODERATE,
                 portfolio_value: float = 100000.0):
        """
        Initialize trader
        
        Args:
            risk_profile: Investment risk profile
            portfolio_value: Current portfolio value in dollars
        """
        self.risk_profile = risk_profile
        self.portfolio_value = portfolio_value
        self.risk_params = RISK_PARAMS[risk_profile]
    
    def generate_signal(self, 
                       price_prediction: Dict[str, Any],
                       regime_data: Optional[Dict[str, Any]] = None,
                       current_price: float = None) -> Dict[str, Any]:
        """
        Generate trading signal based on multiple factors
        
        Args:
            price_prediction: From price_forecaster.py
            regime_data: From market_regime_model.py
            current_price: Current market price
        
        Returns:
            Trading signal with recommendation and risk metrics
        """
        
        # Extract prediction data for 5-day horizon (primary decision)
        # Handle both integer and string keys from JSON
        horizons = price_prediction['horizons']
        horizon_5d = horizons.get(5) or horizons.get('5') or {}
        
        if not horizon_5d:
            return self._create_neutral_signal(price_prediction)
        
        current = price_prediction['current_price']
        predicted_5d = horizon_5d['predicted_price']
        expected_return = horizon_5d['expected_return']
        confidence = horizon_5d['confidence']
        price_range = horizon_5d['price_range']
        
        # =====================================================
        # FACTOR 1: PRICE MOMENTUM
        # =====================================================
        momentum_score = 0.0
        momentum_factors = {}
        
        # Expected return signal
        if expected_return > 0.05:  # 5% up
            momentum_score += 1.0
            momentum_factors['strong_upside'] = 1.0
        elif expected_return > 0.02:  # 2% up
            momentum_score += 0.5
            momentum_factors['mild_upside'] = 0.5
        elif expected_return < -0.05:  # 5% down
            momentum_score -= 1.0
            momentum_factors['strong_downside'] = -1.0
        elif expected_return < -0.02:  # 2% down
            momentum_score -= 0.5
            momentum_factors['mild_downside'] = -0.5
        else:
            momentum_factors['neutral'] = 0.0
        
        # =====================================================
        # FACTOR 2: REGIME ALIGNMENT
        # =====================================================
        regime_score = 0.0
        regime_factors = {}
        
        if regime_data:
            regime_type = regime_data.get('regime_type', 'unknown')
            regime_confidence = regime_data.get('confidence', 0.5)
            
            if regime_type == 'bull':
                regime_score = 0.5 * regime_confidence
                regime_factors['bull_regime'] = regime_score
            elif regime_type == 'bear':
                regime_score = -0.5 * regime_confidence
                regime_factors['bear_regime'] = regime_score
            else:
                regime_factors['volatile_regime'] = 0.0
        
        # =====================================================
        # FACTOR 3: RISK/REWARD RATIO
        # =====================================================
        # Calculate potential risk and reward
        lower_price, upper_price = price_range
        
        # Downside risk (to lower bound)
        downside_risk = abs((current - lower_price) / current)
        
        # Upside reward (to upper bound)
        upside_reward = abs((upper_price - current) / current)
        
        # Risk-reward ratio
        if downside_risk > 0:
            reward_risk_ratio = upside_reward / downside_risk
        else:
            reward_risk_ratio = 1.0
        
        reward_risk_score = 0.0
        if reward_risk_ratio > 2.0:  # 2:1 favorable
            reward_risk_score = 1.0
        elif reward_risk_ratio > 1.5:  # 1.5:1
            reward_risk_score = 0.5
        elif reward_risk_ratio < 0.5:  # BAD: more risk than reward
            reward_risk_score = -1.0
        
        # =====================================================
        # FACTOR 4: PREDICTION CONFIDENCE
        # =====================================================
        # Higher confidence = stronger signal
        confidence_score = (confidence - 0.5) * 2  # Scale to -1 to 1
        confidence_score = np.clip(confidence_score, -1, 1)
        
        # =====================================================
        # COMBINE ALL FACTORS
        # =====================================================
        # Weights for each factor
        weights = {
            'momentum': 0.25,      # Reduced from 0.35
            'regime': 0.25,
            'reward_risk': 0.20,   # Reduced from 0.25
            'confidence': 0.30     # Increased from 0.15 - high confidence is key signal
        }
        
        total_score = (
            momentum_score * weights['momentum'] +
            regime_score * weights['regime'] +
            reward_risk_score * weights['reward_risk'] +
            confidence_score * weights['confidence']
        )
        
        # =====================================================
        # GENERATE TRADING SIGNAL
        # =====================================================
        signal = self._score_to_signal(total_score)
        
        # =====================================================
        # POSITION SIZING
        # =====================================================
        position_size = self._calculate_position_size(
            total_score, confidence, downside_risk
        )
        
        # =====================================================
        # RISK METRICS
        # =====================================================
        stop_loss_price = current * (1 - self.risk_params['stop_loss'])
        take_profit_price = current * (1 + self.risk_params['take_profit'])
        
        # Expected value
        win_probability = (confidence + 0.5) / 2  # Scale confidence to probability
        expected_value = (
            win_probability * expected_return -
            (1 - win_probability) * self.risk_params['stop_loss']
        )
        
        return {
            'symbol': price_prediction.get('symbol', 'UNKNOWN'),
            'current_price': current,
            'predicted_price': predicted_5d,
            'expected_return': expected_return,
            'signal': signal.name,
            'signal_strength': total_score,
            'confidence': confidence,
            'position_size': position_size,
            'position_value': position_size * self.portfolio_value,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'risk_reward_ratio': reward_risk_ratio,
            'downside_risk': downside_risk,
            'expected_value': expected_value,
            'recommendation': self._generate_recommendation(
                signal, expected_value, confidence, position_size
            ),
            'momentum_score': momentum_score,
            'regime_score': regime_score,
            'reward_risk_score': reward_risk_score,
            'confidence_score': confidence_score,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def _score_to_signal(score: float) -> TradeSignal:
        """Convert numeric score to trading signal"""
        if score >= 0.4:
            return TradeSignal.STRONG_BUY
        elif score >= 0.1:
            return TradeSignal.BUY
        elif score >= -0.1:
            return TradeSignal.HOLD
        elif score >= -0.4:
            return TradeSignal.SELL
        else:
            return TradeSignal.STRONG_SELL
    
    def _calculate_position_size(self, 
                                 signal_strength: float,
                                 confidence: float,
                                 risk: float) -> float:
        """Calculate optimal position size based on signal strength and risk"""
        
        base_size = self.risk_params['max_position_size']
        
        # Adjust by signal strength
        size = base_size * abs(signal_strength)
        
        # Adjust by confidence (don't trade if confidence too low)
        if confidence < self.risk_params['min_confidence']:
            size *= 0.5
        
        # Adjust by risk (lower size for high-risk trades)
        if risk > self.risk_params['stop_loss']:
            size *= 0.7
        
        return float(np.clip(size, 0, base_size))
    
    def _generate_recommendation(self,
                                signal: TradeSignal,
                                expected_value: float,
                                confidence: float,
                                position_size: float) -> str:
        """Generate human-readable trading recommendation"""
        
        ev_pct = expected_value * 100
        
        if signal == TradeSignal.STRONG_BUY:
            return f"Strong Buy: EV={ev_pct:.1f}%, Confidence={confidence:.0%}, Size={position_size:.1%}"
        elif signal == TradeSignal.BUY:
            return f"Buy: EV={ev_pct:.1f}%, Confidence={confidence:.0%}, Size={position_size:.1%}"
        elif signal == TradeSignal.HOLD:
            return f"Hold: Neutral setup, monitor for better entry"
        elif signal == TradeSignal.SELL:
            return f"Sell: EV={ev_pct:.1f}%, Risk={1-confidence:.0%}, Consider exit"
        else:  # STRONG_SELL
            return f"Strong Sell: Avoid or exit position, negative EV"
    
    @staticmethod
    def _create_neutral_signal(price_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Create neutral HOLD signal when data is insufficient"""
        return {
            'symbol': price_prediction.get('symbol', 'UNKNOWN'),
            'current_price': price_prediction['current_price'],
            'signal': TradeSignal.HOLD.name,
            'signal_strength': 0.0,
            'confidence': 0.0,
            'position_size': 0.0,
            'recommendation': 'Insufficient data for trading decision',
            'timestamp': datetime.now().isoformat()
        }


def load_predictions(prediction_file: str) -> Optional[List[Dict]]:
    """Load price predictions from price_forecaster output"""
    if not os.path.exists(prediction_file):
        return None
    
    try:
        with open(prediction_file, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"⚠️  Error loading predictions: {e}")
        return None


def load_regime_data(regime_file: str) -> Optional[List[Dict]]:
    """Load regime classifications from market_regime_model output"""
    if not os.path.exists(regime_file):
        return None
    
    try:
        with open(regime_file, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"⚠️  Error loading regime data: {e}")
        return None


def format_signal_output(result: Dict) -> str:
    """Format trading signal for display"""
    output = f"\n{'='*80}\n"
    output += f"TRADING SIGNAL: {result.get('symbol', 'N/A')}\n"
    output += f"{'='*80}\n"
    
    # Helper to safely format values
    def safe_format(value, format_spec=''):
        if value is None:
            return 'N/A'
        try:
            if format_spec.endswith('%'):
                return f"{value:.1%}"
            elif format_spec.endswith(':'):
                return f"{value:+.2f}:1"
            elif format_spec.startswith('$'):
                if isinstance(value, (int, float)):
                    return f"${value:,.0f}"
                return f"${value}"
            elif format_spec == 'price':
                return f"${value:.2f}"
            elif format_spec == 'percent':
                return f"{value:.1%}"
            elif format_spec == 'score':
                return f"{value:+.2f}"
            else:
                return str(value)
        except (TypeError, ValueError):
            return 'N/A'
    
    signal_emoji = {
        'STRONG_BUY': '🚀',
        'BUY': '📈',
        'HOLD': '⏸️ ',
        'SELL': '📉',
        'STRONG_SELL': '💥'
    }
    
    signal = result.get('signal', 'HOLD')
    emoji = signal_emoji.get(signal, '❓')
    output += f"\n{emoji} SIGNAL: {signal}\n"
    output += f"   Strength: {safe_format(result.get('signal_strength', 0), 'score')}/1.0\n"
    output += f"   Confidence: {safe_format(result.get('confidence', 0), 'percent')}\n"
    
    output += f"\n💰 PRICE TARGETS:\n"
    output += f"   Current:  {safe_format(result.get('current_price', 0), 'price')}\n"
    output += f"   Predicted (5D): {safe_format(result.get('predicted_price', 0), 'price')}\n"
    output += f"   Expected Return: {safe_format(result.get('expected_return', 0), 'percent')}\n"
    
    output += f"\n🎯 RISK MANAGEMENT:\n"
    output += f"   Stop Loss:  {safe_format(result.get('stop_loss', 0), 'price')}\n"
    output += f"   Take Profit: {safe_format(result.get('take_profit', 0), 'price')}\n"
    output += f"   Risk/Reward Ratio: {safe_format(result.get('risk_reward_ratio', 0), ':')}\n"
    output += f"   Downside Risk:  {safe_format(result.get('downside_risk', 0), 'percent')}\n"
    
    output += f"\n📊 POSITION SIZING:\n"
    output += f"   Order Size:   {safe_format(result.get('position_size', 0), 'percent')}\n"
    output += f"   Order Value:  {safe_format(result.get('position_value', 0), '$')}\n"
    
    output += f"\n📈 ANALYSIS BREAKDOWN:\n"
    output += f"   Momentum:     {safe_format(result.get('momentum_score', 0), 'score')}\n"
    output += f"   Regime:       {safe_format(result.get('regime_score', 0), 'score')}\n"
    output += f"   Risk/Reward:  {safe_format(result.get('reward_risk_score', 0), 'score')}\n"
    output += f"   Confidence:   {safe_format(result.get('confidence_score', 0), 'score')}\n"
    
    output += f"\n✅ RECOMMENDATION:\n"
    output += f"   {result.get('recommendation', 'No recommendation available')}\n"
    
    output += f"\n{'='*80}\n"
    return output


def save_trading_signals(results: List[Dict], filename: str):
    """Save trading signals to CSV"""
    def safe_format(value, format_spec=''):
        """Safely format values that might be None"""
        if value is None:
            return 'N/A'
        try:
            if format_spec == 'percent':
                return f"{float(value):.1%}"
            elif format_spec == 'percent_signed':
                return f"{float(value):+.2%}"
            elif format_spec == 'score':
                return f"{float(value):+.2f}"
            elif format_spec == 'price':
                return f"${float(value):.2f}"
            elif format_spec == 'money':
                return f"${float(value):,.0f}"
            elif format_spec == 'ratio':
                return f"{float(value):.2f}:1"
            else:
                return str(value)
        except (TypeError, ValueError):
            return 'N/A'
    
    with open(filename, 'w', newline='') as f:
        fieldnames = ['Symbol', 'Signal', 'Strength', 'Confidence', 'Current_Price',
                     'Predicted_Price', 'Expected_Return', 'Position_Size', 'Position_Value',
                     'Stop_Loss', 'Take_Profit', 'Risk_Reward_Ratio', 'Expected_Value']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'Symbol': result.get('symbol', 'Unknown'),
                'Signal': result.get('signal', 'N/A'),
                'Strength': safe_format(result.get('signal_strength', 0), 'score'),
                'Confidence': safe_format(result.get('confidence', 0), 'percent'),
                'Current_Price': safe_format(result.get('current_price', 0), 'price'),
                'Predicted_Price': safe_format(result.get('predicted_price', 0), 'price'),
                'Expected_Return': safe_format(result.get('expected_return', 0), 'percent_signed'),
                'Position_Size': safe_format(result.get('position_size', 0), 'percent'),
                'Position_Value': safe_format(result.get('position_value', 0), 'money'),
                'Stop_Loss': safe_format(result.get('stop_loss', 0), 'price'),
                'Take_Profit': safe_format(result.get('take_profit', 0), 'price'),
                'Risk_Reward_Ratio': safe_format(result.get('risk_reward_ratio', 0), 'ratio'),
                'Expected_Value': safe_format(result.get('expected_value', 0), 'percent_signed')
            })


def main():
    """Main execution"""
    print("\n🤖 REINFORCEMENT LEARNING TRADER")
    print("=" * 80)
    print("Generates buy/sell/hold signals using price predictions + regime data\n")
    
    # Get risk profile from user
    print("Select risk profile:")
    print("  1. Conservative (Low risk)")
    print("  2. Moderate (Medium risk)")
    print("  3. Aggressive (High risk)")
    
    choice = input("\nEnter choice (1-3, default=2): ").strip() or "2"
    
    risk_profile_map = {'1': RiskProfile.CONSERVATIVE, '2': RiskProfile.MODERATE, '3': RiskProfile.AGGRESSIVE}
    risk_profile = risk_profile_map.get(choice, RiskProfile.MODERATE)
    
    print(f"\nUsing {risk_profile.value.upper()} risk profile\n")
    
    # Get portfolio value
    portfolio_input = input("Enter portfolio value (default=$100,000): ").strip() or "100000"
    try:
        portfolio_value = float(portfolio_input)
    except ValueError:
        portfolio_value = 100000.0
    
    # Load latest predictions
    prediction_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('price_forecast_') and f.endswith('.json')]
    if not prediction_files:
        print(f"❌ No price forecasts found. Run price_forecaster.py first.")
        sys.exit(1)
    
    latest_prediction_file = os.path.join(OUTPUT_DIR, sorted(prediction_files)[-1])
    print(f"\n📈 Using predictions from: {latest_prediction_file}")
    
    predictions = load_predictions(latest_prediction_file)
    if not predictions:
        print(f"❌ Could not load predictions")
        sys.exit(1)
    
    # Load latest regime data
    regime_datasets = {}
    regime_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('market_regime_') and f.endswith('.json')]
    if regime_files:
        latest_regime_file = os.path.join(OUTPUT_DIR, sorted(regime_files)[-1])
        print(f"📊 Using regime data from: {latest_regime_file}\n")
        regime_data_list = load_regime_data(latest_regime_file)
        if regime_data_list:
            for regime_item in regime_data_list:
                regime_datasets[regime_item.get('symbol')] = regime_item
    
    # Generate trading signals
    trader = TradeSignalGenerator(risk_profile, portfolio_value)
    results = []
    
    print(f"🚀 GENERATING TRADING SIGNALS\n")
    
    for prediction in predictions:
        symbol = prediction.get('symbol', 'UNKNOWN')
        print(f"Processing {symbol}...")
        
        regime = regime_datasets.get(symbol)
        signal = trader.generate_signal(prediction, regime)
        results.append(signal)
        
        print(format_signal_output(signal))
    
    # Save results
    if results:
        # CSV output
        csv_file = os.path.join(OUTPUT_DIR,
                               f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        save_trading_signals(results, csv_file)
        print(f"✅ CSV results saved to {csv_file}")
        
        # JSON output
        json_file = os.path.join(OUTPUT_DIR,
                                f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ JSON results saved to {json_file}")
        
        # Portfolio summary
        print(f"\n📋 PORTFOLIO SUMMARY:")
        print(f"   Total Portfolio Value: ${portfolio_value:,.0f}")
        
        total_buy_size = sum(r['position_size'] for r in results if 'BUY' in r['signal'])
        total_sell_size = sum(r['position_size'] for r in results if 'SELL' in r['signal'])
        
        print(f"   Total Buy Exposure:  {total_buy_size:.1%}")
        print(f"   Total Sell Exposure: {total_sell_size:.1%}")
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"   Average Confidence:  {avg_confidence:.1%}")


if __name__ == "__main__":
    main()
