#!/usr/bin/env python3
"""
Configuration Manager
---------------------
Handles user input collection and configuration management for the entire pipeline.
Asks for input once and passes it to all stages.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


# Predefined assets
ALL_ASSETS = [
    {"index": 1, "symbol": "^GSPC", "name": "S&P 500", "type": "index"},
    {"index": 2, "symbol": "^IXIC", "name": "NASDAQ Composite", "type": "index"},
    {"index": 3, "symbol": "AAPL", "name": "Apple Inc.", "type": "stock"},
    {"index": 4, "symbol": "MSFT", "name": "Microsoft Corp.", "type": "stock"},
    {"index": 5, "symbol": "GOOGL", "name": "Alphabet Inc.", "type": "stock"},
    {"index": 6, "symbol": "BTC-USD", "name": "Bitcoin", "type": "crypto"},
    {"index": 7, "symbol": "ETH-USD", "name": "Ethereum", "type": "crypto"},
    {"index": 8, "symbol": "EURUSD=X", "name": "EUR/USD", "type": "forex"},
]

# Time window options
WINDOW_OPTIONS = {
    "1d": ("1 day", 1),
    "1w": ("1 week", 7),
    "1mo": ("1 month", 30),
    "3mo": ("3 months", 90),
    "6mo": ("6 months", 180),
    "1y": ("1 year", 365),
    "all": ("All available data", 1000)
}

WINDOW_OPTIONS_LIST = ["1d", "1w", "1mo", "3mo", "6mo", "1y", "all"]


def parse_asset_selection(inp: str) -> List[int]:
    """Parse asset selection input (e.g., '1,3,5-7')"""
    indices = set()
    valid_indices = {a['index'] for a in ALL_ASSETS}
    
    for part in inp.split(','):
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                indices.update(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                indices.add(int(part))
            except ValueError:
                continue
    
    return sorted([i for i in indices if i in valid_indices])


def window_to_dates(window: str) -> Tuple[datetime, datetime]:
    """Convert window to date range"""
    if window not in WINDOW_OPTIONS:
        window = "1mo"
    
    to_date = datetime.now()
    days = WINDOW_OPTIONS[window][1]
    from_date = to_date - timedelta(days=days)
    
    return from_date, to_date


class ConfigManager:
    """Manages pipeline configuration"""
    
    def __init__(self):
        self.config = {}
        self.config_file = "data/output/pipeline_config.json"
        os.makedirs("data/output", exist_ok=True)
    
    def collect_user_input(self) -> Dict:
        """Collect user input interactively"""
        print("\n" + "="*80)
        print(" " * 20 + "🚀 AI-DRIVEN TRADING PIPELINE 🚀")
        print("="*80)
        
        # Display asset list
        print("\n📊 AVAILABLE ASSETS:")
        for a in ALL_ASSETS:
            print(f"  {a['index']}. {a['name']} ({a['symbol']}) [{a['type'].upper()}]")
        
        # Asset selection
        while True:
            inp = input("\n📝 Enter asset numbers (e.g., '1,3,5-7', 'all', or asset symbol like 'AAPL'): ").strip()
            if not inp:
                print("Please enter something.")
                continue
            
            # Check if user entered symbols directly
            if inp.upper() not in ['ALL']:
                # Check if it's comma-separated symbols
                parts = inp.split(',')
                selected_assets = []
                is_symbols = False
                
                for part in parts:
                    part = part.strip()
                    # Try to find by symbol
                    for asset in ALL_ASSETS:
                        if asset['symbol'].upper() == part.upper():
                            selected_assets.append(asset)
                            is_symbols = True
                            break
                
                if is_symbols and selected_assets:
                    break
            
            # Try parsing as indices
            selected_indices = parse_asset_selection(inp)
            if selected_indices:
                selected_assets = [a for a in ALL_ASSETS if a['index'] in selected_indices]
                break
            
            if inp.lower() == 'all':
                selected_assets = ALL_ASSETS
                break
                
            print("No valid assets selected. Try again.")
        
        print("\n✅ Selected assets:")
        for a in selected_assets:
            print(f"   - {a['name']} ({a['symbol']})")
        
        # Time window selection
        print("\n⏱️  TIME WINDOW OPTIONS:")
        for key, (desc, _) in WINDOW_OPTIONS.items():
            print(f"   {key}: {desc}")
        
        while True:
            window = input("\nEnter time window (default 1mo): ").strip()
            if not window:
                window = "1mo"
            if window in WINDOW_OPTIONS:
                break
            print("Invalid window. Choose from:", ", ".join(WINDOW_OPTIONS.keys()))
        
        from_date, to_date = window_to_dates(window)
        
        # Store configuration
        self.config = {
            "timestamp": datetime.now().isoformat(),
            "assets": selected_assets,
            "symbols": [a['symbol'] for a in selected_assets],
            "window": window,
            "from_date": from_date.strftime("%Y-%m-%d"),
            "to_date": to_date.strftime("%Y-%m-%d"),
            "from_date_iso": from_date.isoformat(),
            "to_date_iso": to_date.isoformat(),
        }
        
        print(f"\n⏱️  Analysis period: {self.config['from_date']} to {self.config['to_date']}")
        print(f"📊 Symbols: {', '.join(self.config['symbols'])}")
        
        return self.config
    
    def save_config(self) -> str:
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            # Save a JSON-serializable version
            config_to_save = self.config.copy()
            config_to_save['assets'] = self.config['assets']  # Keep as-is since it's already a list of dicts
            json.dump(config_to_save, f, indent=2)
        
        print(f"\n💾 Configuration saved to {self.config_file}")
        return self.config_file
    
    @staticmethod
    def load_config(config_file: str = "data/output/pipeline_config.json") -> Dict:
        """Load configuration from file"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return config
    
    @staticmethod
    def get_config_args(config: Dict) -> str:
        """Convert config to command-line arguments"""
        symbols = ",".join(config.get('symbols', []))
        window = config.get('window', '1mo')
        return f"--symbols={symbols} --window={window}"


if __name__ == "__main__":
    manager = ConfigManager()
    config = manager.collect_user_input()
    manager.save_config()
    
    print("\n✅ Configuration ready for pipeline execution!")
