#!/usr/bin/env python3
"""
Master Pipeline Orchestrator (Improved)
---------------------------------------
Runs the complete AI-driven trading system in sequence with user input
collected once at the start and passed to all stages.

Pipeline:
1. Sentiment Analysis (sentiment_analyzer.py)
2. Technical Indicators (technical_indicators.py)
3. Market Regime Classification (market_regime_model.py)
4. Price Forecasting (price_forecaster.py)
5. Trading Signal Generation (rl_trader.py)
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    from src.utils.config_manager import ConfigManager
except ImportError:
    print("⚠️  config_manager.py not found. Creating fallback...")
    ConfigManager = None


# ========================== CONFIGURATION ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data/output")

PIPELINE_STEPS = [
    {
        'name': 'Sentiment Analysis',
        'script': 'src/sentiment_analyzer.py',
        'icon': '📰',
        'description': 'Analyze financial news sentiment using FinBERT',
        'required': True,
        'interactive': False,
        'pass_config': True
    },
    {
        'name': 'Technical Indicators',
        'script': 'src/technical_indicators.py',
        'icon': '📊',
        'description': 'Calculate technical indicators and regime predictions',
        'required': True,
        'interactive': True,
        'pass_config': True
    },
    {
        'name': 'Market Regime Model',
        'script': 'src/market_regime_model.py',
        'icon': '🎯',
        'description': 'Classify market regime (Bull/Bear/Volatile)',
        'required': True,
        'interactive': False,
        'pass_config': True
    },
    {
        'name': 'Price Forecaster',
        'script': 'src/price_forecaster.py',
        'icon': '💰',
        'description': 'Predict future prices using ML models',
        'required': True,
        'interactive': False,
        'pass_config': True
    },
    {
        'name': 'Trading Signal Generator',
        'script': 'src/rl_trader.py',
        'icon': '🤖',
        'description': 'Generate buy/sell/hold trading signals',
        'required': True,
        'interactive': True,
        'pass_config': True
    }
]


class PipelineOrchestrator:
    """Orchestrates the complete trading system pipeline"""
    
    def __init__(self, base_dir: str = BASE_DIR):
        """Initialize orchestrator"""
        self.base_dir = base_dir
        self.output_dir = OUTPUT_DIR
        self.execution_log = []
        self.step_results = {}
        self.config = {}
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def print_header(self):
        """Print pipeline header"""
        print("\n" + "=" * 80)
        print(" " * 20 + "🚀 AI-DRIVEN TRADING PIPELINE 🚀")
        print("=" * 80)
    
    def print_pipeline_overview(self):
        """Print pipeline overview"""
        print("\n" + "-" * 80)
        print("PIPELINE STEPS:")
        print("-" * 80)
        
        for i, step in enumerate(PIPELINE_STEPS, 1):
            status = "✓ Required" if step['required'] else "○ Optional"
            interact = " [Interactive]" if step['interactive'] else ""
            print(f"\n{i}. {step['icon']} {step['name']}{interact}")
            print(f"   {step['description']}")
            print(f"   Script: {step['script']}")
    
    def collect_configuration(self):
        """Collect user configuration once"""
        print("\n" + "=" * 80)
        print("📋 PIPELINE CONFIGURATION")
        print("=" * 80)
        
        if ConfigManager:
            manager = ConfigManager()
            self.config = manager.collect_user_input()
            manager.save_config()
        else:
            # Fallback: ask for simple input
            symbols = input("\nEnter symbols (comma-separated, e.g., '^GSPC,BTC-USD'): ").strip()
            window = input("Enter time window (1d, 1w, 1mo, 3mo, 6mo, 1y) [default: 1mo]: ").strip() or "1mo"
            
            if not symbols:
                symbols = "^GSPC,BTC-USD"
            
            self.config = {
                'timestamp': datetime.now().isoformat(),
                'symbols': symbols.split(','),
                'window': window,
            }
    
    def run_step(self, step: Dict[str, Any], step_number: int, total_steps: int) -> bool:
        """
        Run a single pipeline step with configuration passed as CLI arguments
        
        Args:
            step: Step configuration from PIPELINE_STEPS
            step_number: Current step number
            total_steps: Total number of steps
        
        Returns:
            True if successful, False otherwise
        """
        script_path = os.path.join(self.base_dir, step['script'])
        
        if not os.path.exists(script_path):
            print(f"\n❌ {step['name']}: Script not found ({step['script']})")
            return False
        
        print(f"\n{'='*80}")
        print(f"[{step_number}/{total_steps}] {step['icon']} {step['name']}")
        print(f"{'='*80}")
        print(f"Script: {step['script']}")
        print(f"Status: Starting...")
        
        try:
            start_time = datetime.now()
            
            # Build command with config arguments
            cmd = [sys.executable, script_path]
            
            if step['pass_config'] and self.config:
                symbols = ",".join(self.config.get('symbols', []))
                window = self.config.get('window', '1mo')
                cmd.extend(['--symbols', symbols, '--window', window, '--non-interactive'])
            
            # Run script
            if step['interactive']:
                # Interactive mode - direct user interaction
                print(f"⏳ Running (interactive - waiting for your input if needed)...\n")
                process = subprocess.Popen(cmd, cwd=self.base_dir)
                try:
                    returncode = process.wait(timeout=900)  # 15 minute timeout
                    if returncode != 0 and returncode != 130:  # 130 is Ctrl+C
                        print(f"⚠️  {step['name']}: Exited with code {returncode}")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  {step['name']}: Timeout (>15 minutes). Continuing.")
                    process.terminate()
            else:
                # Non-interactive mode with timeout
                print(f"⏳ Running (non-interactive)...\n")
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=self.base_dir,
                        timeout=600  # 10 minute timeout
                    )
                    
                    # Print output
                    if result.stdout:
                        print(result.stdout)
                    
                    if result.stderr:
                        print(f"⚠️  Warnings:\n{result.stderr}")
                    
                    if result.returncode != 0:
                        print(f"❌ {step['name']}: Failed with code {result.returncode}")
                        return False
                    
                except subprocess.TimeoutExpired:
                    print(f"❌ {step['name']}: Timeout (exceeded 10 minutes)")
                    return False
            
            elapsed = datetime.now() - start_time
            print(f"\n✅ {step['name']}: Completed in {elapsed.total_seconds():.1f}s")
            
            self.execution_log.append({
                'step': step['name'],
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'duration': elapsed.total_seconds()
            })
            
            return True
        
        except KeyboardInterrupt:
            print(f"\n⚠️  {step['name']}: Interrupted by user")
            self.execution_log.append({
                'step': step['name'],
                'status': 'interrupted',
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    def run_pipeline(self, mode: str = 'full'):
        """Run pipeline"""
        self.print_header()
        self.print_pipeline_overview()
        
        # Collect configuration once
        self.collect_configuration()
        
        print("\n" + "=" * 80)
        print("🎯 PIPELINE EXECUTION STARTING...")
        print("=" * 80)
        print(f"📊 Configuration:")
        print(f"   Symbols: {', '.join(self.config.get('symbols', []))}")
        print(f"   Window: {self.config.get('window', '1mo')}")
        print(f"   Timestamp: {self.config.get('timestamp', 'N/A')}")
        
        # Run steps
        steps_to_run = [s for s in PIPELINE_STEPS if s['required']]
        if mode != 'full':
            # Could implement partial pipeline modes here
            pass
        
        try:
            for i, step in enumerate(steps_to_run, 1):
                success = self.run_step(step, i, len(steps_to_run))
                if not success and step['required']:
                    print(f"\n❌ Pipeline halted: Required step '{step['name']}' failed")
                    break
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        
        finally:
            # Print summary
            self.print_summary()
    
    def print_summary(self):
        """Print execution summary"""
        print("\n" + "=" * 80)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        
        if not self.execution_log:
            print("No steps executed.")
            return
        
        for entry in self.execution_log:
            status_icon = "✅" if entry['status'] == 'success' else "⚠️ " if entry['status'] == 'interrupted' else "❌"
            duration = f" ({entry['duration']:.1f}s)" if 'duration' in entry else ""
            print(f"{status_icon} {entry['step']}{duration}")
        
        total_duration = sum(e.get('duration', 0) for e in self.execution_log)
        print(f"\n📈 Total execution time: {total_duration:.1f}s")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Check the output files in {self.output_dir}")
        print(f"  2. Review the generated trading signals")
        print(f"  3. Backtest the predictions on historical data (optional)")
        print("=" * 80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI-Driven Trading Pipeline Orchestrator"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'test'],
        help='Execution mode'
    )
    args = parser.parse_args()
    
    try:
        orchestrator = PipelineOrchestrator()
        orchestrator.run_pipeline(mode=args.mode)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
