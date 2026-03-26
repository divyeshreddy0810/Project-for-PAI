# Frontend Integration Guide — Enhanced Pipeline

## Overview
Two modes available — toggle between Baseline and Enhanced.

## How it works

### Baseline mode (existing)
- Calls: `python3 scripts/master_pipeline_v2.py`
- Reads: `data/output/latest.json`
- Same as before — no changes needed

### Enhanced mode (new)
- Calls: `python3 scripts/enhanced_pipeline.py --symbols "^GSPC,BTC-USD,GC=F"`
- Reads: `data/output/enhanced_latest.json`
- Same JSON structure + extra fields

## Output format comparison

### Baseline output (latest.json)
```json
{
  "assets": [{
    "symbol": "^GSPC",
    "overall_mean": 0.55,
    "signal": "BUY"
  }]
}
```

### Enhanced output (enhanced_latest.json)
```json
{
  "pipeline": "enhanced",
  "portfolio": 100000,
  "risk_profile": "moderate",
  "assets": [{
    "symbol": "^GSPC",
    "current_price": 6556.37,
    "signal": "BUY",
    "confidence": 0.67,
    "position_size": 0.067,
    "position_value": 6700.00,
    "take_profit": 7540.00,
    "stop_loss": 6098.00,
    "predicted_price": 6557.62,
    "expected_return": 0.02,
    "regime": "bear",
    "rl_agent": "SAC",
    "votes": {"buy": 2, "sell": 0, "hold": 1},
    "signals": {
      "patchtst": "HOLD",
      "hmm": "SELL",
      "rl_agent": "BUY"
    },
    "enhanced_metrics": {
      "model": "PatchTST + HMM + SAC/PPO",
      "avg_backtest_sharpe": 0.744
    }
  }],
  "summary": {
    "buy_signals": 2,
    "sell_signals": 0,
    "hold_signals": 1,
    "avg_confidence": 0.67
  }
}
```

## What to add to the frontend

### 1. Toggle button
```javascript
const mode = 'baseline' // or 'enhanced'

function runPipeline(mode) {
  if (mode === 'baseline') {
    exec('python3 scripts/master_pipeline_v2.py')
    readFile('data/output/latest.json')
  } else {
    exec(`python3 scripts/enhanced_pipeline.py --symbols "${symbols}"`)
    readFile('data/output/enhanced_latest.json')
  }
}
```

### 2. New fields to display (enhanced mode only)
```javascript
// Show these extra fields when in enhanced mode:
asset.confidence        // "Confidence: 67%"
asset.position_value    // "Position: $6,700"
asset.take_profit       // "TP: $7,540"
asset.stop_loss         // "SL: $6,098"
asset.regime            // "Regime: BEAR"
asset.rl_agent          // "Agent: SAC"
asset.votes             // Show vote breakdown: BUY:2 HOLD:1

// Vote breakdown widget
const votes = asset.votes
// BUY: 2/3 votes → show green bar
// SELL: 0/3 votes
// HOLD: 1/3 votes
```

### 3. Which signal sources voted (enhanced mode only)
```javascript
const signals = asset.signals
// patchtst: "HOLD"  → PatchTST model
// hmm: "SELL"       → HMM regime
// rl_agent: "BUY"   → SAC/PPO agent
// Show as 3 small badges with colours
```

## Supported symbols
```javascript
const SUPPORTED = [
  "^GSPC",   // S&P 500  → SAC agent
  "^IXIC",   // NASDAQ   → SAC agent
  "BTC-USD", // Bitcoin  → PPO agent
  "GC=F",    // Gold     → SAC agent
  "EURUSD=X",// EUR/USD  → SAC agent
  "GBPUSD=X",// GBP/USD  → SAC agent
  // CL=F (Crude Oil) → NOT supported, system skips it
]
```

## Risk profiles
```
conservative → smaller positions, tighter stops
moderate     → balanced (default)
aggressive   → larger positions, wider stops
```

Pass as: `--risk conservative|moderate|aggressive`

## First run warning
First run per symbol trains the RL agent (~2 minutes per symbol).
After first run, model is saved to `data/models/` and loads instantly.
Show a loading spinner with message: "Training AI model for first use..."

## Files your frontend needs to know about
```
data/output/enhanced_latest.json  → read this after enhanced run
data/models/sac_GSPC.pt           → SAC model for S&P 500 (auto-saved)
data/models/ppo_BTC_USD.pt        → PPO model for Bitcoin (auto-saved)
data/models/sac_GCF.pt            → SAC model for Gold (auto-saved)
```
