"""
RL Trainer — trains PPO agent and compares vs rule-based trader
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from src.rl.trading_env import TradingEnvironment, build_signals
from src.rl.ppo_agent   import PPOAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_ppo(df: pd.DataFrame,
              n_episodes: int = 200,
              n_steps:    int = 256,
              verbose:    bool = True) -> PPOAgent:
    """Train PPO agent on historical data."""

    signals = build_signals(df)
    env     = TradingEnvironment(df, signals)
    agent   = PPOAgent(state_dim=env.state_dim, device=DEVICE,
                       n_steps=n_steps)

    if verbose:
        print(f"  🤖 Training PPO on {DEVICE}...")
        print(f"  State dim: {env.state_dim}  Actions: {env.action_dim}")
        print(f"  Episodes: {n_episodes}  Steps/ep: {n_steps}")

    best_sharpe = -999
    best_state  = None

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0
        steps = 0

        while True:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store(state, action, log_prob, reward, value, done)
            state      = next_state
            ep_reward += reward
            steps     += 1

            # Update every n_steps
            if steps % n_steps == 0:
                _, _, last_val = agent.select_action(state)
                agent.update(last_val)

            if done:
                break

        perf = env.get_performance()

        # Save best model
        if perf["sharpe_ratio"] > best_sharpe:
            best_sharpe = perf["sharpe_ratio"]
            best_state  = {k: v.cpu().clone()
                           for k, v in agent.policy.state_dict().items()}

        if verbose and (ep+1) % 50 == 0:
            print(f"    Ep {ep+1:3d}/{n_episodes}  "
                  f"return={perf['total_return']:+.2%}  "
                  f"sharpe={perf['sharpe_ratio']:+.2f}  "
                  f"trades={perf['trade_count']}")

    # Restore best weights
    if best_state:
        agent.policy.load_state_dict(
            {k: v.to(agent.device) for k, v in best_state.items()})

    if verbose:
        print(f"  ✅ PPO trained  best_sharpe={best_sharpe:.3f}")

    return agent


def evaluate(agent: PPOAgent,
             df: pd.DataFrame,
             label: str = "PPO") -> dict:
    """Evaluate agent on test data."""
    signals = build_signals(df)
    env     = TradingEnvironment(df, signals)
    state   = env.reset()

    while True:
        action, _, _ = agent.select_action(state, deterministic=True)
        state, _, done, _ = env.step(action)
        if done:
            break

    perf = env.get_performance()
    perf["label"] = label
    return perf


def rule_based_baseline(df: pd.DataFrame) -> dict:
    """
    Simple rule-based trader for comparison:
    BUY when RSI < 40 and price above SMA20
    SELL when RSI > 60 and price below SMA20
    HOLD otherwise
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    close = df["Close"]
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-8)))
    sma20 = close.rolling(20).mean()

    capital   = 100_000.0
    position  = 0
    tc        = 0.001
    returns   = []

    for i in range(20, len(df)-1):
        px      = float(close.iloc[i])
        nx      = float(close.iloc[i+1])
        r_val   = float(rsi.iloc[i])
        s_val   = float(sma20.iloc[i])
        pr      = (nx - px) / px

        if r_val < 40 and px > s_val and position != 1:
            if position == -1:
                capital *= (1 - tc)
            capital *= (1 - tc)
            position = 1
        elif r_val > 60 and px < s_val and position != -1:
            if position == 1:
                capital *= (1 - tc)
            capital *= (1 - tc)
            position = -1

        if position == 1:
            step_r = pr
        elif position == -1:
            step_r = -pr
        else:
            step_r = 0.0

        capital *= (1 + step_r)
        returns.append(step_r)

    returns = np.array(returns)
    total   = (capital - 100_000) / 100_000
    sharpe  = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252) \
              if len(returns) > 1 else 0.0
    vals    = 100_000 * np.cumprod(1 + returns)
    peak    = np.maximum.accumulate(vals)
    max_dd  = float(np.min((vals - peak) / (peak + 1e-8)))

    return {
        "label":         "Rule-Based",
        "total_return":  float(total),
        "sharpe_ratio":  float(sharpe),
        "max_drawdown":  float(max_dd),
        "final_capital": float(capital),
    }


if __name__ == "__main__":
    ASSETS = [
        ("^GSPC",   "S&P 500",  "equity"),
        ("BTC-USD", "Bitcoin",  "crypto"),
        ("GC=F",    "Gold",     "commodity"),
    ]

    print("="*65)
    print("  PPO RL AGENT vs RULE-BASED TRADER")
    print("  Train: 70% data  Test: last 30%")
    print("="*65)

    all_results = []

    for sym, name, aclass in ASSETS:
        print(f"\n── {name} ({sym}) ──────────────────────────")
        df = yf.download(sym, start="2016-01-01",
                         end="2026-03-25", progress=False)
        if hasattr(df.columns, "droplevel"):
            df.columns = df.columns.droplevel(1)
        df = df.dropna()
        print(f"  Data: {len(df)} rows")

        # Train/test split
        n_train = int(len(df) * 0.70)
        df_tr   = df.iloc[:n_train].reset_index(drop=True)
        df_te   = df.iloc[n_train:].reset_index(drop=True)

        # Train PPO
        agent = train_ppo(df_tr, n_episodes=200, verbose=True)

        # Save model
        os.makedirs("data/models", exist_ok=True)
        agent.save(f"data/models/ppo_{sym.replace('^','').replace('-','_')}.pt")

        # Evaluate both on TEST data
        ppo_perf  = evaluate(agent, df_te, label="PPO")
        rule_perf = rule_based_baseline(df_te)

        print(f"\n  TEST RESULTS ({name}):")
        for perf in [ppo_perf, rule_perf]:
            print(f"  {perf['label']:<15} "
                  f"return={perf['total_return']:+.2%}  "
                  f"sharpe={perf['sharpe_ratio']:+.2f}  "
                  f"dd={perf['max_drawdown']:.2%}")

        all_results.append((name, ppo_perf, rule_perf))

    # Final summary
    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Asset':<12} {'PPO Return':>12} {'Rule Return':>12} "
          f"{'PPO Sharpe':>12} {'Winner':>10}")
    print(f"  {'─'*60}")
    ppo_wins = 0
    for name, ppo, rule in all_results:
        winner = "PPO ✅" if ppo["sharpe_ratio"] > rule["sharpe_ratio"] \
                 else "Rule ✅"
        if "PPO" in winner:
            ppo_wins += 1
        print(f"  {name:<12} "
              f"{ppo['total_return']:>+11.2%}  "
              f"{rule['total_return']:>+11.2%}  "
              f"{ppo['sharpe_ratio']:>+11.2f}  "
              f"{winner}")
    print(f"\n  PPO wins: {ppo_wins}/{len(all_results)}")
