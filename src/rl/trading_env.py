"""
Trading Environment for PPO Agent
----------------------------------
Replays historical price data as a gym-like environment.
State: multimodal signals from PatchTST + HMM + LightGBM + ADX + portfolio.
Actions: 0=BUY, 1=SELL, 2=HOLD
Reward: Sharpe-based (punishes volatility, rewards consistent returns).
"""

import numpy as np
import pandas as pd
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TradingEnvironment:
    """
    Financial trading environment for RL training.
    Wraps historical OHLCV data + precomputed signals into
    a step-by-step environment the PPO agent can learn from.
    """

    # Actions
    BUY  = 0
    SELL = 1
    HOLD = 2

    def __init__(self,
                 df: pd.DataFrame,
                 signals: np.ndarray,
                 initial_capital: float = 100_000.0,
                 transaction_cost: float = 0.001,
                 reward_window: int = 20):
        """
        Args:
            df:               OHLCV dataframe (aligned with signals)
            signals:          Precomputed multimodal signals — shape (T, n_signals)
                              Each row: [ptst_ret, ptst_prob, hmm, adx, atr,
                                         lgbm_ret, lgbm_prob, vol, mom5, mom10]
            initial_capital:  Starting portfolio value
            transaction_cost: 0.001 = 10 bps per trade
            reward_window:    Rolling window for Sharpe reward
        """
        self.df               = df.reset_index(drop=True)
        self.signals          = signals.astype(np.float32)
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.reward_window    = reward_window

        # State dimension = signals + portfolio state
        # signals: 10 features
        # portfolio: [position, unrealised_pnl_pct, days_in_trade]
        self.n_signals    = signals.shape[1]
        self.state_dim    = self.n_signals + 3
        self.action_dim   = 3  # BUY, SELL, HOLD

        self.reset()

    def reset(self):
        """Reset environment to start of data."""
        self.t              = 0
        self.capital        = self.initial_capital
        self.position       = 0      # 1=long, 0=flat, -1=short
        self.entry_price    = 0.0
        self.days_in_trade  = 0
        self.returns_hist   = []
        self.portfolio_val  = [self.initial_capital]
        self.trade_count    = 0
        return self._get_state()

    def _get_state(self):
        """Build state vector from current signals + portfolio info."""
        if self.t >= len(self.signals):
            self.t = len(self.signals) - 1

        sig = self.signals[self.t].copy()

        current_price = float(self.df["Close"].iloc[self.t])
        unrealised_pnl = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealised_pnl = (current_price - self.entry_price) / self.entry_price
            if self.position == -1:
                unrealised_pnl = -unrealised_pnl

        portfolio_features = np.array([
            float(self.position),
            float(unrealised_pnl),
            float(min(self.days_in_trade, 30) / 30.0),  # normalise to 0-1
        ], dtype=np.float32)

        state = np.concatenate([sig, portfolio_features])
        return state

    def step(self, action: int):
        """
        Execute one trading step.

        Returns: (next_state, reward, done, info)
        """
        if self.t >= len(self.signals) - 1:
            return self._get_state(), 0.0, True, {}

        current_price = float(self.df["Close"].iloc[self.t])
        next_price    = float(self.df["Close"].iloc[self.t + 1])
        price_return  = (next_price - current_price) / current_price

        # ── Execute action ─────────────────────────────────────
        prev_position  = self.position
        trade_cost     = 0.0

        if action == self.BUY and self.position != 1:
            self.position  = 1
            self.entry_price = current_price
            self.days_in_trade = 0
            if prev_position != 0:
                trade_cost = self.transaction_cost  # closing previous
            trade_cost += self.transaction_cost     # opening new
            self.trade_count += 1

        elif action == self.SELL and self.position != -1:
            self.position  = -1
            self.entry_price = current_price
            self.days_in_trade = 0
            if prev_position != 0:
                trade_cost = self.transaction_cost
            trade_cost += self.transaction_cost
            self.trade_count += 1

        # HOLD — keep current position

        # ── Calculate step return ──────────────────────────────
        if self.position == 1:
            step_return = price_return - trade_cost
        elif self.position == -1:
            step_return = -price_return - trade_cost
        else:
            step_return = -trade_cost  # flat — only pay cost if traded

        self.returns_hist.append(step_return)
        self.days_in_trade += 1

        # Update capital
        self.capital *= (1 + step_return)
        self.portfolio_val.append(self.capital)

        # ── Sharpe-based reward ────────────────────────────────
        # Reward = rolling Sharpe over last N steps
        # This teaches the agent to seek CONSISTENT returns, not gambles
        if len(self.returns_hist) >= self.reward_window:
            recent = np.array(self.returns_hist[-self.reward_window:])
            mean_r = np.mean(recent)
            std_r  = np.std(recent) + 1e-8
            reward = float(mean_r / std_r)
        else:
            # Simple return reward early in episode
            reward = float(step_return * 100)

        # Bonus for profitable trades, penalty for ruin
        if self.capital < self.initial_capital * 0.5:
            reward -= 10.0  # ruin penalty
            done = True
        else:
            done = False

        self.t += 1

        info = {
            "capital":     self.capital,
            "position":    self.position,
            "price_return":price_return,
            "step_return": step_return,
            "trade_count": self.trade_count,
        }

        return self._get_state(), reward, done, info

    def get_performance(self):
        """Calculate final performance metrics."""
        vals = np.array(self.portfolio_val)
        returns = np.diff(vals) / vals[:-1]

        total_return = (self.capital - self.initial_capital) / self.initial_capital
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        peak    = np.maximum.accumulate(vals)
        dd      = (vals - peak) / peak
        max_dd  = float(np.min(dd))

        return {
            "total_return":  float(total_return),
            "sharpe_ratio":  float(sharpe),
            "max_drawdown":  float(max_dd),
            "final_capital": float(self.capital),
            "trade_count":   int(self.trade_count),
        }


def build_signals(df: pd.DataFrame,
                  patchtst_model=None,
                  lgbm_model=None,
                  hmm_model=None) -> np.ndarray:
    """
    Build the multimodal signal matrix for the environment.
    Each row = one timestep's signals.
    Falls back to technical signals if models not available.
    """
    from src.forecast.patchtst_forecast import build_features

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    n = len(df)
    signals = np.zeros((n, 10), dtype=np.float32)

    # Technical features (always available)
    ret  = df["Close"].pct_change().fillna(0).values
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = (100 - (100 / (1 + gain / (loss + 1e-8)))).fillna(50).values / 100.0

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd  = ((ema12 - ema26) / df["Close"]).fillna(0).values

    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"] -df["Close"].shift()).abs()], axis=1).max(axis=1)
    atr = (tr.rolling(14).mean() / df["Close"]).fillna(0).values

    vol   = pd.Series(ret).rolling(20).std().fillna(0.01).values
    mom5  = df["Close"].pct_change(5).fillna(0).values
    mom10 = df["Close"].pct_change(10).fillna(0).values

    # ADX
    pdm = df["High"].diff().clip(lower=0)
    ndm = (-df["Low"].diff()).clip(lower=0)
    pdm[pdm < ndm] = 0; ndm[ndm < pdm] = 0
    atr_s = tr.rolling(14).mean()
    pdi   = 100 * (pdm.rolling(14).mean() / (atr_s + 1e-8))
    ndi   = 100 * (ndm.rolling(14).mean() / (atr_s + 1e-8))
    dx    = 100 * ((pdi-ndi).abs() / (pdi+ndi+1e-8))
    adx   = dx.rolling(14).mean().fillna(20).values / 100.0

    # Fill signal matrix
    signals[:, 0] = ret          # PatchTST proxy (returns)
    signals[:, 1] = (ret > 0).astype(float)  # up probability proxy
    signals[:, 2] = np.where(ret > 0.001, 1,
                    np.where(ret < -0.001, -1, 0))  # HMM proxy
    signals[:, 3] = adx
    signals[:, 4] = atr
    signals[:, 5] = ret          # LightGBM proxy
    signals[:, 6] = rsi
    signals[:, 7] = vol
    signals[:, 8] = mom5
    signals[:, 9] = mom10

    # If real models provided — override with actual predictions
    if patchtst_model is not None:
        LOOKBACK = 60
        for i in range(LOOKBACK, n):
            try:
                window_df = df.iloc[i-LOOKBACK:i]
                ret_pred  = patchtst_model.predict_return(window_df)
                signals[i, 0] = ret_pred
                signals[i, 1] = 1.0 if ret_pred > 0 else 0.0
            except Exception:
                pass

    return signals
