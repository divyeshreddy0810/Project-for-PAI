"""
PPO Agent for Financial Trading
--------------------------------
Proximal Policy Optimization with Actor-Critic architecture.
Actions: BUY (0), SELL (1), HOLD (2)
State: Multimodal signals + portfolio state
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Shared backbone → separate actor (policy) and critic (value) heads.
    Actor:  outputs action probabilities
    Critic: outputs state value estimate
    """

    def __init__(self, state_dim: int, action_dim: int = 3):
        super().__init__()

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Actor head — policy (what action to take)
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, action_dim),
        )

        # Critic head — value function (how good is this state)
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Initialise weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.backbone(x)
        logits   = self.actor(features)
        value    = self.critic(features).squeeze(-1)
        return logits, value

    def get_action(self, state, deterministic=False):
        """Sample action from policy."""
        logits, value = self.forward(state)
        dist   = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value


class PPOAgent:
    """
    PPO Agent — trains Actor-Critic policy on trading environment.
    """

    def __init__(self, state_dim: int, action_dim: int = 3,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.02,
                 max_grad_norm: float = 0.5,
                 n_steps: int = 256,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 device: str = "cuda"):

        self.device        = torch.device(device)
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_epsilon  = clip_epsilon
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps       = n_steps
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size

        self.policy    = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100)

        # Rollout buffer
        self._reset_buffer()

    def _reset_buffer(self):
        self.buf_states   = []
        self.buf_actions  = []
        self.buf_logprobs = []
        self.buf_rewards  = []
        self.buf_values   = []
        self.buf_dones    = []

    def select_action(self, state: np.ndarray, deterministic=False):
        """Select action given state."""
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.policy.get_action(s, deterministic)
        return int(action.cpu()), float(log_prob.cpu()), float(value.cpu())

    def store(self, state, action, log_prob, reward, value, done):
        """Store one transition in rollout buffer."""
        self.buf_states.append(state)
        self.buf_actions.append(action)
        self.buf_logprobs.append(log_prob)
        self.buf_rewards.append(reward)
        self.buf_values.append(value)
        self.buf_dones.append(done)

    def _compute_gae(self, last_value: float):
        """Generalised Advantage Estimation."""
        rewards  = np.array(self.buf_rewards,  dtype=np.float32)
        values   = np.array(self.buf_values,   dtype=np.float32)
        dones    = np.array(self.buf_dones,    dtype=np.float32)

        T         = len(rewards)
        advantages= np.zeros(T, dtype=np.float32)
        last_gae  = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_value
            else:
                next_val = values[t + 1]

            delta     = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae  = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, last_value: float = 0.0):
        """Update policy using collected rollout."""
        if len(self.buf_states) < self.batch_size:
            self._reset_buffer()
            return {}

        advantages, returns = self._compute_gae(last_value)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states   = torch.tensor(np.array(self.buf_states),  dtype=torch.float32).to(self.device)
        actions  = torch.tensor(np.array(self.buf_actions), dtype=torch.long).to(self.device)
        old_lp   = torch.tensor(np.array(self.buf_logprobs),dtype=torch.float32).to(self.device)
        advs     = torch.tensor(advantages,                 dtype=torch.float32).to(self.device)
        rets     = torch.tensor(returns,                    dtype=torch.float32).to(self.device)

        self._reset_buffer()

        # Multiple epochs over same rollout
        total_loss = 0.0
        for _ in range(self.n_epochs):
            # Mini-batch shuffle
            idx = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                mb_idx = idx[start:start+self.batch_size]
                if len(mb_idx) < 2:
                    continue

                mb_s  = states[mb_idx]
                mb_a  = actions[mb_idx]
                mb_lp = old_lp[mb_idx]
                mb_adv= advs[mb_idx]
                mb_ret= rets[mb_idx]

                logits, values = self.policy(mb_s)
                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(mb_a)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio    = (new_lp - mb_lp).exp()
                clip_r   = ratio.clamp(1 - self.clip_epsilon,
                                       1 + self.clip_epsilon)
                pol_loss = -torch.min(ratio * mb_adv,
                                      clip_r * mb_adv).mean()

                # Value loss
                val_loss = nn.functional.mse_loss(values, mb_ret)

                # Combined loss
                loss = pol_loss + self.value_coef * val_loss \
                                - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()

        self.scheduler.step()
        return {"loss": total_loss}

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)
        print(f"  💾 PPO agent saved → {path}")

    def load(self, path: str):
        self.policy.load_state_dict(
            torch.load(path, map_location=self.device))
        print(f"  📂 PPO agent loaded ← {path}")
