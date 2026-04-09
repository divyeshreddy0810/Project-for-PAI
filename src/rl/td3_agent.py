"""
TD3 Agent — Twin Delayed DDPG for Discrete Action Trading
----------------------------------------------------------
Key improvements over SAC for volatile, news-driven AI-era stocks:
  - Deterministic policy: no entropy stochasticity → commits to best action
  - Delayed actor updates (every 2 critic steps): prevents policy oscillation
  - Target policy smoothing: Gumbel noise on target logits → reduces Q overestimation
  - Separate critic backbones: avoids actor/critic feature coupling

Same interface as SACAgent:
  agent.net                          → state_dict for save/load
  agent.select_action(state, deterministic=True) → int 0/1/2
  agent.store(s, a, r, ns, done)
  agent.update()
  agent.save(path) / agent.load(path)

Reference: Fujimoto et al. (2018) — "Addressing Function Approximation
           Error in Actor-Critic Methods"
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random


# ── Replay Buffer ─────────────────────────────────────────────────────
class ReplayBuffer:
    """Identical to SAC replay buffer."""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int64),
                np.array(r,  dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ── Network ───────────────────────────────────────────────────────────
class TD3Network(nn.Module):
    """
    Actor + twin critics in one module.
    Mirrors SACNetwork interface so save/load is identical.

    TD3 difference: critics have separate backbones from the actor.
    This prevents the actor loss gradient from corrupting Q-value features,
    which matters on volatile assets where Q estimates can swing wildly.
    """
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()

        def _block():
            return nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden, 64),
                nn.LayerNorm(64), nn.GELU(),
            )

        # Actor backbone + head
        self.backbone = _block()
        self.actor    = nn.Linear(64, action_dim)

        # Twin critic backbones + heads (separate from actor)
        self.c1_backbone = _block()
        self.q1          = nn.Linear(64, action_dim)
        self.c2_backbone = _block()
        self.q2          = nn.Linear(64, action_dim)

    def get_actor(self, x):
        return self.actor(self.backbone(x))

    def get_q(self, x):
        return (self.q1(self.c1_backbone(x)),
                self.q2(self.c2_backbone(x)))


# ── Agent ─────────────────────────────────────────────────────────────
class TD3Agent:
    """
    Discrete Twin Delayed DDPG.

    Why better than SAC for SMCI / ARM / IREN / MU style stocks:
      SAC maximises entropy → policy deliberately keeps uncertainty high.
      On news-driven stocks that flip regime fast, that uncertainty becomes
      noise. TD3 is deterministic — once it has a view, it acts on it.
    """
    def __init__(self, state_dim, action_dim=3,
                 lr=3e-4, gamma=0.99,
                 tau=0.005,
                 policy_delay=2,
                 noise_scale=0.1,
                 buffer_size=50000,
                 batch_size=64,
                 device="cuda"):

        self.device       = torch.device(device)
        self.gamma        = gamma
        self.tau          = tau
        self.policy_delay = policy_delay
        self.noise_scale  = noise_scale
        self.batch_size   = batch_size
        self.action_dim   = action_dim
        self._step        = 0

        # Networks
        self.net        = TD3Network(state_dim, action_dim).to(self.device)
        self.target_net = TD3Network(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())

        # Separate optimisers — actor and critics update at different rates
        actor_params = (list(self.net.backbone.parameters()) +
                        list(self.net.actor.parameters()))
        critic_params = (list(self.net.c1_backbone.parameters()) +
                         list(self.net.q1.parameters()) +
                         list(self.net.c2_backbone.parameters()) +
                         list(self.net.q2.parameters()))

        self.actor_optim  = torch.optim.Adam(actor_params,  lr=lr)
        self.critic_optim = torch.optim.Adam(critic_params, lr=lr)

        self.buffer  = ReplayBuffer(buffer_size)
        self.trained = False

    def select_action(self, state, deterministic=False):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.net.get_actor(s)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                # Exploration: Gumbel noise on logits (discrete equivalent of
                # Gaussian noise on continuous actions in original TD3)
                noise  = torch.randn_like(logits) * self.noise_scale
                action = (logits + noise).argmax(dim=-1)
        return int(action.cpu())

    def store(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s  = torch.tensor(s).to(self.device)
        a  = torch.tensor(a).to(self.device)
        r  = torch.tensor(r).to(self.device)
        ns = torch.tensor(ns).to(self.device)
        d  = torch.tensor(d).to(self.device)

        self._step += 1

        # ── Critic update (every step) ─────────────────────────────
        with torch.no_grad():
            # Target policy smoothing: add noise to target logits
            tgt_logits  = self.target_net.get_actor(ns)
            noise       = torch.randn_like(tgt_logits) * self.noise_scale
            next_action = (tgt_logits + noise).argmax(dim=-1)          # (B,)

            tq1, tq2    = self.target_net.get_q(ns)
            min_tq      = torch.min(tq1, tq2)                          # (B, A)
            target_val  = min_tq.gather(1, next_action.unsqueeze(1)).squeeze()
            target_q    = r + self.gamma * (1 - d) * target_val

        q1, q2    = self.net.get_q(s)
        q1_a      = q1.gather(1, a.unsqueeze(1)).squeeze()
        q2_a      = q2.gather(1, a.unsqueeze(1)).squeeze()
        crit_loss = (nn.functional.mse_loss(q1_a, target_q) +
                     nn.functional.mse_loss(q2_a, target_q))

        self.critic_optim.zero_grad()
        crit_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.net.c1_backbone.parameters()) +
            list(self.net.q1.parameters()) +
            list(self.net.c2_backbone.parameters()) +
            list(self.net.q2.parameters()), 1.0)
        self.critic_optim.step()

        # ── Delayed actor update (every policy_delay steps) ────────
        actor_loss_val = 0.0
        if self._step % self.policy_delay == 0:
            logits     = self.net.get_actor(s)
            q1_cur, _  = self.net.get_q(s)
            # Actor loss: maximise expected Q over softmax distribution
            probs      = torch.softmax(logits, dim=-1)
            actor_loss = -(probs * q1_cur).sum(-1).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.net.backbone.parameters()) +
                list(self.net.actor.parameters()), 1.0)
            self.actor_optim.step()
            actor_loss_val = actor_loss.item()

            # Soft target update — only when actor updates (TD3 standard)
            for p, tp in zip(self.net.parameters(),
                             self.target_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self.trained = True
        return {"critic_loss": crit_loss.item(),
                "actor_loss":  actor_loss_val}

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.net.state_dict())
