"""
SAC Agent — Soft Actor-Critic for Financial Trading
------------------------------------------------------
Off-policy, maximum entropy RL.
More sample-efficient than PPO due to replay buffer.
Better exploration via entropy maximisation.

Reference: Haarnoja et al. (2018) — "Soft Actor-Critic"
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer — stores and samples transitions."""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,ns,d = zip(*batch)
        return (np.array(s,dtype=np.float32),
                np.array(a,dtype=np.int64),
                np.array(r,dtype=np.float32),
                np.array(ns,dtype=np.float32),
                np.array(d,dtype=np.float32))

    def __len__(self): return len(self.buffer)


class SACNetwork(nn.Module):
    """Shared backbone for SAC actor and critics."""
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.LayerNorm(64), nn.GELU(),
        )
        # Actor: outputs action logits
        self.actor = nn.Linear(64, action_dim)
        # Two Q-networks (double-Q trick reduces overestimation)
        self.q1 = nn.Linear(64, action_dim)
        self.q2 = nn.Linear(64, action_dim)

    def get_actor(self, x):
        return self.actor(self.backbone(x))

    def get_q(self, x):
        f = self.backbone(x)
        return self.q1(f), self.q2(f)


class SACAgent:
    """
    Discrete SAC with automatic entropy tuning.
    Off-policy: learns from replay buffer → more sample efficient than PPO.
    """
    def __init__(self, state_dim, action_dim=3,
                 lr=3e-4, gamma=0.99,
                 alpha=0.2,           # entropy coefficient
                 tau=0.005,           # soft target update
                 buffer_size=50000,
                 batch_size=64,
                 device="cuda"):

        self.device     = torch.device(device)
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.action_dim = action_dim

        # Networks
        self.net        = SACNetwork(state_dim, action_dim).to(self.device)
        self.target_net = SACNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())

        # Automatic entropy tuning
        self.log_alpha     = torch.tensor(
            np.log(alpha), dtype=torch.float32,
            requires_grad=True, device=self.device)
        self.target_entropy= -np.log(1.0/action_dim) * 0.98
        self.alpha_optim   = torch.optim.Adam([self.log_alpha], lr=1e-4)

        self.optim  = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.trained= False

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        s = torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.net.get_actor(s)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(
                    logits=logits).sample()
        return int(action.cpu())

    def store(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        s,a,r,ns,d = self.buffer.sample(self.batch_size)
        s  = torch.tensor(s).to(self.device)
        a  = torch.tensor(a).to(self.device)
        r  = torch.tensor(r).to(self.device)
        ns = torch.tensor(ns).to(self.device)
        d  = torch.tensor(d).to(self.device)

        with torch.no_grad():
            # Next action distribution
            next_logits = self.target_net.get_actor(ns)
            next_probs  = torch.softmax(next_logits, dim=-1)
            next_log_p  = torch.log_softmax(next_logits, dim=-1)

            # Target Q
            tq1, tq2    = self.target_net.get_q(ns)
            min_tq      = torch.min(tq1, tq2)
            # Soft Bellman target (entropy-augmented)
            v_next      = (next_probs * (min_tq - self.alpha * next_log_p)).sum(-1)
            target_q    = r + self.gamma * (1-d) * v_next

        # Critic loss
        q1, q2 = self.net.get_q(s)
        q1_a   = q1.gather(1, a.unsqueeze(1)).squeeze()
        q2_a   = q2.gather(1, a.unsqueeze(1)).squeeze()
        critic_loss = nn.functional.mse_loss(q1_a, target_q) + \
                      nn.functional.mse_loss(q2_a, target_q)

        # Actor loss (maximum entropy)
        logits   = self.net.get_actor(s)
        probs    = torch.softmax(logits, dim=-1)
        log_probs= torch.log_softmax(logits, dim=-1)
        q1_cur, q2_cur = self.net.get_q(s)
        min_q    = torch.min(q1_cur, q2_cur)
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(-1).mean()

        # Alpha loss (automatic entropy tuning)
        entropy    = -(probs * log_probs).sum(-1).mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach())

        # Update
        self.optim.zero_grad()
        (critic_loss + actor_loss).backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optim.step()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Soft update target network
        for p, tp in zip(self.net.parameters(),
                         self.target_net.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)

        self.trained = True
        return {"critic_loss": critic_loss.item(),
                "actor_loss":  actor_loss.item(),
                "alpha":       float(self.alpha)}

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(
            torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.net.state_dict())
