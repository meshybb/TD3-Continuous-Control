"""
TD3: Twin Delayed Deep Deterministic Policy Gradient
Fujimoto et al., 2018 — https://arxiv.org/abs/1802.09477
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    """Uniform circular replay buffer stored as pre-allocated numpy arrays."""

    def __init__(self, state_dim: int, action_dim: int, max_size: int = 1_000_000):
        self.max_size = max_size
        self.ptr  = 0
        self.size = 0

        self.states      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.rewards     = np.zeros((max_size, 1),          dtype=np.float32)
        self.dones       = np.zeros((max_size, 1),          dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.states     [self.ptr] = state
        self.actions    [self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards    [self.ptr] = reward
        self.dones      [self.ptr] = done
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states     [idx]).to(self.device),
            torch.FloatTensor(self.actions    [idx]).to(self.device),
            torch.FloatTensor(self.next_states[idx]).to(self.device),
            torch.FloatTensor(self.rewards    [idx]).to(self.device),
            torch.FloatTensor(self.dones      [idx]).to(self.device),
        )

    def __len__(self) -> int:
        return self.size


class Actor(nn.Module):
    """Deterministic policy: state  →  action ∈ [-max_action, max_action]."""

    def __init__(self, state_dim: int, action_dim: int,
                 max_action: float, hidden_dim: int = 256):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim,  hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """Twin Q-networks that share architecture but have independent weights."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = state_dim + action_dim

        self.q1_net = nn.Sequential(
            nn.Linear(in_dim,    hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2_net = nn.Sequential(
            nn.Linear(in_dim,    hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        return self.q1_net(sa), self.q2_net(sa)

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1_net(torch.cat([state, action], dim=1))


class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient.

    Three key modifications over DDPG:
      1. Clipped Double Q-learning  — two critics, target = min(Q1, Q2)
      2. Delayed policy updates     — actor updated every `policy_freq` critic steps
      3. Target policy smoothing    — Gaussian noise added to target actions
    """

    def __init__(
        self,
        state_dim:    int,
        action_dim:   int,
        max_action:   float,
        discount:     float = 0.99,
        tau:          float = 0.005,
        policy_noise: float = 0.2,
        noise_clip:   float = 0.5,
        policy_freq:  int   = 2,
        actor_lr:     float = 3e-4,
        critic_lr:    float = 3e-4,
        hidden_dim:   int   = 256,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor        = Actor(state_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt    = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic        = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt    = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action   = max_action
        self.discount     = discount
        self.tau          = tau
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_freq  = policy_freq
        self._it          = 0  # gradient step counter

    # ── Inference ─────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.actor(s).cpu().numpy().flatten()

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        self._it += 1
        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # --- Target policy smoothing (Sec 5.3) ---
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )

            # --- Clipped Double Q-learning target (Eq. 10) ---
            q1_t, q2_t = self.critic_target(next_states, next_actions)
            y = rewards + (1.0 - dones) * self.discount * torch.min(q1_t, q2_t)

        # --- Critic update ---
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Delayed actor + target network updates (Sec 5.2) ---
        if self._it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft update: θ' ← τ θ + (1 − τ) θ'
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.lerp_(p.data, self.tau)
            for p, p_t in zip(self.actor.parameters(), self.actor_target.parameters()):
                p_t.data.lerp_(p.data, self.tau)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target = copy.deepcopy(self.critic)
