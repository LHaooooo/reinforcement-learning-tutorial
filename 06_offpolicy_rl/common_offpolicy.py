#!/usr/bin/env python3
"""Shared building blocks for DDPG / TD3 / SAC on Pendulum-v1."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_env(seed: int | None = None, render: bool = False):
    env = gym.make("Pendulum-v1", render_mode="human" if render else None)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]):
        """Store a transition; accepts either a tuple or unpacked arguments."""
        if len(transition) == 1 and isinstance(transition[0], tuple):
            self.buf.append(transition[0])
        else:
            self.buf.append(tuple(transition))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones, dtype=np.float32)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buf)


def mlp(in_dim, out_dim, hidden=256):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class ActorDeterministic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = mlp(obs_dim, act_dim)
        self.act_limit = act_limit

    def forward(self, x):
        return torch.tanh(self.net(x)) * self.act_limit


class CriticQ(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class ActorGaussian(nn.Module):
    """Gaussian policy with squashed tanh actions."""

    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)
        self.act_limit = act_limit

    def forward(self, s):
        mu_logstd = self.net(s)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        a = torch.tanh(z)
        logp = normal.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        logp = logp.sum(dim=-1, keepdim=True)
        return a * self.act_limit, logp


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(tau * sp.data)
