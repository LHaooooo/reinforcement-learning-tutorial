#!/usr/bin/env python3
"""Shared utilities for Actor-Critic / PPO on CartPole-v1 (discrete actions)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def make_env(seed: int | None = None, render: bool = False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        feat = self.feature(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat)
        return logits, value


def categorical_policy(logits):
    return torch.distributions.Categorical(logits=logits)


def compute_returns(rewards, dones, gamma: float):
    """Compute discounted returns for a full trajectory."""
    G = 0.0
    returns = []
    for r, d in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1.0 - d)
        returns.insert(0, G)
    return np.array(returns, dtype=np.float32)


def compute_gae(rewards, values, dones, gamma: float, lam: float):
    """Generalized Advantage Estimation (GAE-Lambda)."""
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns
