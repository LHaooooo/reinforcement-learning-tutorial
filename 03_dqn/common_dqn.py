#!/usr/bin/env python3
"""Common utilities for DQN family on CartPole."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


@dataclass
class EpsilonScheduler:
    start: float = 1.0
    end: float = 0.05
    decay: float = 0.995

    def step(self, eps: float) -> float:
        return max(self.end, eps * self.decay)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        self.buf.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class PrioritizedReplayBuffer:
    """Simple proportional PER; sufficient for small-scale demos.

    - Sampling prob p_i ∝ (priority_i + eps)^alpha
    - IS weight w_i = (N * p_i)^(-beta) / max_w  (normalize for stability)
    - Priorities updated with absolute TD error (+eps to avoid zero)
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, eps: float = 1e-3):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.data: list = []
        self.priorities: list[float] = []
        self.next_idx = 0

    def __len__(self):
        return len(self.data)

    def push(
        self,
        transition: Tuple[np.ndarray, int, float, np.ndarray, bool],
        priority: float | None = None,
    ):
        if priority is None:
            priority = max(self.priorities, default=1.0)
        if len(self.data) < self.capacity:
            self.data.append(transition)
            self.priorities.append(priority)
        else:
            self.data[self.next_idx] = transition
            self.priorities[self.next_idx] = priority
        self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        assert len(self.data) > 0, "PER buffer is empty"
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios**self.alpha  # priority -> sampling prob (α 控制偏斜程度)
        probs /= probs.sum()
        idxs = np.random.choice(len(self.data), batch_size, p=probs)
        weights = (len(self.data) * probs[idxs]) ** (-beta)  # IS 权重，β 退火到 1
        weights /= weights.max()

        batch = [self.data[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            idxs,
            weights.astype(np.float32),
        )

    def update_priorities(self, idxs, prios):
        for i, p in zip(idxs, prios):
            self.priorities[i] = float(max(p, self.eps))  # 避免优先级为 0


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingMLP(nn.Module):
    """Dueling architecture: shared trunk then value + advantage heads."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, out_dim)

    def forward(self, x):
        feat = self.feature(x)
        value = self.value(feat)
        adv = self.advantage(feat)
        # subtract mean advantage to keep identifiability
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q


def build_q_net(in_dim: int, out_dim: int, dueling: bool = False, hidden: int = 128):
    if dueling:
        return DuelingMLP(in_dim, out_dim, hidden)
    return MLP(in_dim, out_dim, hidden)


def make_env(render: bool = False, seed: int | None = None):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env
