#!/usr/bin/env python3
"""Shared utilities for tabular CartPole examples."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from dataclasses import dataclass


@dataclass
class EpsilonScheduler:
    start: float = 1.0
    end: float = 0.05
    decay_episodes: int = 300

    def value(self, episode: int) -> float:
        eps = self.start - (self.start - self.end) * episode / max(
            1, self.decay_episodes
        )
        return max(self.end, eps)


class Discretizer:
    def __init__(self, n_bins: int = 40):
        self.n_bins = n_bins
        self.bins = dict(
            pos=np.linspace(-2.4, 2.4, n_bins),
            vel=np.linspace(-3.0, 3.0, n_bins),
            ang=np.linspace(-0.21, 0.21, n_bins),
            ang_vel=np.linspace(-3.5, 3.5, n_bins),
        )

    def __call__(self, obs):
        cart_pos, cart_vel, pole_angle, pole_vel = obs
        idx = np.array(
            [
                np.digitize(cart_pos, self.bins["pos"]),
                np.digitize(cart_vel, self.bins["vel"]),
                np.digitize(pole_angle, self.bins["ang"]),
                np.digitize(pole_vel, self.bins["ang_vel"]),
            ],
            dtype=np.int64,
        )
        idx = np.clip(idx, 0, self.n_bins - 1)
        return tuple(idx)


def make_env(render: bool = False, seed: int | None = None):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def epsilon_greedy(q_table: np.ndarray, state, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(q_table.shape[-1])
    return int(np.argmax(q_table[state]))
