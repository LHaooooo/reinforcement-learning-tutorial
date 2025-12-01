#!/usr/bin/env python3
"""REINFORCE on CartPole-v1 with optional entropy bonus and seeding."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class PolicyNet(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


def make_env(seed: int | None = None, render: bool = False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def compute_returns(rewards, gamma: float):
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def main():
    seed = 42
    render = False
    gamma = 0.99
    lr = 0.01
    entropy_coef = 0.0  # set >0 to encourage exploration
    episodes = 500

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed=seed, render=render)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyNet(n_states, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed)
        log_probs, rewards, entropies = [], [], []
        done = False

        while not done:
            s = torch.tensor(state, dtype=torch.float32)
            probs = policy(s)
            dist = torch.distributions.Categorical(
                probs
            )  # 构造离散分布（每个动作的概率）
            action = dist.sample()  # 按当前策略随机采样动作，保证探索

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            rewards.append(reward)
            state = next_state

            if render:
                env.render()

        returns = compute_returns(rewards, gamma)
        # baseline: mean normalization to reduce variance
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        loss = -(log_probs * returns).sum() - entropy_coef * entropies.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(
                f"[REINFORCE] Episode {episode}/{episodes}, total reward={sum(rewards):.1f}"
            )

    env.close()


if __name__ == "__main__":
    main()
