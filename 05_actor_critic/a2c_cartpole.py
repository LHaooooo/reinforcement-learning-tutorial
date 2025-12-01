#!/usr/bin/env python3
"""Advantage Actor-Critic (A2C, single-step TD) on CartPole-v1."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common_ac import ActorCriticNet, categorical_policy, make_env


def main():
    seed = 42
    render = False
    episodes = 500
    gamma = 0.99
    value_coef = 0.5
    entropy_coef = 0.01
    lr = 1e-3

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed=seed, render=render)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = ActorCriticNet(obs_dim, act_dim, hidden=128)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    device = next(net.parameters()).device

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed)
        done = False
        ep_reward = 0.0

        while not done:
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = net(s)
            dist = categorical_policy(logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done_flag = terminated or truncated
            ep_reward += reward

            with torch.no_grad():
                s2 = torch.tensor(
                    next_obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                _, next_value = net(s2)
                target = reward + gamma * next_value * (1.0 - float(done_flag))

            advantage = target - value

            policy_loss = -(log_prob * advantage.detach())
            value_loss = (advantage**2) * value_coef
            loss = policy_loss + value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs
            done = done_flag

            if render:
                env.render()

        if ep % 10 == 0:
            print(f"[A2C] Episode {ep}/{episodes}, reward={ep_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()
