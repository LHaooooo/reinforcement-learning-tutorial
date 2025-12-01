#!/usr/bin/env python3
"""DQN family for CartPole-v1: vanilla / Double / Dueling / PER / soft target."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common_dqn import (
    EpsilonScheduler,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    build_q_net,
    make_env,
)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_tau: int = 20,
        dueling: bool = False,
        double_dqn: bool = True,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        tau_polyak: float | None = None,  # if set, use soft target update every step
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.double_dqn = double_dqn
        self.use_per = use_per
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.tau_polyak = tau_polyak
        self.device = torch.device(device)

        self.policy_net = build_q_net(state_dim, action_dim, dueling=dueling).to(
            self.device
        )
        self.target_net = build_q_net(state_dim, action_dim, dueling=dueling).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        if use_per:
            self.memory = PrioritizedReplayBuffer(buffer_size, alpha=per_alpha)
        else:
            self.memory = ReplayBuffer(buffer_size)
        self.steps = 0

    def act(self, state: np.ndarray, epsilon: float, action_space) -> int:
        if np.random.rand() < epsilon:
            return action_space.sample()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
                0
            )
            return int(self.policy_net(s).argmax(dim=1).item())

    def push(self, *transition, priority=None):
        if self.use_per:
            self.memory.push(transition, priority=priority)
        else:
            self.memory.push(transition)

    def _loss(self, batch):
        if self.use_per:
            states, actions, rewards, next_states, dones, idxs, weights = batch
        else:
            states, actions, rewards, next_states, dones = batch
            idxs, weights = None, None

        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: action from policy, value from target
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        td_error = q_values - target
        if weights is not None:
            w = torch.tensor(weights, device=self.device).unsqueeze(1)
            loss = (td_error.pow(2) * w).mean()
        else:
            loss = td_error.pow(2).mean()
        return loss, td_error.detach(), idxs

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        if self.use_per:
            beta = min(
                1.0,
                self.per_beta_start
                + self.steps
                / max(1, self.per_beta_frames)
                * (1.0 - self.per_beta_start),
            )
            batch = self.memory.sample(self.batch_size, beta=beta)
        else:
            batch = self.memory.sample(self.batch_size)

        loss, td_error, idxs = self._loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.tau_polyak is not None:
            with torch.no_grad():
                for t, s in zip(
                    self.target_net.parameters(), self.policy_net.parameters()
                ):
                    t.data.mul_(1 - self.tau_polyak).add_(self.tau_polyak * s.data)
        elif self.steps % self.target_tau == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.use_per and idxs is not None:
            prios = td_error.abs().cpu().numpy().flatten() + getattr(
                self.memory, "eps", 1e-3
            )  # |TD error| 作为新优先级
            self.memory.update_priorities(idxs, prios)

        return float(loss.item())


def main():
    seed = 42
    render = False
    env = make_env(render=render, seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        lr=1e-3,
        buffer_size=50_000,
        batch_size=64,
        target_tau=20,
        dueling=True,  # toggle dueling head
        double_dqn=True,  # toggle double DQN target
        use_per=True,  # prioritized replay
        tau_polyak=0.01,  # soft target update each step (set None for hard)
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    eps_sched = EpsilonScheduler(start=1.0, end=0.05, decay=0.995)

    episodes = 500
    epsilon = 1.0
    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(state, epsilon, env.action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            agent.train_step()

        epsilon = eps_sched.step(epsilon)

        if ep % 10 == 0:
            print(
                f"[DQN] Episode {ep}/{episodes}, epsilon={epsilon:.3f}, total reward={total_reward:.1f}"
            )

    env.close()


if __name__ == "__main__":
    main()
