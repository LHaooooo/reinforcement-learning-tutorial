#!/usr/bin/env python3
"""Mini RLPD-style distillation on Pendulum-v1.

- Teacher: SAC actor (train briefly if no checkpoint).
- Student: Encoder + Gaussian policy, supervised to match teacher actions.
- Goal: 演示表征+蒸馏流程，而非追求最优成绩。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# make offpolicy utilities accessible
sys.path.append(str(Path(__file__).resolve().parent.parent / "06_offpolicy_rl"))
from common_offpolicy import ActorGaussian, CriticQ, ReplayBuffer, make_env, soft_update

CHECKPOINT = Path("07_representation_rl/teacher_sac.pth")


class StudentPolicy(nn.Module):
    """Encoder + Gaussian head (mean-only, fixed log_std) for simplicity."""

    def __init__(self, obs_dim, act_dim, act_limit, hidden=128, log_std=-0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std, requires_grad=False)
        self.act_limit = act_limit

    def forward(self, s):
        z = self.encoder(s)
        mu = self.mu_head(z)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        z_sample = dist.rsample()
        a = torch.tanh(z_sample) * self.act_limit
        return a, dist


def train_teacher_sac(env, steps=80_000, device="cpu"):
    """Quick SAC training to get a reasonable teacher."""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = ActorGaussian(obs_dim, act_dim, act_limit).to(device)
    q1 = CriticQ(obs_dim, act_dim).to(device)
    q2 = CriticQ(obs_dim, act_dim).to(device)
    q1_t = CriticQ(obs_dim, act_dim).to(device)
    q2_t = CriticQ(obs_dim, act_dim).to(device)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())

    buf = ReplayBuffer(500_000)
    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -float(act_dim)

    gamma, tau = 0.99, 0.005
    batch_size, start_steps = 256, 5000
    obs, _ = env.reset(seed=42)
    for step in range(1, steps + 1):
        if step < start_steps:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _ = actor(
                    torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                )
            act = a.cpu().numpy()[0]

        next_obs, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        buf.push(obs, act, reward, next_obs, done)
        obs = next_obs if not done else env.reset(seed=42)[0]

        if len(buf) < batch_size:
            continue
        s, a, r, s2, d = buf.sample(batch_size)
        s, a, r, s2, d = (
            s.to(device),
            a.to(device),
            r.to(device),
            s2.to(device),
            d.to(device),
        )

        with torch.no_grad():
            a2, logp2 = actor(s2)
            q1_tgt = q1_t(s2, a2)
            q2_tgt = q2_t(s2, a2)
            q_tgt = torch.min(q1_tgt, q2_tgt) - torch.exp(log_alpha) * logp2
            y = r + gamma * (1 - d) * q_tgt

        q1_pred = q1(s, a)
        q2_pred = q2(s, a)
        q_loss = ((q1_pred - y) ** 2 + (q2_pred - y) ** 2).mean()
        q_opt.zero_grad()
        q_loss.backward()
        q_opt.step()

        a_new, logp_new = actor(s)
        q_new = torch.min(q1(s, a_new), q2(s, a_new))
        actor_loss = (torch.exp(log_alpha) * logp_new - q_new).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        alpha_loss = -(log_alpha * (logp_new + target_entropy).detach()).mean()
        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()

        soft_update(q1_t, q1, tau)
        soft_update(q2_t, q2, tau)

    torch.save({"actor": actor.state_dict()}, CHECKPOINT)
    return actor


def collect_dataset(env, teacher, device, n_samples=20_000):
    obs_list, act_list = [], []
    obs, _ = env.reset(seed=123)
    with torch.no_grad():
        for _ in range(n_samples):
            s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a_t, _ = teacher(s_t)
            a_np = a_t.cpu().numpy()[0]
            obs_list.append(obs)
            act_list.append(a_np)
            obs, _, terminated, truncated, _ = env.step(a_np)
            if terminated or truncated:
                obs, _ = env.reset()
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def distill_student(env, teacher, device, epochs=50, batch_size=256):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    student = StudentPolicy(obs_dim, act_dim, act_limit).to(device)
    opt = optim.Adam(student.parameters(), lr=3e-4)

    obs_arr, act_arr = collect_dataset(env, teacher, device)
    n = len(obs_arr)
    idxs = np.arange(n)

    for ep in range(1, epochs + 1):
        np.random.shuffle(idxs)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            mb = idxs[start : start + batch_size]
            s = torch.tensor(obs_arr[mb], dtype=torch.float32, device=device)
            a_teacher = torch.tensor(act_arr[mb], dtype=torch.float32, device=device)
            a_pred, _ = student(s)
            loss = ((a_pred - a_teacher) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(mb)
        if ep % 10 == 0:
            print(f"[Student] epoch {ep}/{epochs}, mse={total_loss/n:.4f}")
    torch.save(student.state_dict(), "07_representation_rl/student_rlpd.pth")
    return student


def evaluate(env, policy, device, episodes=5):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, ret = False, 0.0
        while not done:
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a, _ = policy(s)
            obs, reward, terminated, truncated, _ = env.step(a.cpu().numpy()[0])
            done = terminated or truncated
            ret += reward
        returns.append(ret)
    return np.mean(returns), np.std(returns)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(seed=42, render=False)

    if CHECKPOINT.exists():
        teacher = ActorGaussian(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            float(env.action_space.high[0]),
        ).to(device)
        teacher.load_state_dict(torch.load(CHECKPOINT)["actor"])
        print("[Info] Loaded existing Teacher checkpoint.")
    else:
        print("[Info] Training Teacher SAC...")
        teacher = train_teacher_sac(env, steps=80_000, device=device)

    print("[Info] Distilling Student...")
    student = distill_student(env, teacher, device, epochs=50, batch_size=256)

    m_teacher, s_teacher = evaluate(env, teacher, device, episodes=5)
    m_student, s_student = evaluate(env, student, device, episodes=5)
    print(f"[Eval] Teacher mean return {m_teacher:.1f} ± {s_teacher:.1f}")
    print(f"[Eval] Student mean return {m_student:.1f} ± {s_student:.1f}")

    env.close()


if __name__ == "__main__":
    main()
