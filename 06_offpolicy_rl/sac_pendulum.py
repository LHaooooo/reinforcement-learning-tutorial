#!/usr/bin/env python3
"""Soft Actor-Critic on Pendulum-v1 with automatic entropy tuning."""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from common_offpolicy import (
    ActorGaussian,
    CriticQ,
    ReplayBuffer,
    make_env,
    soft_update,
)


def main():
    seed = 42
    render = False
    gamma = 0.99
    tau = 0.005
    lr = 3e-4
    batch_size = 256
    buffer_size = 500_000
    start_steps = 5_000
    total_steps = 300_000
    act_limit = 2.0  # Pendulum action bound

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed=seed, render=render)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = ActorGaussian(obs_dim, act_dim, act_limit)
    q1 = CriticQ(obs_dim, act_dim)
    q2 = CriticQ(obs_dim, act_dim)
    q1_tgt = CriticQ(obs_dim, act_dim)
    q2_tgt = CriticQ(obs_dim, act_dim)
    q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt.load_state_dict(q2.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)

    # entropy temperature
    target_entropy = -float(act_dim)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=lr)

    buf = ReplayBuffer(buffer_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, q1, q2, q1_tgt, q2_tgt = (
        actor.to(device),
        q1.to(device),
        q2.to(device),
        q1_tgt.to(device),
        q2_tgt.to(device),
    )

    obs, _ = env.reset(seed=seed)
    ep_ret, ep_len, ep_count = 0.0, 0, 0
    recent = []
    for step in range(1, total_steps + 1):
        if step < start_steps:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a, _ = actor(s)
            act = a.cpu().numpy()[0]

        next_obs, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        buf.push(obs, act, reward, next_obs, done)
        obs = next_obs if not done else env.reset(seed=seed)[0]
        ep_ret += reward
        ep_len += 1

        if done:
            ep_count += 1
            recent.append(ep_ret)
            if len(recent) > 10:
                recent.pop(0)
            if ep_count % 5 == 0:
                avg = sum(recent) / len(recent)
                print(
                    f"[SAC] ep {ep_count}, ret={ep_ret:.1f}, avg10={avg:.1f}, len={ep_len}"
                )
            ep_ret, ep_len = 0.0, 0

        if len(buf) >= batch_size:
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
                q1_t = q1_tgt(s2, a2)
                q2_t = q2_tgt(s2, a2)
                q_tgt = torch.min(q1_t, q2_t) - torch.exp(log_alpha) * logp2
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

            soft_update(q1_tgt, q1, tau)
            soft_update(q2_tgt, q2, tau)

        if step % 50000 == 0:
            print(f"[SAC] step {step}/{total_steps}")

    env.close()


if __name__ == "__main__":
    main()
