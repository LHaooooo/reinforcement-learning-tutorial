#!/usr/bin/env python3
"""Minimal DDPG on Pendulum-v1, using shared off-policy utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from common_offpolicy import (
    ActorDeterministic,
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
    lr = 1e-3
    batch_size = 64
    buffer_size = 200_000
    start_steps = 10_000
    total_steps = 200_000
    act_noise = 0.1

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed=seed, render=render)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = ActorDeterministic(obs_dim, act_dim, act_limit)
    actor_tgt = ActorDeterministic(obs_dim, act_dim, act_limit)
    actor_tgt.load_state_dict(actor.state_dict())

    critic = CriticQ(obs_dim, act_dim)
    critic_tgt = CriticQ(obs_dim, act_dim)
    critic_tgt.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    critic_opt = optim.Adam(critic.parameters(), lr=lr)

    buf = ReplayBuffer(buffer_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, actor_tgt, critic, critic_tgt = (
        actor.to(device),
        actor_tgt.to(device),
        critic.to(device),
        critic_tgt.to(device),
    )

    obs, _ = env.reset(seed=seed)
    ep_ret, ep_len, ep_count = 0.0, 0, 0
    recent = []
    for step in range(1, total_steps + 1):
        if step < start_steps:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                a = (
                    actor(torch.tensor(obs, dtype=torch.float32, device=device))
                    .cpu()
                    .numpy()
                )
            act = (a + np.random.normal(0, act_noise, size=act_dim)).clip(
                -act_limit, act_limit
            )

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
                    f"[DDPG] ep {ep_count}, ret={ep_ret:.1f}, avg10={avg:.1f}, len={ep_len}"
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
                a2 = actor_tgt(s2)
                q_target = r + gamma * critic_tgt(s2, a2) * (1 - d)

            q = critic(s, a)
            critic_loss = torch.mean((q - q_target) ** 2)

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            actor_loss = -critic(s, actor(s)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            soft_update(critic_tgt, critic, tau)
            soft_update(actor_tgt, actor, tau)

        if step % 50000 == 0:
            print(f"[DDPG] step {step}/{total_steps}")

    env.close()


if __name__ == "__main__":
    main()
