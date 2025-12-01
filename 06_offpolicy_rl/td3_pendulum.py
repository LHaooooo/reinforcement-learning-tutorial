#!/usr/bin/env python3
"""TD3 on Pendulum-v1 (twin critics, target policy smoothing, delayed actor)."""

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
    batch_size = 128
    buffer_size = 200_000
    start_steps = 10_000
    total_steps = 200_000
    act_noise = 0.1
    target_noise = 0.2
    target_clip = 0.5
    policy_delay = 2

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed=seed, render=render)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = ActorDeterministic(obs_dim, act_dim, act_limit)
    actor_tgt = ActorDeterministic(obs_dim, act_dim, act_limit)
    actor_tgt.load_state_dict(actor.state_dict())

    critic1 = CriticQ(obs_dim, act_dim)
    critic2 = CriticQ(obs_dim, act_dim)
    critic1_tgt = CriticQ(obs_dim, act_dim)
    critic2_tgt = CriticQ(obs_dim, act_dim)
    critic1_tgt.load_state_dict(critic1.state_dict())
    critic2_tgt.load_state_dict(critic2.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    critic_opt = optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=lr
    )

    buf = ReplayBuffer(buffer_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, actor_tgt = actor.to(device), actor_tgt.to(device)
    critic1, critic2, critic1_tgt, critic2_tgt = (
        critic1.to(device),
        critic2.to(device),
        critic1_tgt.to(device),
        critic2_tgt.to(device),
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
                    f"[TD3] ep {ep_count}, ret={ep_ret:.1f}, avg10={avg:.1f}, len={ep_len}"
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
                noise = (torch.randn_like(a) * target_noise).clamp(
                    -target_clip, target_clip
                )
                a2 = (actor_tgt(s2) + noise).clamp(-act_limit, act_limit)
                q1_t = critic1_tgt(s2, a2)
                q2_t = critic2_tgt(s2, a2)
                q_target = r + gamma * torch.min(q1_t, q2_t) * (1 - d)

            q1 = critic1(s, a)
            q2 = critic2(s, a)
            critic_loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            if step % policy_delay == 0:
                actor_loss = -critic1(s, actor(s)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                soft_update(actor_tgt, actor, tau)
                soft_update(critic1_tgt, critic1, tau)
                soft_update(critic2_tgt, critic2, tau)

        if step % 50000 == 0:
            print(f"[TD3] step {step}/{total_steps}")

    env.close()


if __name__ == "__main__":
    main()
