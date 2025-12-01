#!/usr/bin/env python3
"""PPO (clip) on CartPole-v1 with GAE advantage and shared A2C backbone."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common_ac import ActorCriticNet, categorical_policy, compute_gae, make_env


def rollout(env, net, steps_per_epoch, gamma, lam, device, seed=None):
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(steps_per_epoch):
        s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = net(s)
        dist = categorical_policy(logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        obs_buf.append(obs)
        act_buf.append(action.item())
        logp_buf.append(logp.item())
        rew_buf.append(reward)
        val_buf.append(value.item())
        done_buf.append(float(done))

        obs = next_obs
        if done:
            obs, _ = env.reset()

    obs_arr = np.array(obs_buf, dtype=np.float32)
    act_arr = np.array(act_buf, dtype=np.int64)
    logp_arr = np.array(logp_buf, dtype=np.float32)
    rew_arr = np.array(rew_buf, dtype=np.float32)
    val_arr = np.array(val_buf, dtype=np.float32)
    done_arr = np.array(done_buf, dtype=np.float32)

    adv_arr, ret_arr = compute_gae(rew_arr, val_arr, done_arr, gamma, lam)
    adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

    return obs_arr, act_arr, logp_arr, adv_arr, ret_arr


def main():
    seed = 42
    render = False
    epochs = 200
    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
    train_iters = 4
    batch_size = 64
    steps_per_epoch = 2048
    value_coef = 0.5
    entropy_coef = 0.01
    lr = 3e-4

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed=seed, render=render)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = ActorCriticNet(obs_dim, act_dim, hidden=64)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    device = next(net.parameters()).device

    for epoch in range(1, epochs + 1):
        obs_arr, act_arr, logp_arr, adv_arr, ret_arr = rollout(
            env, net, steps_per_epoch, gamma, lam, device, seed=None
        )

        # tensors
        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_arr, dtype=torch.long, device=device)
        logp_old_t = torch.tensor(logp_arr, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_arr, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_arr, dtype=torch.float32, device=device)

        idxs = np.arange(len(obs_arr))
        for _ in range(train_iters):
            np.random.shuffle(idxs)
            for start in range(0, len(obs_arr), batch_size):
                mb_idx = idxs[start : start + batch_size]
                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_logp_old = logp_old_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                logits, value = net(mb_obs)
                dist = categorical_policy(logits)
                logp = dist.log_prob(mb_act)
                ratio = torch.exp(logp - mb_logp_old)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((mb_ret - value.squeeze()) ** 2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epoch % 10 == 0:
            mean_ret = ret_arr.mean()
            print(f"[PPO] Epoch {epoch}/{epochs}, mean return approx={mean_ret:.1f}")

    env.close()


if __name__ == "__main__":
    main()
