#!/usr/bin/env python3
"""First-visit Monte Carlo control (on-policy) on CartPole-v1."""

import numpy as np
from collections import defaultdict

from common_tabular import (
    Discretizer,
    EpsilonScheduler,
    epsilon_greedy,
    make_env,
)

n_bins = 40
gamma = 0.99
epsilon_sched = EpsilonScheduler(start=1.0, end=0.05, decay_episodes=500)
episodes = 800
seed = 42
render = False


def main():
    if seed is not None:
        np.random.seed(seed)

    env = make_env(render=render, seed=seed)
    disc = Discretizer(n_bins=n_bins)
    n_actions = env.action_space.n
    Q = np.zeros([n_bins] * 4 + [n_actions], dtype=np.float32)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for ep in range(1, episodes + 1):
        epsilon = epsilon_sched.value(ep)

        # 生成一条完整轨迹
        obs, _ = env.reset(seed=seed)
        episode = []
        done = False
        while not done:
            s = disc(obs)
            a = epsilon_greedy(Q, s, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(a)
            if render:
                env.render()
            done = terminated or truncated
            episode.append((s, a, reward))
            obs = next_obs

        # 计算每个 (s,a) 的首次出现回报并更新 Q
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) in visited:
                continue  # first-visit
            visited.add((s, a))
            returns_sum[(s, a)] += G
            returns_count[(s, a)] += 1
            Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]

        if ep % 10 == 0:
            total_reward = sum(r for _, _, r in episode)
            print(
                f"[MC] Episode {ep}/{episodes}, epsilon={epsilon:.3f}, total reward={total_reward}"
            )

    env.close()


if __name__ == "__main__":
    main()
