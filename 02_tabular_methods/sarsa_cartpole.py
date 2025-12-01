#!/usr/bin/env python3
"""On-policy Sarsa for CartPole-v1 with state discretization."""

import numpy as np

from common_tabular import (
    Discretizer,
    EpsilonScheduler,
    epsilon_greedy,
    make_env,
)

n_bins = 40
gamma = 0.99
epsilon_sched = EpsilonScheduler(start=1.0, end=0.05, decay_episodes=300)
episodes = 500
alpha = 0.1
seed = 42
render = False


def main():
    if seed is not None:
        np.random.seed(seed)

    env = make_env(render=render, seed=seed)
    disc = Discretizer(n_bins=n_bins)
    n_actions = env.action_space.n
    Q = np.zeros([disc.n_bins] * 4 + [n_actions], dtype=np.float32)

    for ep in range(1, episodes + 1):
        epsilon = epsilon_sched.value(ep)
        obs, _ = env.reset(seed=seed)
        s = disc(obs)
        a = epsilon_greedy(Q, s, epsilon)
        done = False
        total_reward = 0.0

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(a)
            if render:
                env.render()
            done = terminated or truncated
            s2 = disc(next_obs)
            a2 = epsilon_greedy(Q, s2, epsilon)

            td_target = reward + gamma * Q[s2][a2] * (1.0 - float(done))
            Q[s][a] += alpha * (td_target - Q[s][a])

            s, a = s2, a2
            total_reward += reward

        if ep % 10 == 0:
            print(
                f"[Sarsa] Episode {ep}/{episodes}, epsilon={epsilon:.3f}, total reward={total_reward}"
            )

    env.close()


if __name__ == "__main__":
    main()
