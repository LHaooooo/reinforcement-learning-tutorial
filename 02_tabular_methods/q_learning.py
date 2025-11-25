import gymnasium as gym
import numpy as np

"""简单示例：在 CartPole-v1 上使用离散化 Q-Learning"""

env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_bins = 40

# 状态离散化
def discretize(obs):
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    bins_pos = np.linspace(-2.4, 2.4, n_bins)
    bins_vel = np.linspace(-3.0, 3.0, n_bins)
    bins_angle = np.linspace(-0.21, 0.21, n_bins)
    bins_pole_vel = np.linspace(-3.5, 3.5, n_bins)
    idx = (
        np.digitize(cart_pos, bins_pos),
        np.digitize(cart_vel, bins_vel),
        np.digitize(pole_angle, bins_angle),
        np.digitize(pole_vel, bins_pole_vel),
    )
    return idx

Q = np.zeros([n_bins]*4 + [n_actions], dtype=np.float32)
alpha = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 500

for ep in range(episodes):
    obs, _ = env.reset()
    s = discretize(obs)
    done = False
    total_reward = 0
    while not done:
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s])

        next_obs, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        s2 = discretize(next_obs)

        Q[s][a] += alpha * (reward + gamma * np.max(Q[s2]) - Q[s][a])

        s = s2
        total_reward += reward

    if (ep + 1) % 10 == 0:
        print(f"[Q-Learning] Episode {ep+1}/{episodes}, total reward = {total_reward}")

env.close()
