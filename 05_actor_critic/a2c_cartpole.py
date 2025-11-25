import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, act_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        feat = self.feature(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat)
        return logits, value

def a2c_train(num_episodes=500, gamma=0.99, value_coef=0.5, entropy_coef=0.01):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = ActorCriticNet(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = net(s)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done_flag = terminated or truncated
            ep_reward += reward

            s2 = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, next_value = net(s2)
                if done_flag:
                    target_value = torch.tensor([[reward]], dtype=torch.float32, device=device)
                else:
                    target_value = torch.tensor([[reward]], dtype=torch.float32, device=device) + gamma * next_value

            advantage = target_value - value

            policy_loss = -(log_prob * advantage.detach())
            value_loss = (advantage ** 2) * value_coef
            loss = policy_loss + value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs
            if done_flag:
                break

        if (ep + 1) % 10 == 0:
            print(f"[A2C] Episode {ep+1}/{num_episodes}, total reward = {ep_reward}")

    env.close()

if __name__ == "__main__":
    a2c_train()
