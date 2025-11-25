import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class PolicyNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


env = gym.make("CartPole-v1")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
policy = PolicyNet(n_states, n_actions)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
gamma = 0.99

for episode in range(500):
    state, _ = env.reset()
    log_probs, rewards = [], []
    done = False
    while not done:
        s = torch.tensor(state, dtype=torch.float32)
        probs = policy(s)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        log_probs.append(dist.log_prob(action))
        rewards.append(reward)
        state = next_state

    returns, G = [], 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    loss = -torch.sum(torch.stack(log_probs) * returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (episode + 1) % 10 == 0:
        print(f"[REINFORCE] Episode {episode+1}/500, total reward = {sum(rewards)}")

env.close()
