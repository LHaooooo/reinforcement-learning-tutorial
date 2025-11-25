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
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        feat = self.feature(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat)
        return logits, value


def ppo_cartpole(
    num_episodes=500,
    gamma=0.99,
    lam=0.95,
    clip_ratio=0.2,
    lr=3e-4,
    train_iters=4,
    batch_size=64,
    steps_per_epoch=2000,
):

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = ActorCriticNet(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    def compute_advantages(rewards, values, dones, gamma, lam):
        adv = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_values = 0
                next_nonterminal = 1.0 - dones[t]
            else:
                next_values = values[t + 1]
                next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
            lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
            adv[t] = lastgaelam
        return adv

    for ep in range(num_episodes):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        obs, _ = env.reset()
        ep_return = 0

        for step in range(steps_per_epoch):
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = net(s)
            dist = torch.distributions.Categorical(logits=logits)
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
            ep_return += reward

            if done:
                obs, _ = env.reset()
                ep_return = 0

        # 转为数组
        obs_arr = np.array(obs_buf, dtype=np.float32)
        act_arr = np.array(act_buf, dtype=np.int64)
        logp_arr = np.array(logp_buf, dtype=np.float32)
        rew_arr = np.array(rew_buf, dtype=np.float32)
        val_arr = np.array(val_buf, dtype=np.float32)
        done_arr = np.array(done_buf, dtype=np.float32)

        adv_arr = compute_advantages(rew_arr, val_arr, done_arr, gamma, lam)
        ret_arr = adv_arr + val_arr

        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_arr, dtype=torch.long, device=device)
        logp_old_t = torch.tensor(logp_arr, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_arr, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_arr, dtype=torch.float32, device=device)

        for _ in range(train_iters):
            idxs = np.arange(len(obs_arr))
            np.random.shuffle(idxs)
            for start in range(0, len(obs_arr), batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]
                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_logp_old = logp_old_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                logits, value = net(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(mb_act)
                ratio = torch.exp(logp - mb_logp_old)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((mb_ret - value.squeeze()) ** 2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if (ep + 1) % 5 == 0:
            print(f"[PPO] Epoch {ep+1}/{num_episodes} finished")

    env.close()


if __name__ == "__main__":
    ppo_cartpole()
