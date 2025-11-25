import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_sizes=(128, 128), activation=nn.ReLU
    ):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128), activation=nn.ReLU):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        self.feature = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, act_dim)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, x):
        feat = self.feature(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat)
        return logits, value
