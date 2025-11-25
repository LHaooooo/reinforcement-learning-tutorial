import numpy as np
import torch

def discount_cumsum(rewards, gamma):
    res = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        res[t] = running
    return res

def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32, device=device)
    return torch.tensor(np.array(x, dtype=np.float32), device=device)
