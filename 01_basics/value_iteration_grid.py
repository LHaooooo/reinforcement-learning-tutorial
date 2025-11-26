#!/usr/bin/env python3
"""Tiny GridWorld value iteration demo (deterministic, tabular)."""

import numpy as np


# 4x4 GridWorld，终止状态在 (0,0) 和 (3,3)
HEIGHT, WIDTH = 4, 4
TERMINAL = {(0, 0), (HEIGHT - 1, WIDTH - 1)}
ACTIONS = {
    0: (-1, 0),  # 上
    1: (1, 0),  # 下
    2: (0, -1),  # 左
    3: (0, 1),  # 右
}
GAMMA = 0.99
THETA = 1e-4  # 收敛阈值
REWARD_STEP = -1.0  # 每步扣分，鼓励更快到终点


def in_grid(r, c):
    """检查坐标是否在网格内。"""
    return 0 <= r < HEIGHT and 0 <= c < WIDTH


def step(state, action):
    """确定性转移：出界则原地不动。"""
    if state in TERMINAL:
        return state, 0.0  # stay
    dr, dc = ACTIONS[action]
    r2, c2 = state[0] + dr, state[1] + dc
    if not in_grid(r2, c2):
        r2, c2 = state  # bump into wall, stay
    return (r2, c2), REWARD_STEP


def value_iteration():
    V = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    policy = np.zeros((HEIGHT, WIDTH), dtype=np.int64)

    iteration = 0
    while True:
        delta = 0.0
        for r in range(HEIGHT):
            for c in range(WIDTH):
                s = (r, c)
                if s in TERMINAL:
                    continue
                # 评估所有动作得到 Q(s,a) 并取最大值
                q_values = []
                for a in ACTIONS:
                    s2, rwd = step(s, a)
                    q_values.append(rwd + GAMMA * V[s2])
                new_v = max(q_values)
                delta = max(delta, abs(new_v - V[s]))
                V[s] = new_v
        iteration += 1
        if delta < THETA:
            break

    # 基于收敛后的 V 贪心抽取策略
    for r in range(HEIGHT):
        for c in range(WIDTH):
            s = (r, c)
            if s in TERMINAL:
                continue
            q_values = []
            for a in ACTIONS:
                s2, rwd = step(s, a)
                q_values.append(rwd + GAMMA * V[s2])
            policy[s] = int(np.argmax(q_values))

    return V, policy, iteration


def main():
    # 价值迭代
    V, policy, iters = value_iteration()

    # log
    action_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    print(f"Converged in {iters} iterations (theta={THETA})")
    print("Value function:")
    for r in range(HEIGHT):
        row = " ".join(f"{V[r, c]:6.2f}" for c in range(WIDTH))
        print(row)
    print("\nGreedy policy (arrows, terminals stay):")
    for r in range(HEIGHT):
        row = []
        for c in range(WIDTH):
            if (r, c) in TERMINAL:
                row.append("T")
            else:
                row.append(action_map[policy[r, c]])
        print(" ".join(row))


if __name__ == "__main__":
    main()
