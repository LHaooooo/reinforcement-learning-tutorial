# 第 3 章：深度 Q 网络（DQN）

面向连续/高维状态、离散动作的 value-based 方法，用神经网络逼近 $Q(s,a)$。延续前两章结构：思想 → 公式 → 训练步骤 → 优缺点，并给出常见变体。

---

## 3.1 Vanilla DQN
- > **思想**：用神经网络逼近 $Q_\theta$，配合经验回放 + 目标网络稳定训练。
- **核心模块 / 公式**  
  - TD 目标：$y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')$  
  - 损失：$\mathcal{L} = \big(Q_\theta(s,a) - y\big)^2$  
  - 经验回放：打乱相关性、提升样本复用；批量采样 mini-batch。  
  - 目标网络：$\theta^-$ 定期/软更新，降低目标漂移。
- **训练步骤**  
  1) $\epsilon$-greedy 采样，存入 Replay Buffer $(s,a,r,s',d)$；  
  2) 若 buffer 足够大，采样 batch；  
  3) 用目标网络计算 $y$，最小化 MSE；  
  4) 每若干步将 $\theta$ 复制到 $\theta^-$；  
  5) 衰减 $\epsilon$，重复。
- **优缺点**  
  - 优：简单、样本可复用；比表格方法能处理高维状态。  
  - 缺：对超参和稳定技巧敏感；高估偏差；仅支持离散动作。

## 3.2 常见变体与技巧
- **Double DQN**：用当前网络选动作、目标网络评估，减小高估：  
  $y = r + \gamma Q_{\theta^-}\big(s', \arg\max_{a'} Q_\theta(s',a')\big)$
- **Dueling DQN**：分解 $Q(s,a)=V(s)+A(s,a)$，提升对状态价值估计的稳定性。
- **Prioritized Replay (PER)**：按 TD 误差采样，重要性采样权重修正偏差；$\alpha$ 控制偏斜，$\beta$ 退火到 1。
- **NoisyNet / ε-anneal**：用参数化噪声替代 ε，改善探索。
- **n-step DQN**：用多步回报提升学习速度。
- **Soft target update**：$\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$，平滑更新（$\tau\approx0.005\sim0.02$）。

## 3.3 关键超参数提示
- 学习率 1e-3~2.5e-4；批量 32/64；gamma=0.99。  
- Replay 容量：50k~1M；起始填充 1k+ 后再训练。  
- 目标网络更新：硬拷贝每 10~1000 步，或软更新 $\tau\in[0.001,0.01]$。  
- ε 调度：起始 1.0 → 0.05，指数或线性衰减；或用 NoisyNet。

## 3.4 小结对比（与第 2 章）

| 维度 | MC/Sarsa/Q-Learning（表格/离散化） | DQN |
| --- | --- | --- |
| 状态表示 | 表格/分箱 | 神经网络 |
| 样本利用 | 逐步/回合更新 | Replay Buffer 反复采样 |
| 稳定技巧 | 无/少量 | 目标网络、经验回放、Double/Dueling/PER |
| 动作空间 | 离散；连续需离散化 | 离散；连续需转为 DDPG/TD3/SAC |
| 计算成本 | 低 | 较高（前向/反向传播） |

## 3.5 代码索引
- `common_dqn.py`：ReplayBuffer、MLP、epsilon 调度、环境创建。
- `dqn_cartpole.py`：Vanilla DQN（目标网络 + 回放 + ε 衰减），结构与前章代码一致，可扩展 Double/Dueling/PER。

> 运行示例：`uv run python 03_dqn/dqn_cartpole.py`
