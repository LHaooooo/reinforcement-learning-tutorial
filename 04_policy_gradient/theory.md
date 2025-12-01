# 第 4 章：策略梯度（Policy Gradient）

面向连续/高维状态且动作可离散的场景，直接对策略 $\pi_\theta(a|s)$ 建模，目标是最大化期望回报。延续前几章结构：思想 → 理论 → 训练步骤 → 优缺点，并列出常见改进。

---

## 4.1 基础思想
- > 直接参数化策略（概率分布），通过梯度上升最大化 $J(\theta)=\mathbb{E}_\pi[\sum \gamma^t r_t]$。
- 不需要显式价值表/网络（但可用基线降低方差）。

## 4.2 核心公式：策略梯度定理
> $$
\nabla_\theta J(\theta) = \mathbb{E}_{s,a\sim\pi_\theta}\big[ \nabla_\theta \log \pi_\theta(a|s)\, Q^\pi(s,a) \big]
$$
- 若用回报 $G_t$ 近似 $Q^\pi$ 即得到 REINFORCE 更新：  
  $\theta \leftarrow \theta + \alpha\, \nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t$
- 减基线（baseline）：用 $b(s)$ 或均值替代，仍然无偏，可显著降方差。
- 损失常写作 $L = - \sum_t \log\pi_t \cdot G_t$，因为深度学习框架做“最小化”。对这个损失做梯度下降，等价于对 $J$ 做梯度上升（只差一个负号）。
- 推导要点（对数技巧）：  
  1) 目标 $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$；  
  2) $\nabla_\theta J = \mathbb{E}_\tau\big[R(\tau)\nabla_\theta \log p_\theta(\tau)\big]$；  
  3) $p_\theta(\tau)=p(s_0)\prod_t \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)$，环境转移与 $\theta$ 无关，故 $\nabla_\theta\log p_\theta(\tau)=\sum_t \nabla_\theta\log \pi_\theta(a_t|s_t)$；  
  4) 把整条回报 $R(\tau)$ 换成每步回报估计 $G_t$ 得到 REINFORCE 形式；基线项期望为 0，可降方差。

## 4.3 训练步骤（REINFORCE）
1) 用当前策略采样一条完整 episode，记录 $(\log\pi, r)$ 序列。  
2) 从末尾计算折扣回报 $G_t$。  
3) 归一化/减基线以降方差（如回报标准化）。  
4) 损失 $L = - \sum_t \log\pi_t \cdot G_t$，反向更新。  
5) 重复多 episode，渐进改进策略。

## 4.4 优缺点
- 优：实现简单、端到端可微；可直接输出随机策略，适配随机性要求任务。  
- 缺：方差高、收敛慢；样本效率低；需要整回合回传（无 bootstrap）。  
- 改进方向：基线/优势函数（Actor-Critic）、方差缩减（GAE）、信赖域或截断更新（TRPO/PPO）。

## 4.5 常见变体 / 技巧
- **方差降低**：baseline（均值或价值函数）、优势函数 $A=Q-V$。  
- **奖励塑形 / 归一化**：回报标准化、奖励裁剪。  
- **熵正则**：在损失中加 $-\beta H(\pi)$ 促进探索。  
- **多步/批量**：按 batch 多条轨迹求梯度，方差更低。  
- **Actor-Critic**（见第 5 章）：同时学价值网络作为基线。  
- **PPO/TRPO**：限制策略更新步长（剪切或 KL 约束），提升稳定性。

## 4.6 与前几章的对比

| 维度 | 表格/TD (Q/Sarsa/MC) | DQN | Policy Gradient (REINFORCE) |
| --- | --- | --- | --- |
| 目标 | 学 $Q$ 或 $V$ | 学 $Q$（NN） | 直接学 $\pi$ |
| 更新信号 | TD 目标或回报 | TD 目标 + 回放/目标网 | 回报/优势，整回合 |
| 动作空间 | 离散/需离散化 | 离散 | 离散（连续需 PG/Actor-Critic 高斯） |
| 方差 | 低-中 | 中 | 高（无基线） |
| 样本效率 | 较高 | 中 | 低 |

## 4.7 代码索引
- `reinforce_cartpole.py`：REINFORCE（回报标准化降方差）。可加入熵正则、baseline（价值网络）等改进。

> 运行示例：`uv run python 04_policy_gradient/reinforce_cartpole.py`
