# 第 6 章：Off-Policy 深度强化学习（DDPG / TD3 / SAC）

面向连续动作控制，使用经验回放与目标网络，在“行为策略 ≠ 目标策略”下高效复用样本。按“思想 → 公式 → 步骤 → 优缺点”梳理，并对三种典型算法做对比。

---

## 6.1 Off-policy 要点
- 行为策略收集数据，目标策略用于更新；可反复从 Replay Buffer 抽样，样本效率高。
- 关键稳定技巧：经验回放、目标网络（软更新）、动作噪声（探索）、目标策略平滑（TD3/SAC）。

## 6.2 DDPG（Deterministic Policy Gradient）
- **思想**：连续动作版的 DQN + Actor-Critic，策略输出确定性动作 $\mu_\theta(s)$，Critic 估计 $Q_\phi(s,a)$。
- **TD 目标**：$y = r + \gamma Q_{\phi^-}(s', \mu_{\theta^-}(s'))$
- **损失**：$L_Q = \big(Q_\phi(s,a) - y\big)^2$；策略用链式法则最大化 $Q$：$\nabla_\theta J \approx \nabla_a Q_\phi(s,a)|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s)$。
- **步骤**：采样 → 加噪声行动（如 OU / 高斯）→ 存缓冲 → 批量更新 Q，再更新 Actor → 软更新目标网。
- **优缺点**：高样本效率；但对超参敏感，易过估计/过拟合。

## 6.3 TD3（Twin Delayed DDPG）
- **针对 DDPG 的三大改进**：  
  1) 双 Critic 取最小值，抑制 Q 高估：$y = r + \gamma \min_i Q_{\phi_i^-}(s', \tilde a')$；  
  2) 目标策略平滑：$\tilde a' = \mu_{\theta^-}(s') + \text{clip}(\mathcal{N}(0,\sigma), -c, c)$；  
  3) 延迟策略更新：Critic 多步更新一次 Actor/目标网。
- **效果**：显著提升稳定性，成为连续控制常用 baseline。

## 6.4 SAC（Soft Actor-Critic，最大熵）
- **思想**：最大化“回报 + 温度调节的熵”，鼓励多样且高回报的策略；策略为高斯分布，Actor 学习重参数化采样。
- **目标**：$\max_\pi \mathbb{E}\big[\sum \gamma^t (r_t + \alpha \mathcal{H}(\pi(\cdot|s_t)))\big]$
- **关键公式**：  
  - Q 目标：$y = r + \gamma \big(\min_i Q_{\phi_i^-}(s', a') - \alpha \log \pi_\theta(a'|s')\big)$，其中 $a'$ 来自当前策略。  
  - 策略更新（重参数化）：最小化 $\mathbb{E}\big[\alpha \log \pi_\theta(a|s) - \min_i Q_{\phi_i}(s,a)\big]$。  
  - 温度自适应：优化 $L_\alpha = \mathbb{E}\big[-\alpha (\log \pi_\theta(a|s) + \bar{\mathcal{H}})\big]$，使熵接近期望值。
- **特点**：样本效率高、稳定、探索充分，常为连续控制首选。

## 6.5 关键超参数（典型范围）
- 共享：buffer 1e5~1e6，批量 64~256，$\gamma=0.99$，软更新 $\tau=0.005$。  
- DDPG：动作噪声 $\sigma=0.1$~0.2；Actor lr 1e-3，Critic lr 1e-3。  
- TD3：目标平滑噪声 $\sigma=0.2$，clip 0.5；策略延迟 2；lr 同上。  
- SAC：温度初值 0.2~0.5（或自适应）；lr 3e-4；target entropy 约 = -动作维度。

## 6.6 对比小表
| 维度 | DDPG | TD3 | SAC |
| --- | --- | --- | --- |
| 策略 | 确定性 | 确定性 | 随机（高斯） |
| 估计 | 单 Q | 双 Q 取最小 | 双 Q 取最小 |
| 稳定性 | 较脆弱 | 更稳定（去高估+平滑+延迟） | 最稳（熵正则+随机策略） |
| 探索 | 外部噪声 | 外部噪声+目标平滑 | 策略内生随机性 |
| 适用 | 简单/低维 | 多数连续控制基线 | 推荐默认首选 |

## 6.7 代码索引
- `ddpg_pendulum.py`：DDPG（外部噪声 + 软更新）。  
- `td3_pendulum.py`：TD3（三大改进齐备）。  
- `sac_pendulum.py`：SAC 主体实现；详见长文 `sac_tutorial.md`。  
- `sac_tutorial.md`：SAC 动机、推导与逐段代码讲解。

> 运行示例（uv 环境）：  
> `uv run python 06_offpolicy_rl/ddpg_pendulum.py`  
> `uv run python 06_offpolicy_rl/td3_pendulum.py`  
> `uv run python 06_offpolicy_rl/sac_pendulum.py`
