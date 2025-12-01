# 第 5 章：Actor-Critic 与 PPO

在策略梯度的高方差与纯值方法的高偏差之间，Actor-Critic 结合“策略 + 价值”来降低方差、提升稳定性；PPO 则在此基础上加上更新约束。延续前几章结构：思想 → 公式 → 步骤 → 优缺点，并给出关键超参与代码索引。

---

## 5.1 Actor-Critic 总体思路
- > 同时学习：Actor 输出策略 $\pi_\theta(a|s)$；Critic 估计价值（$V_\phi(s)$ 或 $Q_\phi$）作为基线/优势。
- 目标：最大化 $J(\theta)=\mathbb{E}[\sum \gamma^t r_t]$，用 $\nabla_\theta \log\pi \cdot \hat A$ 做梯度，上式的 $\hat A$ 由 Critic 给出。

## 5.2 优势函数与低方差梯度
- 优势 $\hat A_t = \hat Q_t - \hat V_t$（或 TD 误差）可显著降低方差。
- 单步 TD Advantage：$\hat A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。
- GAE-$\lambda$：在偏差/方差间平衡，见 PPO 部分。

## 5.3 A2C（同步 Advantage Actor-Critic）
- > **思想**：on-policy，一步 TD 计算 advantage，策略和价值同时更新。  
- **损失**：  
  - 策略：$L_\pi = -\log\pi(a_t|s_t)\,\hat A_t$  
  - 价值：$L_V = (\hat A_t)^2$  
  - 熵正则：$- \beta H(\pi)$ 促进探索  
  - 总损失：$L = L_\pi + c_v L_V - c_e H$
- **更新步骤**：采样一步 → 估计 $\hat A_t$ → 反向传播；可多并行环境同步（此处示例单环境）。
- **优缺点**：方差远低于纯 PG；但仍是 on-policy，样本复用低，易受学习率/熵系数影响。

## 5.4 PPO（Clip 版，直观拆解）

> **思想**：限制策略一步更新幅度，避免“跳太远”导致性能崩溃；相比 TRPO，PPO 用简单的剪切替代二阶约束。

**关键比率**：$r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$，衡量新旧策略对同一动作的相对概率。

**剪切策略损失**：$L_{\text{clip}}=\mathbb{E}[\min(r_t \hat A_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]$  
- 先算“原比例收益” $r_t \hat A_t$；  
- 再算“截断后收益” $\text{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t$；  
- 取两者较小值做期望，防止 $r_t$ 偏离太大。  

**价值损失**：$L_V = (V_\phi - R_t)^2$  

**熵正则**：$- \beta H(\pi)$ 促进探索  

**总损失（最小化）**：$L = L_{\text{clip}} + c_v L_V - c_e H$

**优势估计（GAE-$\lambda$）**：  
$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$；  
$\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$，$\lambda \in [0,1]$ 控制偏差/方差。

**训练流程（单线程示例）**  
1) 旧策略采样 $T$ 步，存 $(s,a,\log\pi_{\text{old}},r,V,d)$。  
2) 计算 $\hat A$, $R$（优势常做标准化）。  
3) 多轮打乱 mini-batch，最小化剪切 + 价值 - 熵。  
4) 把当前参数作为“旧策略”，进入下一轮。

**优缺点**  
- 优：更新稳定、实现简单，业界常用 baseline。  
- 缺：仍 on-policy，样本需求大；对 $\epsilon$、迭代轮数较敏感。

## 5.5 关键超参数
- A2C：lr ~1e-3，熵系数 0.01，价值系数 0.5，gamma 0.99。  
- PPO：clip 0.1–0.3；GAE $\lambda$ 0.9–0.97；lr 3e-4；steps_per_epoch 2048；train_iters 3–10；batch 64–256；价值系数 0.5；熵系数 0.01。

## 5.6 与前几章对比（简表）

| 维度 | REINFORCE | A2C | PPO |
| --- | --- | --- | --- |
| 方差 | 高 | 低（基线+TD） | 低（基线+GAE+剪切） |
| 样本效率 | 低 | 中 | 中偏高（仍 on-policy） |
| 更新稳定性 | 低 | 中 | 高 |
| 约束 | 无 | 无 | 剪切/信赖域 |
| 适用 | 小任务/教学 | 简单连续离散 | 通用 baseline |

## 5.7 代码索引
- `common_ac.py`：环境构造、ActorCriticNet、Categorical 策略、GAE / 回报工具。
- `a2c_cartpole.py`：单环境 A2C（单步 TD advantage）。
- `ppo_cartpole.py`：PPO-Clip + GAE，mini-batch 多轮更新。

> 运行示例：  
> `uv run python 05_actor_critic/a2c_cartpole.py`  
> `uv run python 05_actor_critic/ppo_cartpole.py`
