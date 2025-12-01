# 第 2 章：表格型方法（Tabular Methods）

面向可枚举/可离散化的小规模任务（如 CartPole 经过分箱），介绍三种核心方法。每种都按「思想 → 理论公式 → 训练步骤 → 优缺点」整理，并在末尾给出对比表与代码索引。

---

## 2.1 蒙特卡洛（MC）
- > **思想**：整条 episode 跑完，用真实累积回报 $G_t$ 直接估计价值，不做 bootstrap。
- **理论/公式**  
  - 预测（every-visit）：$V(s)\leftarrow V(s)+\alpha\big(G_t-V(s)\big)$  
  - 首访控制（均值形式）：$Q(s,a)=\frac{1}{N(s,a)}\sum G(s,a)$，等价于增量式 $Q\leftarrow Q+\frac{1}{N}(G-Q)$
- **训练步骤（首访控制）**  
  1) 按当前策略（常用 $\epsilon$-greedy w.r.t $Q$）采样一条完整 episode；  
  2) 从末尾反向累积 $G \leftarrow r_t+\gamma G$；  
  3) 对首访的 $(s,a)$ 用当前 $G$ 更新 $Q$ 的均值估计；  
  4) 用更新后的 $Q$ 做 $\epsilon$-greedy，进入下一回合。
- **优缺点**  
  - 优：无偏；不依赖环境模型；概念直观。  
  - 缺：方差高；必须等回合结束；样本效率低。

## 2.2 时序差分 / Sarsa（on-policy TD）
- > **思想**：一步 bootstrap，用“估计的未来”替代完整回报；同时用当前策略的下一动作更新（on-policy）。
- **理论/公式**  
  - 状态值：$V(s_t)\leftarrow V(s_t)+\alpha\big(r_t+\gamma V(s_{t+1})-V(s_t)\big)$  
  - Sarsa：$Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\big(r_t+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\big)$
- **训练步骤（Sarsa）**  
  1) 观察 $s_t$，按 $\epsilon$-greedy 选 $a_t$；  
  2) 执行动作得 $r_t, s_{t+1}$，再按策略选 $a_{t+1}$；  
  3) TD 更新 $Q(s_t,a_t)$；若未终止，令 $(s,a)\leftarrow(s_{t+1},a_{t+1})$ 循环。  
  4) 终止后进入下一个 episode（可衰减 $\epsilon$）。
- **优缺点**  
  - 优：方差低于 MC，可在线更新；对短回合友好。  
  - 缺：有 bootstrap 偏差；on-policy 收敛速度可能慢于 off-policy。

## 2.3 Q-Learning（off-policy TD）
- > **思想**：仍用一步 TD，但目标用 $\max_{a'}Q(s',a')$，学习最优策略，采样策略可与目标策略不同（off-policy）。
- **理论/公式**  
  $Q(s,a)\leftarrow Q(s,a)+\alpha\big(r+\gamma \max_{a'}Q(s',a')-Q(s,a)\big)$
- **训练步骤**  
  1) 观察 $s$，行为策略（通常 $\epsilon$-greedy）选 $a$；  
  2) 得到 $(r,s')$；  
  3) 用目标 $\max_{a'}Q(s',a')$ 做 TD 更新；  
  4) 若未终止，$s\leftarrow s'$ 继续。  
  5) 衰减 $\epsilon$ 重复多 episode，逼近 $Q^*$。
- **优缺点**  
  - 优：off-policy，可复用经验，表格条件下可收敛到最优；  
  - 缺：有偏，易受高估影响（需 Double 等改进）；在函数逼近时更需稳定技巧。

---

## 2.4 重要超参数与技巧
- 学习率 $\alpha$：过大震荡，过小收敛慢；可随时间衰减或用常数步长。  
- 折扣 $\gamma$：CartPole 常用 0.99。  
- 探索率 $\epsilon$：线性/指数衰减 1.0→0.05 常见；可更快衰减提早“利用”。  
- 状态离散化：bin 过粗影响最优性，过细样本稀疏；本例默认 40，也可调低到 12–20 加快收敛。  
- 终止/截断：环境有 `truncated` 时，更新时要将终止标志纳入。

## 2.5 MC / Sarsa / Q-Learning 对比

| 维度 | MC | Sarsa (TD, on-policy) | Q-Learning (TD, off-policy) |
| --- | --- | --- | --- |
| 更新时机 | 回合结束 | 每步 | 每步 |
| 目标 | 真实回报 $G$ | $r+\gamma Q(s',a')$ | $r+\gamma\max_{a'}Q(s',a')$ |
| 偏差/方差 | 无偏，高方差 | 有偏，较低方差 | 有偏，较低方差 |
| 策略关系 | 行为=评估策略 | 行为=目标（同一 $\epsilon$-greedy） | 行为 $\epsilon$-greedy，目标贪心 |
| 收敛（表格） | 到 $Q^\pi$ | 到 $Q^\pi$ | 条件满足到 $Q^*$ |
| 适用 | 自然终止任务，直观评估 | on-policy 场景、稳健 | 追求最优、可复用样本 |

## 2.6 代码索引（结构统一）
- 公共工具：`common_tabular.py`（离散化、epsilon 调度、env 创建、epsilon-greedy）
- MC 控制：`mc_control_cartpole.py`（首访、均值回报）
- Sarsa：`sarsa_cartpole.py`（on-policy TD）
- Q-Learning：`q_learning.py`（off-policy TD）

> 运行示例（uv 环境）：  
> `uv run python 02_tabular_methods/q_learning.py`  
> `uv run python 02_tabular_methods/sarsa_cartpole.py`  
> `uv run python 02_tabular_methods/mc_control_cartpole.py`
