# 强化学习系统教程（含理论 + 代码）

本仓库是一份**从零到进阶**的强化学习（Reinforcement Learning, RL）教程工程：

- 每个章节单独文件夹
- **理论用 Markdown (`.md`)**
- **代码用 Python (`.py`)**
- 离散控制算法统一用 **CartPole-v1**
- 连续控制算法统一用 **Pendulum-v1**
- 一些高级章节（如 MDP 基础、表征学习等）可能只有理论，没有代码

---

## 环境安装

1. 建议使用虚拟环境（venv / conda / uv 均可）
2. 安装依赖：

```bash
pip install -r requirements.txt
```

### 环境验证（渲染测试）

安装好依赖后，可用脚本快速确认本机渲染链路是否正常（需要图形界面）：

```bash
uv run python scripts/test_gym_render.py --seconds 5 --verbose
```

默认测试 `CartPole-v1`；能弹出窗口并在终端看到日志，即表示 Gym/Gymnasium 渲染可用。可用 `--env Pendulum-v1` 切换环境。

---

## 推荐学习路线图

### 第 0 阶段：环境跑通

先确保你能跑起来一个简单脚本，例如：

```bash
python 02_tabular_methods/q_learning.py
```

如果窗口有 CartPole 可以正常晃来晃去，环境就没问题了。

---

### 第 1 阶段：基础理论

对应文件夹：`01_basics/`

建议阅读顺序：

1. `1.1_reinforcement_learning_intro.md`  
   - 什么是 RL、Agent / Environment、奖励、回报、折扣因子

2. `1.2_elements_of_rl.md`  
   - 状态、动作、策略、价值函数、回报等正式定义

3. `2.1_markov_decision_process.md`  
   - MDP，转移概率，Bellman 方程

4. `1.3_classification_of_rl.md`  
   - Value-based / Policy-based / Actor-Critic / On-policy / Off-policy

5. 动态规划代码小样例：`value_iteration_grid.py`  
   - 4x4 GridWorld 的表格价值迭代，演示 Bellman 最优方程与贪心策略抽取

---

### 第 2 阶段：表格方法（Tabular RL）

文件夹：`02_tabular_methods/`

- `theory.md`：
  - MC、TD、Q-learning 核心公式
- `q_learning.py`：
  - 在 CartPole 上做离散化 + Q 表

推荐运行：

```bash
python 02_tabular_methods/q_learning.py
```

> 建议在这里加自己的一些改动，比如不同的 epsilon-greedy 策略、学习率调度等。

---

### 第 3 阶段：深度 Q 网络 DQN

文件夹：`03_dqn/`

- `theory.md`：
  - 函数逼近、经验回放、目标网络
- `dqn_cartpole.py`：
  - 一个标准 DQN 实现

运行：

```bash
python 03_dqn/dqn_cartpole.py
```

可以观察：
- 有没有收敛到满分（CartPole 200）
- 不同网络结构 / 学习率对性能的影响

---

### 第 4 阶段：策略梯度（Policy Gradient）

文件夹：`04_policy_gradient/`

- `theory.md`：
  - REINFORCE 推导、策略梯度定理、高方差问题
- `reinforce_cartpole.py`：
  - 直接输出动作概率，按 $ \log \pi(a|s) G_t $ 更新

运行：

```bash
python 04_policy_gradient/reinforce_cartpole.py
```

观察：
- 收敛速度 vs DQN
- 回报曲线抖动更大（高方差）

---

### 第 5 阶段：Actor-Critic / PPO

文件夹：`05_actor_critic/`

- `theory.md`：
  - Actor-Critic 框架、优势函数、A2C、PPO 思路
- `a2c_cartpole.py`：
  - 单步 TD Advantage 更新
- `ppo_cartpole.py`：
  - Clip PPO 实现，典型 on-policy 算法

运行：

```bash
python 05_actor_critic/a2c_cartpole.py
python 05_actor_critic/ppo_cartpole.py
```

这一步你会看到：
- 引入 Critic 后，策略梯度更稳定
- PPO 的 Clip 技巧，如何防止策略更新过头

---

### 第 6 阶段：Off-Policy 深度 RL（连续动作）

文件夹：`06_offpolicy_rl/`

- `theory.md`：
  - Off-policy 与 Replay Buffer
  - DDPG / TD3 / SAC 的核心思想
- `sac_tutorial.md`：  
  - **SAC 专题长文教程**（动机 + 推导 + 代码逐段讲解）
- 代码：
  - `ddpg_pendulum.py`
  - `td3_pendulum.py`
  - `sac_pendulum.py`

运行：

```bash
python 06_offpolicy_rl/ddpg_pendulum.py
python 06_offpolicy_rl/td3_pendulum.py
python 06_offpolicy_rl/sac_pendulum.py
```

本章开始切到 **Pendulum-v1** 连续动作任务，是工业界/机器人里非常常见的设置。

---

### 第 7 阶段：表示学习 + RLPD

文件夹：`07_representation_rl/`

- `theory.md`：
  - 表示学习在 RL 中的角色
  - RLPD 思路：表征学习 + 策略蒸馏
- `rlpd_pendulum.py`：
  - 用已经训练好的 SAC 作为 Teacher
  - 采样一批 $(s, a_{teacher})$
  - 训练 Student（带 Encoder）模仿 Teacher 动作
  - 对比 Teacher vs Student 在环境中的表现

运行顺序建议：

```bash
# 先训练 Teacher（SAC）
python 06_offpolicy_rl/sac_pendulum.py

# 再做 RLPD 式 Student 蒸馏
python 07_representation_rl/rlpd_pendulum.py
```

---

### 第 8 阶段：评估与工具

文件夹：`08_evaluation_and_tools/`

- `theory.md`：
  - 如何系统评估 RL 算法（平均回报、方差、学习曲线）
  - 如何统一不同算法的评估接口
- `common/`：
  - `env_utils.py`：封装 CartPole / Pendulum 创建
  - `nets.py`：通用 MLP / Actor-Critic 网络（可选）
  - `utils.py`：折扣回报、保存模型等工具
- `evaluate_policy.py`：
  - 统一载入多个算法的模型，对比其平均回报

---

### 第 9 阶段：附录

文件夹：`09_appendix/`

- `notation_and_math.md`：符号表、常用公式
- `references.md`：论文与推荐教材
- `glossary.md`：术语表（value-based、bootstrapping、entropy bonus 等）

---

## 推荐使用方式

- 把本仓库当成「强化学习课程的配套代码 + 讲义」
- 每完成一章：
  1. 先看 `theory.md` / 专题 `*.md`
  2. 再直接运行相应 `.py`
  3. 最后修改一些超参数或网络结构，观察变化

---

## 常见问题

**Q：我想用 Jupyter Notebook 学习怎么办？**  
A：你可以把每一章的 md 和 py 合并成一个 notebook（md 做 markdown cell，py 做 code cell），整体结构不需要变。

**Q：我想换成自己的环境（比如自定义机器人）？**  
A：建议保留：
- 算法的「核心训练 loop」
- 通用工具（Replay Buffer、Actor/Critic 网络）
然后只改：
- `env_utils.py` 里创建环境的函数
- 状态维度、动作维度、动作的缩放方式

---

## 致谢

本教程偏重「工程 + 数学直觉」，适合作为个人学习笔记、组内培训或课程实验基础。  
你可以根据自己项目 / 论文需求，在现有框架上继续扩展（如加入多智能体、模型式 RL、Offline RL 等）。
