# 第 7 章：表示学习与 RLPD（Representation Learning with Policy Distillation）

本章关注“学好状态表征”与“策略蒸馏”在 RL 中的结合：用已训练的 Teacher 策略生成带标签的数据，Student 通过表征网络+策略头拟合 Teacher，从而得到更紧凑或泛化更好的策略。

---

## 7.1 为什么要做表示学习
- 高维观测（传感器/视觉）下，直接在原空间学策略样本效率低、泛化差。
- 学到的低维表征可复用到新任务（迁移/微调），或在小网络上部署。
- 在 off-policy 设置下，可以用已有经验和 Teacher 策略批量“蒸馏”。

## 7.2 RLPD 核心思路
- 有一个性能不错的 Teacher 策略（如已训练的 SAC）。  
- 收集 $(s, a_{\text{teacher}})$ 数据集。  
- Student 包含 Encoder（表征）+ Policy 头，最小化 Student 输出与 Teacher 动作分布的差异（离线模仿）。  
- 可选：冻结 Teacher，不再交互；或在蒸馏后再做少量在线微调。

## 7.3 训练目标
- 对连续动作，用高斯策略匹配：最小化 KL / MSE。  
- 典型损失：  
  $L_{\text{policy}} = \mathbb{E}_{s\sim\mathcal{D}}\big[ \| \mu_\theta(s) - a_{\text{teacher}}\|^2 \big]$  
  也可用 Teacher 的对数概率做交叉熵或 KL。
- 若 Student 编码器输出 $z$，可附加表征正则（如 L2 / 对比损失）。

## 7.4 步骤梳理
1) 训练或载入 Teacher（本示例用 SAC Pendulum）。  
2) 采样数据集 $\mathcal{D}=\{(s,a_T)\}$，固定 Teacher。  
3) Student：Encoder + Actor，高斯策略输出均值/对数方差。  
4) 监督蒸馏：最小化动作 MSE（或 KL），可混入动作噪声增强。  
5) 可选在线微调：用 Student 继续 SAC/TD3 训练，初始化为蒸馏参数。

## 7.5 优缺点
- 优：样本效率高（离线批量）；可压缩、可迁移；不需要环境交互即可蒸馏。  
- 缺：Teacher 质量决定上限；分布外数据会导致 Student 偏差；仍需少量在线验证/微调。

## 7.6 代码索引
- `rlpd_pendulum.py`：小规模示例——若有 Teacher checkpoint 则直接蒸馏；否则先快速训练 Teacher（SAC），再蒸馏到轻量 Student。

> 运行示例：`uv run python 07_representation_rl/rlpd_pendulum.py`
