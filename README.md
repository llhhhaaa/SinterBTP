# SinterBTP: 烧结终点智能预测与工况识别系统 (Intelligent BTP Prediction & Condition Recognition)

[![Project Status: Alpha](https://img.shields.io/badge/Status-v0.1.0--alpha-blue?style=flat-square)](https://github.com/llhhhaaa/btp_project_cloud)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square)](LICENSE)
[![Vibe Coding](https://img.shields.io/badge/Vibe%20Coding-100%25-blueviolet?style=flat-square)](https://twitter.com/search?q=vibecoding)
[![Academic](https://img.shields.io/badge/Project-Undergraduate%20Thesis-blue?style=flat-square)](https://github.com/llhhhaaa/btp_project_cloud)

> **简介**：**SinterBTP** 是一套专为烧结工艺设计的工业级预测框架。系统基于**物理启发式特征引擎 (Physics-Inspired Feature Engine)** 与**改进型 Transformer 架构**，实现了对烧结终点 (Burn-through Point) 的高精度预测。通过三次样条插值物理重构与分位数回归技术，系统不仅能提供点预测，还能输出具备物理意义的置信区间与实时工况健康度评分。

---

## 🏗️ 快速门户 (Quick Portal)

### 导航菜单
- [📍 核心特性](#核心特性)
- [🚀 运行指南](#运行指南)
- [🧠 物理特征引擎](#物理特征引擎)
- [🏗️ 模型架构深度解析](#模型架构深度解析)
- [🏥 健康度模型与工况识别](#健康度模型与工况识别-mdphi-v20)
- [📉 损失函数理论](#损失函数理论)
- [🛠️ 工程实现细节](#工程实现细节)
- [📜 开发日志](#开发日志)
- [⚖️ 项目治理](#项目治理-project-governance)

### 核心特性
- **物理启发特征**：利用 `CubicSpline` 还原 5000 点连续温度曲线，提取 BTP 位置、峰值温度、热量斜率及累积积分面积（AUC）。
- **非平稳性建模**：集成 **RevIN (Reversible Instance Normalization)**，在保留局部波动的条件下消除样本间的分布漂移。
- **自适应周期项**：引入 **Time2Vec** 可学习时间编码，自动识别烧结过程中的生产节奏周期。
- **多维度分位数预测**：直接输出 $[Q_{10}, Q_{25}, Q_{50}, Q_{75}, Q_{90}]$ 预测区间，解决单一残差无法表征不确定性的痛点。
- **全自动化消融实验**：一键量化 RevIN、Time2Vec、物理特征及校准模块对性能的边际贡献。

### 运行指南
```bash
# 1. 环境准备
pip install -r requirements.txt

# 2. 基础运行 (训练 + 验证)
python scripts/main.py

# 3. 运行自动化对比、消融、深度测试实验 (在 config.py 中设置 enable_ablation_study = True)
python scripts/main.py
```

---

## 🧠 深度技术档案 (Technical Deep Dive)

### 物理特征引擎 (Physics Engine)
系统不再直接将原始 15# - 24# 风箱温度喂入模型，而是通过物理重构层提取形态描述符。

#### 三次样条插值 (Cubic Spline Interpolation)
对于观测到的离散点 $(x_i, y_i)$ ，我们构建分段三次多项式 $S(x)$ ，满足：

$$
S(x_i) = y_i, \quad S'_i(x_i) = S'_{i+1}(x_i), \quad S''_i(x_i) = S''_{i+1}(x_i)
$$

通过对插值函数进行 **5000 点高密度采样**，我们精确定位：
1.  **BTP 峰值位置** ($x_{peak}$) ： $\arg\max_{x} S(x)$ ，表征烧结终点的精确物理坐标。
2.  **BTP 峰值温度** ($y_{peak}$) ： $\max S(x)$ ，表征烧结过程的最高热强度。
3.  **热量分布斜率** ($Slope$) ： $S'(x_{peak})$ ，反映热交换的剧烈程度。
4.  **累积热量面积** ($AUC$) ： $\int_{15}^{24} S(x) dx$ ，反映总体的热输入水平。

---

### 模型架构深度解析 (Architecture Details)

#### RevIN: 消除分布漂移
针对工业传感器数据的非平稳性，模型在输入端执行实例级归一化：

$$
\hat{x}^{(i)} = \gamma \cdot \frac{x^{(i)} - \mathbb{E}[x^{(i)}]}{\sqrt{\text{Var}[x^{(i)}] + \epsilon}} + \beta
$$

在输出端，模型通过物理状态路径（Phys-Path）保留原始量纲信息，配合 `QuantileDeltaHead` 还原预测尺度。

#### Time2Vec: 可学习周期性

$$
t2v(\tau)[i] = 
\begin{cases} 
\omega_i \tau + \phi_i, & i = 0 \\ 
\sin(\omega_i \tau + \phi_i), & 1 \leq i \leq d 
\end{cases}
$$

相比固定频率的 PE（Position Encoding），Time2Vec 允许模型通过反向传播自动学习与机速或料层厚度相关的生产周期频率。

#### QuantileDeltaHead: 非单调性交叉保证
为了彻底消除传统分位数回归中 $Q_{high} < Q_{low}$ 的逻辑错误（Quantile Crossing），我们设计了基于 `Softplus` 的链式输出结构：

$$
\begin{aligned}
Q_{50} &= \text{Linear}(h) \\
Q_{25} &= Q_{50} - \text{Softplus}(d_{low1}) \\
Q_{10} &= Q_{25} - \text{Softplus}(d_{low2}) \\
Q_{75} &= Q_{50} + \text{Softplus}(d_{high1}) \\
Q_{90} &= Q_{75} + \text{Softplus}(d_{high2})
\end{aligned}
$$

这种结构在数学上保证了： $Q_{10} \leq Q_{25} \leq Q_{50} \leq Q_{75} \leq Q_{90}$ 始终成立。

---

### 🏥 健康度模型与工况识别 (MDPHI v2.0)

**烧结 BTP 多维动态势能健康度体系 (Multi-Dimensional Potential Health Index)** 是本系统对预测结果进行业务决策转化的核心模块。它通过对预测分位数分布的深度建模，量化当前生产状态的稳健性。

#### 核心评分维度
1.  **静态偏离度** ($H_{pos}$) ：基于高斯势能函数 $e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ ，衡量 BTP 预测中位数相对于工艺目标值（如 22.5m）的物理偏差，并根据预测区间的宽度（置信度）引入罚分机制。
2.  **动态稳定性** ($H_{stab}$) ：基于滚动窗口标准差 $\sigma_{roll}$ 的反比函数，捕捉预测序列的局部波动频率，识别过程抖动。
3.  **趋势风险度** ($H_{trend}$) ：利用方向导数 $\text{sign}(\text{bias}) \cdot v_t$ 评估当前 BTP 变化的趋势。若变化方向正快速偏离目标值，则大幅削减健康度评分。

#### 数学实现与平滑逻辑
-   **乘性融合**: 最终健康度评分采用加权幂乘融合 $H = H_{pos}^{w_1} \cdot H_{stab}^{w_2} \cdot H_{trend}^{w_3}$ ，确保任何一维度的溃败都会反映在总分上。
-   **二阶 EWMA 平滑**: 采用两级级联的指数加权移动平均（Second-Order Cascaded EWMA），在有效抑制传感器毛刺噪声的同时，保证无超调（No Overshoot）特性，使评分曲线丝滑且具有物理意义。
-   **施密特触发器 (Schmidt Trigger)**: 针对工况识别引入迟滞判定逻辑。在 65（健康）和 38（故障）分界线附近设置滞后带宽，防止因微小波动导致的工况频繁闪烁。

#### 工况映射体系
系统将健康度与物理偏差方向耦合，自动映射至 5 个生产等级：
-   **过烧 (Over-burn)**: 健康度极低且偏小。
-   **疑似过烧**: 健康度中低。
-   **正常 (Normal)**: 健康度高且稳定。
-   **疑似欠烧**: 健康度中低。
-   **欠烧 (Under-burn)**: 健康度极低且偏大。

#### 调优工具：`scripts/tune_health_params.py`
为了适配不同烧结机的物理特性，提供 GUI 调优台。它支持：
-   实时加载预测报告，通过交互式滑块调整 $\mu, \sigma, w$ 等 10 余项关键参数。
-   基于 **Corr & MAE** 的双目标自动优化算法，寻找使预测健康度与真值健康度最匹配的参数组合。

---

### 📉 损失函数理论 (Loss Function Theory)

本系统采用多约束复合损失函数 $\mathcal{L}_{total}$ ，由以下部分组成：

1.  **Pinball Loss (核心)**：

$$
\begin{aligned}
\mathcal{L}_{pin}(y, \hat{q}) = \sum_{\tau} \max(\tau(y - \hat{q}), (\tau - 1)(y - \hat{q}))
\end{aligned}
$$

2.  **覆盖率惩罚 (Coverage Penalty)**：

    当真值落在 $[Q_{10}, Q_{90}]$ 之外时，施加二阶惩罚：

$$
\begin{aligned}
\mathcal{L}_{cov} = \text{ReLU}(Q_{10} - y)^2 + \text{ReLU}(y - Q_{90})^2
\end{aligned}
$$

3.  **最小宽度约束 (Min Width)**：

    防止区间坍缩，要求区间宽度不小于 Batch 波动率的一定比例：

$$
\begin{aligned}
\mathcal{L}_{width} = \text{ReLU}(\eta \cdot \sigma_{batch} - (Q_{90} - Q_{10}))^2
\end{aligned}
$$

4.  **二阶差分平滑 (Smoothness)**：

    要求预测趋势的变化率与真实变化率匹配，消除阶梯状波动：

$$
\begin{aligned}
\mathcal{L}_{smooth} = \|\Delta \hat{y} - \Delta y\|_2^2
\end{aligned}
$$

---

### 🛠️ 工程实现细节 (Engineering Implementation)

-   **分位数截断策略**：预处理阶段仅对 **原始传感器列** 进行 1%/99% 截断，**计算特征 (BTP_*)** 与 **目标列** 保持原始值，防止截断导致的物理信息丢失（即“阶梯状失真”）。
-   **高效训练流**：
    -   使用 `Multi-worker prefetch` 保证 GPU 零等待。
    -   `Warmup` 学习率策略，在前 5 个 Epoch 线性增加权重，避免 `Softplus` 在初始阶段由于梯度过大导致模型崩溃。
-   **自动化消融实验**：系统会自动遍历 `[base, no_revin, no_fitting, no_time2vec]` 四种变体，并在结果中自动计算 `Wilcoxon` 符号秩检验，确保改进具有统计学意义。

---

##  🛠️项目结构 (Project structure)

### 标准化目录树
```text
btp_project/
├── btp/                      # 核心算法库
│   ├── model.py              # Enhanced Transformer 架构
│   ├── health_model.py       # MDPHI v2.0 健康度评估引擎
│   ├── preprocessor.py       # 物理特征引擎与数据截断
│   ├── calibrator.py         # 区间校准 (V4.1, Legacy)
│   ├── trainer.py            # 训练逻辑与复合损失
│   └── config.py             # 全局超参数与实验开关
├── scripts/                  # 运行脚本
│   ├── main.py               # 主入口 (支持标准/消融/CV)
│   ├── tune_health_params.py # 健康度参数交互式调优工具
│   ├── analyze_ablation.py   # 消融实验自动化统计
│   └── gui.py                # 在线推理演示 (WIP)
├── data/                     # 数据与缓存
└── logs/                     # 训练审计日志
```

## 📜 开发日志 (Developer Logs)

### 2026-02-22: 架构定型与性能飞跃
-   **[Feature] 引入 Time2Vec**：
    -   *Why*: 之前的 Positional Encoding 是静态的，无法适配烧结机速变化引起的周期偏移。
    -   *Result*: MAE 下降 3.3%，对长周期波动捕获更准。
-   **[Cleanup] 完全废弃 CQR (Conformal Quantile Regression)**：
    -   *Why*: 消融实验证明 CQR 在线校准由于反馈滞后，不仅没有提高覆盖率，反而导致了显著的相位偏移。
    -   *Decision*: 转而强化损失函数中的 `Coverage Penalty`。
-   **[Refactor] CQR 校准器 V4.1 (保留供研究使用)**：
    -   将响应速度提升 40 倍，引入自适应学习率 $\alpha_{adaptive}$ 。

### 2026-02-21: 跨特征注意力尝试 (失败)
-   **[Experiment] 新增 CrossFeatureAttention 模块**：
    -   *Attempt*: 试图让各风箱温度通道在时间融合前先进行空间特征交互。
    -   *Conclusion*: **消融实验证弥**。该模块不仅增加了 15% 的显存消耗，且由于参数冗余导致收敛变慢，MAE 上升 2.4%。
    -   *Action*: 已于 02-22 完全移除代码。

### 2026-02-18: 关键性 Bug 修复
-   **[Fix] 阶梯状失真修复**：
    -   *Problem*: 预测曲线在峰值处出现诡异的平顶。
    -   *Cause*: 预处理脚本误对 BTP 目标值进行了 1.5 IQR 截断。
    -   *Solution*: 重新设计 `DataPreprocessor` 的截断白名单。
-   **[Architecture] 简化为 RevIN + Transformer**：
    -   *Why*: 针对“为加而加”的模块（如趋势分解、频率增强）进行了大规模剪枝，遵循奥卡姆剃刀原则。

---



### ⚖️ 项目治理 (Project Governance)

#### 许可证 (License)
本项目采用 [Apache License 2.0](LICENSE) 协议开源。

#### 📊 数据保密性声明 (Data Privacy Statement)
- **原始生产数据不包含在本公开仓库中。**
- `data/raw/` 目录下仅提供经过脱敏处理的少量示例数据，用于验证流水线可运行性。
- 完整数据集需经相关机构授权，严禁擅自传播。

#### ⚠️ 免责声明 (Disclaimer)
- **学术研究声明**：本项目为本科毕业设计的学术研究成果，旨在探索机器学习在烧结工艺优化中的应用可能性。**本系统不构成任何形式的工业生产指导建议。**
- **预测结果使用警告**：本系统输出的预测值（包括点预测和区间预测）仅供研究参考，**严禁**未经充分验证直接接入烧结机自动控制系统。
- **责任限制**：在法律允许的最大范围内，本项目作者不对因使用本软件或其预测结果而导致的任何直接、间接、附带、特殊或后果性损害承担责任。

### 📖 引用 (Citation)
如果本项目对你的研究或工作有帮助，请考虑引用：
```bibtex
@software{btp_prediction_2026,
  author       = {[llhhhaaa]},
  title        = {烧结终点智能预测与工况识别系统：基于物理启发特征与改进 Transformer 的 BTP 预测},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/llhhhaaa/btp_project_cloud},
}
```

### 🤝 贡献指南 (Contributing)
欢迎通过 Pull Request 或 Issue 提交改进建议。请确保您的代码符合项目现有的代码风格，并提供必要的测试用例。

### 📬 联系方式 (Contact)
- **GitHub Issues**：技术问题、Bug 报告


### 🙏 致谢 (Acknowledgments)
- **算法致谢**：本项目参考了 **RevIN** (Kim et al., ICLR 2022) 与 **Time2Vec** (Kazemi et al., ICLR 2020) 的设计思想。

---
*烧结工程工况识别项目组 | Last Updated: 2026-02-27*
