# AI_README - 烧结终点温度(BTP)预测系统深度理解手册

> **重要说明**：本文档旨在为后续接手的 AI 助手提供全方位、无死角的工程上下文。它不仅涵盖了"是什么"，更深入解释了"为什么"和"怎么做"。阅读本文档后，你应该能够像原作者一样精准地操控代码，避免任何逻辑上的误区或幻觉。

---

## 1. 工程核心元数据与物理背景

*   **项目定义**：一个高精度、具有不确定性量化能力的烧结终点温度（BTP）预测系统。
*   **开发状态**：**v0.1.1-alpha (生产就绪增强)**。
*   **物理对象**：大型烧结机。通过控制机速、负压等参数，确保烧结过程在预定位置（BTP）达到最高温度。
*   **版本意义**：在 v0.1.0 实验闭环基础上，新增模型/预测/健康度数据的持久化保存功能，并优化健康度模型为3状态分类，提升工业实用性。
*   **核心挑战**：
    *   **时延性**：操作动作与结果之间存在显著滞后（分钟级）。
    *   **非平稳性**：原料、班次、环境导致的数据分布随时间漂移（Distribution Shift）。
    *   **不确定性**：传感器噪声大，单点预测不足以支撑决策。
*   **技术路线**：`EnhancedTransformer` = `RevIN(仅输入端归一化)` + 输入投影 + `Time2Vec(可学习时间编码)` + `Transformer Encoder` + **物理拟合特征拼接 (Fitting Module)** + `QuantileDeltaHead`（**不确定性量化输出**）。

---

## 2. 核心架构深度拆解 (Implementation Details)

### 2.1 实验逻辑链条与模块协同
本工程的设计遵循从“原始数据”到“物理意义”再到“预测建模”的逻辑链条：
1.  **物理拟合模块 (Fitting Module)**：
    -   **核心作用**：这是连接原始传感器与深度模型的桥梁。它不参与梯度下降，而是通过**三次样条插值**（Cubic Spline）将离散、含噪的风箱温度传感器数据还原为高维连续曲线。
    -   **特征提取**：从还原的物理曲线中精准捕捉 `BTP_Pos` (烧结终点位置/峰值坐标)、`BTP_Peak` (峰值温度)、`Slope` (BTP 位置一阶导数/热量斜率) 和 `AUC` (15-24风箱区间积分面积)。
    -   **逻辑价值**：深度学习模型无需自行挖掘风箱温度间的物理关联，而是直接利用这些具有明确物理意义的特征进行预测，显著降低了收敛难度并提升了稳健性。
2.  **分位数回归头 (Quantile Head)**：
    -   **核心作用**：负责**不确定性量化 (Uncertainty Quantization)**。
    -   **输出形式**：输出 $[Q_{10}, Q_{25}, Q_{50}, Q_{75}, Q_{90}]$ 五个分位数，直接形成预测区间。
    -   **区别于物理拟合**：物理拟合负责“特征增强”，而分位数回归负责“概率预测”。两者虽都提升了模型表现，但维度完全不同。
3.  **RevIN (Reversible Instance Normalization)**：
    -   仅作用于原始输入维度，用于处理非平稳工业序列的均值/方差偏移。消融实验证明它是防止模型在工业波动中失效的最关键模块。

### 2.2 EnhancedTransformer（当前方案）
```
输入 (B, T, D)
    ↓
RevIN (仅输入端归一化，消除分布漂移)
    ↓
输入投影 (Linear)
    ↓
Time2Vec (可学习时间编码，捕获动态生产节奏)
    ↓
Transformer Encoder
    ↓
取最后时刻特征
    ↓
拼接物理拟合特征 (Fitting Module: BTP_Peak, BTP_Pos, Slope, AUC)
    ↓
QuantileDeltaHead (分位数回归：基于 Softplus 的链式结构，保证分位数不交叉)
    ↓
输出 (B, Steps, 5) -> [Q10, Q25, Q50, Q75, Q90]
    ↓
┌───────────────────────────────────────────────────────────┐
│                    结果持久化 (v0.1.1 新增)                 │
├───────────────────────────────────────────────────────────┤
│  output/models/        → 模型权重保存 (.pt 文件)            │
│  output/predictions/   → 预测结果保存 (.npy 文件)           │
│  output/health_data/   → 健康度计算输入数据保存 (.csv 文件)  │
└───────────────────────────────────────────────────────────┘
```
*   **Time2Vec**：可学习的时间表示，替代固定正弦位置编码。实验证明能显著提升对非固定周期波动的捕获能力。
*   **QuantileDeltaHead**：这种结构在数学上保证了： $Q_{10} \leq Q_{25} \leq Q_{50} \leq Q_{75} \leq Q_{90}$ 始终成立，是工业决策（如置信区间告警）的逻辑基石。
*   **结果持久化**：v0.1.1 新增功能，支持将训练好的模型、预测结果和健康度计算输入数据自动保存到 `output/` 目录下，便于生产环境部署和后续分析。

### 2.2 特征引擎 (Physics Engine) [`btp/preprocessor.py`](btp/preprocessor.py)
*   **三次样条插值**：利用 15# 到 24# 风箱的离散温度点，构建 5000 点连续物理形状。
*   **关键衍生特征**：`BTP_Pos` (峰值坐标), `BTP_Peak` (峰值温度), `Slope` (热量斜率), `AUC` (累积热量)。

### 2.3 增强损失函数 (QuantileLoss) [`btp/model.py`](btp/model.py)
*   **复合约束**：Pinball Loss + Coverage Penalty (二阶覆盖惩罚) + Min Width (防止区间坍缩) + Smoothness (趋势平滑)。

### 2.4 健康度模型 (HealthModel MDPHI v2.1) [`btp/health_model.py`](btp/health_model.py)
*   **三维评分**:
    *   **静态偏离度 (H_pos)**: 中位数与工艺目标的距离。
    *   **动态稳定性 (H_stab)**: 局部波动幅度。
    *   **趋势风险度 (H_trend)**: 偏离方向的瞬时速度。
*   **平滑与判定**:
    *   **二阶 EWMA**: 两级级联滤波，提供极度丝滑的评分曲线且无超调。
    *   **施密特触发器**: 引入迟滞带宽（Hysteresis Band），防止工况状态在阈值边缘频繁闪烁。
*   **三状态分类 (v2.1 重大改进)**:
    *   **过烧 (Overburn)**: 温度持续偏高，可能造成能源浪费和设备损耗。
    *   **正常 (Normal)**: 工况处于理想范围，生产效率最优。
    *   **欠烧 (Underburn)**: 温度持续偏低，可能导致产品质量下降。
    *   *注：原 v2.0 的五状态分类（含"疑似过烧"、"疑似欠烧"）已简化为三状态，提高了分类稳健性和可操作性。*
*   **参数优化 (v2.1 敏感度提升)**:
    | 参数 | v2.0 | v2.1 | 说明 |
    |------|------|------|------|
    | sigma_left | 0.25 | 0.2 | 左侧敏感度提升 |
    | sigma_right | 0.2 | 0.3 | 右侧敏感度调整 |
    | beta (EWMA) | 0.95 | 0.7 | 响应速度加快 |
    | thresh_normal | 65.0 | 75.0 | 正常阈值收紧 |
    | hysteresis_band | 2.0 | 3.0 | 迟滞带宽放宽 |
    | max_penalty | 0.3 | 0.5 | 最大罚分增加 |
    | initial_filter_state | 1.0 | 0.8 | 冷启动初始状态优化 |

---

## 3. 规范化目录结构

```text
btp_project/
├── btp/                      # 核心算法包
│   ├── model.py              # Enhanced Transformer 架构与分位数 Loss
│   ├── health_model.py       # MDPHI v2.1 健康度评分引擎 (3状态分类)
│   ├── preprocessor.py       # 物理特征引擎 (CubicSpline) 与数据预处理
│   ├── data_loader.py        # 滑动窗口序列构建
│   ├── trainer.py            # 训练逻辑、Warmup 与复合优化策略
│   ├── config.py             # 集中化配置管理 (TrainConfig + 新增保存参数)
│   ├── metrics.py            # 分位数专用评价指标 (PICP, PINAW, etc.)
│   └── visualizer.py         # 预测结果与健康度深度可视化
├── scripts/                  # 运行与自动化脚本
│   ├── main.py               # 生产/实验统一入口 (含结果持久化逻辑)
│   ├── tune_health_params.py # 健康度模型交互式调优台 (GUI, 旧版)
│   ├── interactive_health_tuner.py # [v0.1.2 新增] 交互式健康度参数调优器 (实时绘图)
│   ├── run_ablation.py       # 自动化消融实验调度
│   └── analyze_results.py    # 实验结果统计分析
├── output/                   # [v0.1.1 新增] 持久化输出目录
│   ├── models/               # 训练模型保存 (.pt 文件)
│   ├── predictions/          # 预测结果保存 (.npy 文件)
│   └── health_data/          # 健康度计算输入数据保存 (.csv 文件)
├── data/                     # 数据存储 (原始 Excel、Scaler 缓存、实验结果)
├── logs/                     # 训练与系统日志
├── tests/                    # 单元测试 (模型前向、插值逻辑)
└── tools/                    # 辅助工具 (文档转换、数据复制等)
```

---

## 4. 开发状态与重大重构记录

### 4.1 状态概览
- **当前阶段**: v0.1.1-alpha (生产就绪增强)
- **核心成果**: 在 v0.1.0 实验闭环基础上，新增结果持久化功能，优化健康度模型为3状态分类，验证结果：精确准确率 68.10%，模糊准确率 99.68%。

### 4.2 重构历程 (Major Refactorings)
- **2026-03-03 [Current - 下午更新]**:
    - **新增交互式健康度调优工具**: [`scripts/interactive_health_tuner.py`](scripts/interactive_health_tuner.py)
        - 支持加载已有实验数据 (`data/run_20260303_153938`)
        - 交互式调整15项健康度参数（滑块实时响应）
        - 实时绘制三种关键图表：健康度全景图、混淆矩阵、相关性散点图
        - 支持参数导出为 JSON 格式
    - **使用方式**: `python scripts/interactive_health_tuner.py`
- **2026-03-03 [上午更新]**:
    - **结果持久化功能**: 新增模型保存、预测结果保存、健康度计算输入数据保存功能，支持配置参数 `save_model`, `save_predictions`, `save_health_data`。
    - **健康度模型升级 (v2.0 → v2.1)**: 状态分类从5状态简化为3状态（过烧/正常/欠烧），提高分类稳健性。
    - **健康度参数优化**: 调整 sigma_left/right、beta、thresh_normal、hysteresis_band、max_penalty、initial_filter_state 等参数，提高敏感度。
    - **新增可配置参数**: `health_thresh_normal`, `health_thresh_fault`, `health_hysteresis_band`, `health_max_penalty`, `health_initial_filter_state`。
    - **验证结果**: 健康度评分分布更分散（49.19 ~ 91.22），状态分类能识别过烧(10.1%)和欠烧(8.1%)。
- **2026-03-02**:
    - **热力图可视化优化**: 改进 5×5 混淆矩阵热力图，使用 YlOrRd 颜色映射确保"数值越高颜色越深"，并添加金色边框凸显对角线（正确分类），更好地展示模型分类准确率。
    - **交叉验证热力图优化**: 修正颜色逻辑，对于越小越好的指标（MAE、MAPE、RMSE、Sharpness）进行反转处理，确保"颜色越绿越好"的统一语义。
- **2026-02-27**:
    - **文档同步**: 完成 README.md 与 AI_README.md 的最终对齐，记录实验 1-4 的成果。
    - **明确模块定义**: 严格区分**物理拟合模块 (Fitting Module)** 的特征提取作用与**分位数回归 (Quantile Head)** 的不确定性量化作用。
    - **健康度引擎升级**: 发布 MDPHI v2.0，引入二阶 EWMA 平滑与施密特触发器，确保业务端展示的评分极度稳健。
- **2026-02-22**:
    - **架构剪枝**: 彻底废弃 CrossFeatureAttention 与 CQR 校准，回归奥卡姆剃刀原则，大幅减少了参数量并提升了收敛稳定性。
    - **时间编码升级**: 引入 Time2Vec 替代静态 PE。
- **2026-02-18**: 
    - **数据协议修复**: 修正 BTP 目标值被误截断导致的“阶梯状失真” Bug。

---

## 6. 给"下一个 AI"的上下文 (Context for Next AI)
如果你被要求继续改进或维护此项目，请务必关注以下几点：
1.  **物理一致性**：不要随意移除 `btp/preprocessor.py` 中的物理拟合特征。消融实验证明，没有这些特征，模型在处理极端异常工况时的泛化能力会显著下降。
2.  **RevIN 的平衡**：目前 RevIN 只在输入侧开启。如果预测步长（Forecast Steps）非常长，考虑在输出侧也开启反归一化，但要注意可能带来的不稳定性。
3.  **分位数单调性**：`QuantileDeltaHead` 是核心逻辑，不要试图将其替换为普通的多输出 Linear 层，否则会出现预测区间交叉的悖论。
4.  **业务逻辑平滑**：健康度评分 $H$ 的"平滑感"来自于二阶 EWMA，调整参数时需小心相位偏移带来的延迟感。
5.  **三状态分类**：v2.1 版本将健康度状态简化为三状态（过烧/正常/欠烧），不要随意扩展状态数量，五状态分类已证明过于复杂且不稳定。
6.  **结果持久化**：训练完成后会自动保存模型、预测结果和健康度数据到 `output/` 目录，确保这些数据被妥善管理和备份。


## 5. 核心文件索引清单

| 文件 | 关键职责 | 核心代码块/类 |
| :--- | :--- | :--- |
| [`btp/config.py`](btp/config.py) | 全局超参数与实验开关 | `TrainConfig` (含 save_model, save_predictions, save_health_data) |
| [`btp/model.py`](btp/model.py) | 模型定义与损失函数 | `EnhancedTransformer`, `QuantileLoss` |
| [`btp/health_model.py`](btp/health_model.py) | 业务逻辑转化 | `HealthModel` (MDPHI v2.1, 3状态分类) |
| [`btp/preprocessor.py`](btp/preprocessor.py) | 物理特征提取 | `_extract_spline_features` |
| [`btp/visualizer.py`](btp/visualizer.py) | 可视化模块（含热力图） | `plot_diagnosis_confusion_matrix` |
| [`scripts/main.py`](scripts/main.py) | 实验流水线控制（含结果持久化） | `main()` 优先级调度逻辑 |
| [`scripts/tune_health_params.py`](scripts/tune_health_params.py) | 交互式调优 (旧版) | `HealthVisualizerApp` |
| [`scripts/interactive_health_tuner.py`](scripts/interactive_health_tuner.py) | [新增] 实时绘图调优器 | `InteractiveHealthTuner` |

---

## 7. v0.1.1 新增配置参数参考

### 7.1 结果持久化参数
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `save_model` | bool | True | 是否保存训练好的模型到 `output/models/` |
| `save_predictions` | bool | True | 是否保存预测结果到 `output/predictions/` |
| `save_health_data` | bool | True | 是否保存健康度计算输入数据到 `output/health_data/` |

### 7.2 健康度模型参数 (v2.1 新增可配置)
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `health_thresh_normal` | float | 75.0 | 正常状态阈值（原 65.0） |
| `health_thresh_fault` | float | 50.0 | 故障阈值 |
| `health_hysteresis_band` | float | 3.0 | 迟滞带宽（原 2.0） |
| `health_max_penalty` | float | 0.5 | 最大罚分比例（原 0.3） |
| `health_initial_filter_state` | float | 0.8 | 冷启动初始状态（原 1.0） |

---
*Generated by Documentation Writer | Last Updated: 2026-03-03*
