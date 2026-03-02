# AI_README - 烧结终点温度(BTP)预测系统深度理解手册

> **重要说明**：本文档旨在为后续接手的 AI 助手提供全方位、无死角的工程上下文。它不仅涵盖了"是什么"，更深入解释了"为什么"和"怎么做"。阅读本文档后，你应该能够像原作者一样精准地操控代码，避免任何逻辑上的误区或幻觉。

---

## 1. 工程核心元数据与物理背景

*   **项目定义**：一个高精度、具有不确定性量化能力的烧结终点温度（BTP）预测系统。
*   **开发状态**：**v0.1.0-alpha (实验成果闭环)**。
*   **物理对象**：大型烧结机。通过控制机速、负压等参数，确保烧结过程在预定位置（BTP）达到最高温度。
*   **版本意义**：本次大版本更新完成了从数据探索到深度验证的全流程闭环。通过消融实验确定了 **RevIN** 与 **Fitting Module** 的不可替代性，并通过 **Diagnostics** 验证了模型的稳健性。
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
```
*   **Time2Vec**：可学习的时间表示，替代固定正弦位置编码。实验证明能显著提升对非固定周期波动的捕获能力。
*   **QuantileDeltaHead**：这种结构在数学上保证了： $Q_{10} \leq Q_{25} \leq Q_{50} \leq Q_{75} \leq Q_{90}$ 始终成立，是工业决策（如置信区间告警）的逻辑基石。

### 2.2 特征引擎 (Physics Engine) [`btp/preprocessor.py`](btp/preprocessor.py)
*   **三次样条插值**：利用 15# 到 24# 风箱的离散温度点，构建 5000 点连续物理形状。
*   **关键衍生特征**：`BTP_Pos` (峰值坐标), `BTP_Peak` (峰值温度), `Slope` (热量斜率), `AUC` (累积热量)。

### 2.3 增强损失函数 (QuantileLoss) [`btp/model.py`](btp/model.py)
*   **复合约束**：Pinball Loss + Coverage Penalty (二阶覆盖惩罚) + Min Width (防止区间坍缩) + Smoothness (趋势平滑)。

### 2.4 健康度模型 (HealthModel MDPHI v2.0) [`btp/health_model.py`](btp/health_model.py)
*   **三维评分**:
    *   **静态偏离度 (H_pos)**: 中位数与工艺目标的距离。
    *   **动态稳定性 (H_stab)**: 局部波动幅度。
    *   **趋势风险度 (H_trend)**: 偏离方向的瞬时速度。
*   **平滑与判定**:
    *   **二阶 EWMA**: 两级级联滤波，提供极度丝滑的评分曲线且无超调。
    *   **施密特触发器**: 引入迟滞带宽（Hysteresis Band），防止工况状态在阈值边缘频繁闪烁。

---

## 3. 规范化目录结构

```text
btp_project/
├── btp/                      # 核心算法包
│   ├── model.py              # Enhanced Transformer 架构与分位数 Loss
│   ├── health_model.py       # MDPHI v2.0 健康度评分引擎
│   ├── preprocessor.py       # 物理特征引擎 (CubicSpline) 与数据预处理
│   ├── data_loader.py        # 滑动窗口序列构建
│   ├── trainer.py            # 训练逻辑、Warmup 与复合优化策略
│   ├── config.py             # 集中化配置管理 (TrainConfig)
│   ├── metrics.py            # 分位数专用评价指标 (PICP, PINAW, etc.)
│   └── visualizer.py         # 预测结果与健康度深度可视化
├── scripts/                  # 运行与自动化脚本
│   ├── main.py               # 生产/实验统一入口
│   ├── tune_health_params.py # 健康度模型交互式调优台 (GUI)
│   ├── run_ablation.py       # 自动化消融实验调度
│   └── analyze_results.py    # 实验结果统计分析
├── data/                     # 数据存储 (原始 Excel、Scaler 缓存、实验结果)
├── logs/                     # 训练与系统日志
├── tests/                    # 单元测试 (模型前向、插值逻辑)
└── tools/                    # 辅助工具 (文档转换、数据复制等)
```

---

## 4. 开发状态与重大重构记录

### 4.1 状态概览
- **当前阶段**: v0.1.0-alpha (实验成果闭环)
- **核心成果**: 完成了四个关键实验（1. 数据分析；2. 统计显著性；3. 消融实验；4. 鲁棒性诊断），确立了系统的学术与工程基准。

### 4.2 重构历程 (Major Refactorings)
- **2026-03-02 [Current]**:
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

## 6. 给“下一个 AI”的上下文 (Context for Next AI)
如果你被要求继续改进或维护此项目，请务必关注以下几点：
1.  **物理一致性**：不要随意移除 `btp/preprocessor.py` 中的物理拟合特征。消融实验证明，没有这些特征，模型在处理极端异常工况时的泛化能力会显著下降。
2.  **RevIN 的平衡**：目前 RevIN 只在输入侧开启。如果预测步长（Forecast Steps）非常长，考虑在输出侧也开启反归一化，但要注意可能带来的不稳定性。
3.  **分位数单调性**：`QuantileDeltaHead` 是核心逻辑，不要试图将其替换为普通的多输出 Linear 层，否则会出现预测区间交叉的悖论。
4.  **业务逻辑平滑**：健康度评分 $H$ 的“平滑感”来自于二阶 EWMA，调整参数时需小心相位偏移带来的延迟感。


## 5. 核心文件索引清单

| 文件 | 关键职责 | 核心代码块/类 |
| :--- | :--- | :--- |
| [`btp/config.py`](btp/config.py) | 全局超参数与实验开关 | `TrainConfig` |
| [`btp/model.py`](btp/model.py) | 模型定义与损失函数 | `EnhancedTransformer`, `QuantileLoss` |
| [`btp/health_model.py`](btp/health_model.py) | 业务逻辑转化 | `HealthModel` (MDPHI v2.0) |
| [`btp/preprocessor.py`](btp/preprocessor.py) | 物理特征提取 | `_extract_spline_features` |
| [`btp/visualizer.py`](btp/visualizer.py) | 可视化模块（含热力图） | `plot_diagnosis_confusion_matrix` |
| [`scripts/main.py`](scripts/main.py) | 实验流水线控制 | `main()` 优先级调度逻辑 |
| [`scripts/tune_health_params.py`](scripts/tune_health_params.py) | 交互式调优 | `HealthVisualizerApp` |

---
*Generated by Documentation Writer | Last Updated: 2026-03-02*
