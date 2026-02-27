# Copyright (c) 2026
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List

from pathlib import Path

# 路径初始化：获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data"
LOG_ROOT = BASE_DIR / "logs"

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(LOG_ROOT, exist_ok=True)


@dataclass
class TrainConfig:
    # ==========================================
    # 0) 系统与消融实验开关 (System & Ablation)
    # ==========================================
    raw_seq_len: int = 720             # 原始高频序列长度 (EnhancedTransformer 输入)
    
    USE_DATA_CACHE: bool = True        # 是否使用数据缓存
    CACHE_DIR: str = "data/cache"      # 缓存目录
    PREGENERATE_ONLY: bool = False     # 是否仅运行预生成逻辑
    enable_cv: bool = True                # 是否开启交叉验证 (1为开启, 0为关闭)

    # 是否开启 RevIN (消除逐实例分布漂移)
    enable_revin: bool = True          # 是否开启 RevIN (消除逐实例分布漂移)
    
    # 是否开启在线自适应分位数回归 (用于修正分布漂移)
    enable_online_cqr: bool = False        # 是否开启在线自适应分位数回归 (用于修正分布漂移)
    enable_cv_cqr: bool = False            # CV 阶段是否也跑 CQR 消融对比 (raw vs calibrated)
    
    # 拟合模块开关 (Fitting Module - Enhanced Transformer 专用)
    enable_fitting_module: bool = True  # 是否启用三次样条插值衍生特征 (BTP_pos, BTP_temp, BTP_slope, BTP_auc)
    
    # Time2Vec 开关 (可学习时间编码 - ICLR 2020)
    enable_time2vec: bool = True  # 是否启用 Time2Vec 替代固定正弦位置编码
    
    # 模型架构选择
    model_type: str = "enhanced_transformer"  # 使用 EnhancedTransformer
    # MC Dropout 配置 (用于不确定性量化)
    mc_samples: int = 1                # MC采样次数 (1=禁用 MC Dropout)

    # ==========================================
    # 统计显著性参数
    # ==========================================
    enable_model_comparison: bool = False     # 是否运行模型对比实验
    stats_alpha: float = 0.05          # 显著性水平
    comparison_models: List[str] = field(default_factory=lambda: [
      #  "baseline_transformer",
      #  "baseline_lstm",
      #  "baseline_gru"
    ])
    
    # ==========================================
    # 消融实验开关 (Ablation Study)
    # ==========================================
    
    # 消融实验模式
    enable_ablation_study: bool = False        # 是否运行消融实验
    ablation_modules: List[str] = field(default_factory=list)  # 要消融的模块列表 (旧版兼容)
    ablation_variants: List[str] = field(default_factory=lambda: [
        'base',           # 基准模型（RevIN + Fitting + Time2Vec）
        'no_revin',       # 无 RevIN
        'no_fitting',     # 无拟合模块
        'no_time2vec',    # 无 Time2Vec（使用固定正弦位置编码）
    ])
    
    num_transformer_layers: int = 2           # Transformer 层数 (默认2层)
    ffn_dim_multiplier: float = 4.0           # FFN 维度倍数
    
    # ==========================================
    # 深度验证模式
    # ==========================================
    enable_model_diagnostics: bool = False      # 是否运行 Model Diagnostics 深度验证实验
    
    # Model Diagnostics 子模块开关
    enable_residual_analysis: bool = True     # 残差分析
    enable_robustness_test: bool = True       # 鲁棒性测试
    enable_hyperparameter_sensitivity: bool = True  # 超参数敏感性分析
    
    # 超参数敏感性测试配置
    hyperparam_sensitivity_config: dict = field(default_factory=lambda: {
        'learning_rate': [0.0001, 0.0005, 0.001, 0.002, 0.005],  # 学习率范围
        'hidden_size': [64, 128, 256],                           # 隐藏层大小
        'num_layers': [1, 2, 3],                                 # 层数
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],                   # Dropout 率
        'epochs_per_test': 30,                                   # 每个配置训练的 epoch 数
        'repeat_times': 3                                        # 每个配置重复次数（取平均）
    })
    
    # Model Diagnostics 输出目录
    diagnostics_output_dir: str = "实验记录，用于论文叙事/4.Model Diagnostics (深度验证)"

    # ==========================================
    # 1) 模型架构配置 (Architecture)
    # ==========================================
    hidden_size: int = 128             # 隐藏层维度 (对于360步序列，128足够)
    dropout: float = 0.3               # Dropout 率 (MC Dropout 关键参数，0.2-0.3 较好)
    attn_heads: int = 4                # Transformer 头数 (hidden_size 需能被 attn_heads 整除)
    multiscale_kernels: List[int] = field(default_factory=list)  # 多尺度卷积核大小（PatchTST方案不使用）
    patch_size: int = 12               # PatchTST: patch长度
    
    # ==========================================
    # 2) 数据处理与序列设定 (Data & Sequence)
    # ==========================================
    excel_path: str = ""               # 数据源文件路径
    target_column: str = "北侧_计算BTP位置" # 预测目标
    forecast_steps: int = 1            # 预测未来步数
    
    # 预测区间配置
    prediction_offset: int = 60        # 预测起始偏移量（从输入序列末尾跳过多少行开始预测）
    target_step_weight: float = 5.0    # 目标步（最后一步）的损失权重
    
    # 数据清洗逻辑
    target_smooth_span: int = 0       # 目标 EMA 平滑跨度 (0=不平滑, 建议 5~15)
    anchor_smooth_span: int = 0        # 锚点 EMA 平滑跨度 (0=不平滑, 建议 3~7)
    max_gap_fill_multiplier: float = 4.0 # 最大填补空洞倍数
    min_valid_ratio_core: float = 0.8  # 核心特征有效率阈值
    min_valid_ratio_aux: float = 0.5   # 辅助特征有效率阈值
    min_valid_ratio_overall: float = 0.6 
    validity_check_mode: str = "layered" 
    enable_resample: bool = False      
    strict_mode: bool = 0
    
    # ==========================================
    # 3) 交叉验证参数 (Cross Validation)
    # ==========================================
    cv_n_splits: int = 8               # CV 折数

    # ==========================================
    # 4) 训练超参数 (Hyperparameters)
    # ==========================================
    loss_final_step_weight: float = 0.9  # 最终步损失权重
    smoothness_weight: float = 0.5       # 趋势拟合权重 (导数匹配，调大有助于减少滞后)
    direction_weight: float = 0.5        # 方向一致性损失权重 (惩罚反向预测)
    coverage_weight: float = 0.5         # 覆盖率惩罚权重 (惩罚真值落在 Q10-Q90 区间外)
    min_width_weight: float = 0.3        # 最小区间宽度惩罚权重 (防止区间坍缩)
    min_width_ratio: float = 0.8         # 最小区间宽度 = ratio × batch_std
    trend_weight: float = 0.5            # 宏观趋势跟踪权重 (Q50 跟踪整体趋势方向与幅度)
    centering_weight: float = 1.0        # Q50 居中约束权重 (防止系统性偏移)
    diff_loss_weight: float = 0.5        # TrendAware-Lite: 差分匹配损失权重
    lr: float = 0.0001                   # 学习率 (Adam 优化器)
    epochs: int = 15                     # 最大训练轮数
    batch_size: int = 256                # 批大小
    val_split: float = 0.15               # 验证集比例
    test_split: float = 0.15              # 测试集比例 (0 表示不划分独立测试集)
    seed: int = 42                       # 随机种子
    num_workers: int = 4                 # 数据加载线程数
    weight_decay: float = 0.01           # L2 正则化
    patience: int = 5                    # 早停耐心值
    warmup_epochs: int = 5               # 学习率预热轮数
    min_lr: float = 1e-7                 # 最小学习率

    # 缺失值补全
    enable_gap: bool = True            
    interpolate_strategy: str = "forward" 
    optimize_gap_size: bool = True     

    # 差分预测锚点
    delta_anchor_minutes: float = 0.0833 
    delta_anchor_smooth_minutes: float = 2 

    # ==========================================
    # 5) 在线校准 (CQR 偏差平移)
    # ==========================================
    cqr_bias_window: int = 60              # 偏差滑动窗口大小
    cqr_min_samples: int = 20              # 最少校准样本数


    # ==========================================
    # 6) 业务逻辑：BTP阈值与健康度算法
    # ==========================================
    btp_L_low: float = 22.44           
    btp_L_r: float = 22.56             
    btp_L_up: float = 22.7             

    h_normal_min: float = 0.6          
    h_suspect_min: float = 0.4         
    health_out_penalty_p1: float = 0.2 
    health_out_penalty_h1: float = 0.91 
    health_out_penalty_p2: float = 0.6 
    health_out_penalty_h2: float = 0.50 
    health_center_dead_zone: float = 0.15 
    health_center_penalty_power: float = 6.0 
    health_center_K_c: float = 1.0     
    health_inertia_alpha1: float = 0.05 
    health_inertia_alpha2: float = 0.02 
    volatility_window_size: int = 5    
    lambda_volatility: float = 2.7     
    lambda_trend: float = 0.6          
    k_attenuation: float = 0.7         
    ewma_alpha_penalty: float = 0.05   
    exceed_enter_eps: float = 0.05     
    exceed_exit_eps: float = 0.03      
    
    health_mu: float = 22.56           
    health_sigma_left: float = 0.25    
    health_sigma_right: float = 0.2    
    health_sigma_limit: float = 0.15   
    health_k_stab: float = 2.0         
    health_alpha_trend: float = 2      
    health_W_pos: float = 2.79         
    health_W_stab: float = 1.46        
    health_W_trend: float = 2          
    health_beta_ewma: float = 0.97     
    
    
    enable_delta_forecast: bool = 0    # 是否预测"变化量"(Delta)
    
    # ==========================================
    # 7) 日志记录与输出配置 (Logging)
    # ==========================================
    exp_name: str = ""                 
    save_aux_outputs: bool = True      
    output_dir: str = field(default="") 

    def __post_init__(self):
        """ 初始化后的逻辑 """
        pass  # 保留用于未来扩展

    def to_dict(self):
        return asdict(self)

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
