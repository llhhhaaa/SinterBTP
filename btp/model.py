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

import logging
from typing import Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# =========================================================
# 0. Config Helpers
# =========================================================

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None: return default
    if isinstance(cfg, dict): return cfg.get(key, default)
    return getattr(cfg, key, default)

# =========================================================
# 1. RevIN: Reversible Instance Normalization (ICLR 2022)
#    训练阶段消除逐实例分布漂移，推理阶段还原原始尺度
# =========================================================

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (仅输入端).
    
    对输入沿时间轴做 instance-wise normalize (减均值除标准差)，
    消除每个窗口的局部统计差异，让模型专注于学习时序模式。
    
    注意: 本项目中输入(scaler_core空间)和输出(scaler_y空间)处于不同归一化空间，
    因此 **不做输出端反归一化**。输出头结合物理特征自行学习正确尺度。
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        """
        Args:
            num_features: 输入特征维度 D
            eps: 数值稳定性
            affine: 是否使用可学习的仿射参数 (gamma, beta)
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        输入归一化: x -> (x - mean) / std * gamma + beta
        Returns:
            x_norm, mean, std
        """
        # 沿时间轴计算每个实例的均值和标准差
        mean = x.mean(dim=1, keepdim=True).detach()  # (B, 1, D)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()  # (B, 1, D)
        std = std.clamp_min(self.eps)
        if (not torch.isfinite(std).all()) or (std.min() < self.eps * 10):
            logger.warning(
                "[RevIN] std异常: min=%.6g, max=%.6g, finite=%s",
                std.min().item(), std.max().item(), torch.isfinite(std).all().item()
            )
        
        x_norm = (x - mean) / std
        
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        
        return x_norm, mean, std

    def denormalize(self, x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        反归一化: x_norm -> (x_norm - beta) / gamma * std + mean
        Args:
            x_norm: (Batch, Steps, Features) or (Batch, Steps)
            mean, std: (Batch, 1, Features) from forward
        """
        if self.affine:
            # Broadcast dimensions if necessary
            if x_norm.ndim == 3 and self.affine_weight.shape[-1] == x_norm.shape[-1]:
                x_norm = (x_norm - self.affine_bias) / (self.affine_weight + self.eps)
        
        # Determine if we are denormalizing a specific feature subset or the whole tensor
        # Case 1: Denormalizing the target only (x_norm is B, Steps, 5 or B, Steps)
        # We assume mean/std passed here are already sliced to the target feature(s)
        if x_norm.ndim == mean.ndim:
            x = x_norm * std + mean
        else:
            # Case where x_norm has extra dim (quantiles) and mean/std are (B, 1)
            # x_norm: (B, Steps, 5), mean: (B, 1)
            x = x_norm * std.unsqueeze(-1) + mean.unsqueeze(-1)
            
        return x

# =========================================================
# 2. 损失函数：QuantileLoss + Smoothness + Coverage + MinWidth + Trend
# =========================================================
class QuantileLoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        self.steps = getattr(cfg, "forecast_steps", 5)
        self.smoothness_weight = getattr(cfg, "smoothness_weight", 0.5)
        self.direction_weight = getattr(cfg, "direction_weight", 0.0)
        # 新增三项增强损失的权重
        self.coverage_weight = getattr(cfg, "coverage_weight", 0.5)
        self.min_width_weight = getattr(cfg, "min_width_weight", 0.3)
        self.min_width_ratio = getattr(cfg, "min_width_ratio", 0.8)
        self.trend_weight = getattr(cfg, "trend_weight", 0.5)
        self.centering_weight = getattr(cfg, "centering_weight", 1.0)
        
        # 动态计算步长权重
        # 使用 target_step_weight 作为目标步（最后一步）的权重倍数
        target_step_weight = getattr(cfg, "target_step_weight", 5.0)
        if self.steps > 1:
            # 其他步权重为 1.0，最后一步（目标点）使用高权重
            step_w = torch.ones(self.steps)
            step_w[-1] = target_step_weight
            # 归一化使总权重为 steps（保持损失量级稳定）
            step_w = step_w / step_w.sum() * self.steps
        else:
            step_w = torch.tensor([1.0])
        self.register_buffer('step_weights', step_w)
        
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds: (Batch, Steps, 5)  target: (Batch, Steps) 或 (Batch, Steps, 1)
        if target.ndim == 2: target = target.unsqueeze(-1)
        total_loss_map = torch.zeros_like(target.squeeze(-1))
        for i, q in enumerate(self.quantiles):
            pred_q = preds[..., i:i+1]
            errors = target - pred_q
            loss_q = torch.max((q - 1) * errors, q * errors).squeeze(-1)
            total_loss_map += loss_q
        
        base_loss = (total_loss_map * self.step_weights.unsqueeze(0)).sum(dim=1).mean()

        # ---- 增强损失项 ----
        pred_median = preds[..., 2]       # (Batch, Steps) Q50
        pred_q10 = preds[..., 0]          # (Batch, Steps) Q10
        pred_q90 = preds[..., 4]          # (Batch, Steps) Q90
        true_vals = target.squeeze(-1)    # (Batch, Steps)

        # === A. 覆盖率惩罚 ===
        # 当真值落在 [Q10, Q90] 区间外时，施加额外惩罚
        # 这迫使模型学会输出足够宽的区间来包住波动
        below_q10 = torch.relu(pred_q10 - true_vals)  # 真值低于Q10的部分
        above_q90 = torch.relu(true_vals - pred_q90)   # 真值高于Q90的部分
        coverage_loss = (below_q10 ** 2 + above_q90 ** 2).mean()

        # === B. 最小区间宽度约束 ===
        # 用 batch 内真值的标准差作为波动率参考，惩罚区间宽度小于此值的情况
        # 防止模型为了最小化 pinball loss 而坍缩区间
        interval_width = pred_q90 - pred_q10  # (Batch, Steps)
        batch_std = true_vals.std(dim=0, keepdim=True).detach()  # (1, Steps)
        min_width_target = self.min_width_ratio * batch_std
        width_deficit = torch.relu(min_width_target - interval_width)
        min_width_loss = (width_deficit ** 2).mean()

        # === C. 宏观趋势跟踪 ===
        if self.steps > 1:
            pred_diff = torch.diff(pred_median, dim=1)
            true_diff = torch.diff(true_vals, dim=1)
            # 二阶约束：预测变化率匹配真实变化率
            smooth_loss = F.mse_loss(pred_diff, true_diff)
            # 一阶约束：方向一致性
            dir_penalty = torch.relu(-pred_median * true_vals).mean()
            # 趋势幅度匹配：不再给 1.5x 容忍度，直接要求匹配
            trend_amplitude_loss = torch.relu(true_diff.abs() - pred_diff.abs()).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=preds.device)
            dir_penalty = torch.tensor(0.0, device=preds.device)
            trend_amplitude_loss = torch.tensor(0.0, device=preds.device)

        # === D. Q50 居中约束 ===
        # 直接惩罚 Q50 与真值之间的偏差，防止系统性偏移
        # 使用 Huber Loss 对大偏差更鲁棒
        centering_loss = F.smooth_l1_loss(pred_median, true_vals)

        total_loss = base_loss + \
                     self.smoothness_weight * smooth_loss + \
                     self.direction_weight * dir_penalty + \
                     self.coverage_weight * coverage_loss + \
                     self.min_width_weight * min_width_loss + \
                     self.trend_weight * trend_amplitude_loss + \
                     self.centering_weight * centering_loss
        return total_loss


# =========================================================
# 2. 输出头 (Quantile Output Head)
# =========================================================

class QuantileDeltaHead(nn.Module):
    def __init__(self, in_dim: int, steps: int , hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.GELU(), 
            nn.Dropout(dropout),
        )
        self.out_median = nn.Linear(hidden_dim, steps)
        self.out_in_low = nn.Linear(hidden_dim, steps)
        self.out_in_high = nn.Linear(hidden_dim, steps)
        self.out_out_low = nn.Linear(hidden_dim, steps)
        self.out_out_high = nn.Linear(hidden_dim, steps)

    def forward(self, z: torch.Tensor, phys_state: Optional[torch.Tensor] = None):
        if phys_state is not None:
            z = torch.cat([z, phys_state], dim=-1)
        h = self.trunk(z)
        q50 = self.out_median(h) 
        q25 = q50 - F.softplus(self.out_in_low(h))
        q75 = q50 + F.softplus(self.out_in_high(h))
        q10 = q25 - F.softplus(self.out_out_low(h))
        q90 = q75 + F.softplus(self.out_out_high(h))
        return torch.stack([q10, q25, q50, q75, q90], dim=2)


# =========================================================
# 3. 提出的模型: Granulation + Transformer (Efficient)
# =========================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# =========================================================
# 3.1 Time2Vec: 可学习的时间编码 (ICLR 2020)
# =========================================================

class Time2Vec(nn.Module):
    """
    Time2Vec: 可学习的时间表示模块
    
    论文: "Time2Vec: Learning to Represent Time" (ICLR 2020)
    
    核心思想: 将时间位置编码为可学习的周期性表示，替代固定的正弦位置编码。
    
    公式:
        t2v(τ)[0] = ω_0 · τ + φ_0           (线性项，捕捉趋势)
        t2v(τ)[i] = sin(ω_i · τ + φ_i)      (周期项，捕捉周期模式)
    
    其中 ω_i (频率) 和 φ_i (相位) 都是可学习参数。
    
    优势:
    - 可学习的频率能自适应数据中的周期模式
    - 线性项能捕捉时间趋势
    - 初始化时可模拟标准正弦编码，保证"只好不坏"
    """
    def __init__(self, d_model: int, seq_len: int = 720):
        """
        Args:
            d_model: 输出维度 (与 hidden_dim 一致)
            seq_len: 序列长度 (用于频率初始化)
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # 线性项参数 (1维): ω_0 · τ + φ_0
        self.linear_weight = nn.Parameter(torch.zeros(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))
        
        # 周期项参数 (d_model-1 维): sin(ω_i · τ + φ_i)
        self.periodic_weight = nn.Parameter(torch.zeros(d_model - 1))
        self.periodic_bias = nn.Parameter(torch.zeros(d_model - 1))
        
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化权重，使 Time2Vec 初始行为接近标准正弦位置编码
        """
        with torch.no_grad():
            # 线性项初始化为小值 (趋势项)
            self.linear_weight.fill_(0.1)
            self.linear_bias.fill_(0.0)
            
            # 周期项频率初始化为对数均匀分布 (模拟标准正弦编码)
            # 标准正弦编码: freq = 1 / (10000^(2i/d))
            d = self.d_model - 1
            freqs = 1.0 / (10000 ** (torch.arange(d).float() / d))
            self.periodic_weight.copy_(freqs * 2 * np.pi * self.seq_len)
            self.periodic_bias.fill_(0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成可学习的时间编码
        
        Args:
            x: 输入张量 (B, T, D)，仅用于获取 batch_size、seq_len 和 device
        
        Returns:
            时间编码 (B, T, d_model)，与输入相加使用
        """
        B, T, _ = x.shape
        device = x.device
        
        # 生成归一化时间步 τ ∈ [0, 1]
        tau = torch.linspace(0, 1, T, device=device).view(1, T, 1)  # (1, T, 1)
        
        # 线性项: ω_0 · τ + φ_0
        linear_part = self.linear_weight * tau + self.linear_bias  # (1, T, 1)
        
        # 周期项: sin(ω_i · τ + φ_i)
        # 注意: 乘以 seq_len 使频率与序列长度相关
        periodic_part = torch.sin(
            self.periodic_weight.view(1, 1, -1) * tau * self.seq_len +
            self.periodic_bias.view(1, 1, -1)
        )  # (1, T, d_model-1)
        
        # 拼接线性项和周期项
        time_emb = torch.cat([linear_part, periodic_part], dim=-1)  # (1, T, d_model)
        
        return time_emb.expand(B, -1, -1)  # (B, T, d_model)
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, seq_len={self.seq_len}'

# =========================================================
# 3.5 DecompTransformer 组件：时序分解 + 频域增强
# =========================================================

# [已删除] GranulationTransformer 类 - 粒化功能已移除，请使用 EnhancedTransformer

# =========================================================
# 3.6 EnhancedTransformer: 基线Transformer + MC Dropout多次采样平均
#     架构和基线完全一样，MC采样用于提升预测稳定性
# =========================================================

class EnhancedTransformer(nn.Module):
    """
    EnhancedTransformer: RevIN + Time2Vec + Transformer Encoder + Physics Path + Quantile Head
    
    可选模块:
    - RevIN: 可逆实例归一化，消除分布漂移
    - Time2Vec: 可学习的时间编码，替代固定正弦位置编码
    """
    def __init__(self, cfg: Any, input_dim: int):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.enable_revin = bool(getattr(cfg, "enable_revin", False))
        self.enable_time2vec = bool(getattr(cfg, "enable_time2vec", True))  # 默认启用
        self.dropout_rate = getattr(cfg, "dropout", 0.3)
        self.raw_seq_len = int(getattr(cfg, "raw_seq_len", 360))

        num_layers = int(getattr(cfg, "num_transformer_layers", 2))
        ffn_multiplier = float(getattr(cfg, "ffn_dim_multiplier", 4.0))
        
        hidden_dim = getattr(cfg, "hidden_size", 128)
        dropout = self.dropout_rate
        nhead = getattr(cfg, "attn_heads", 4)
        output_steps = int(getattr(cfg, "forecast_steps", 5))
        self.multiscale_kernels = list(getattr(cfg, "multiscale_kernels", []))
        
        # RevIN: 仅输入端归一化
        if self.enable_revin:
            self.revin = RevIN(num_features=input_dim, affine=True)
            logger.info(f"[EnhancedTransformer] RevIN 已启用 (input_dim={input_dim})")
        
        # 目标特征索引 (根据 enable_fitting_module 动态计算)
        # enable_fitting_module=True 时，前4列是拟合衍生特征 (BTP_pos, BTP_temp, BTP_slope, BTP_auc)
        # enable_fitting_module=False 时，直接从原始特征开始
        self.enable_fitting_module = bool(getattr(cfg, "enable_fitting_module", True))
        target_col = getattr(cfg, "target_column", "北侧_计算BTP位置")
        
        if self.enable_fitting_module:
            # 有拟合衍生特征时的特征布局:
            # [南侧BTP位置, 南侧BTP温度, 南侧BTP斜率, 南侧BTP积分, 北侧BTP位置, 北侧BTP温度, 北侧BTP斜率, 北侧BTP积分, 机速, 料层厚度, 风箱负压, BTP_vel, BTP_acc]
            if "南" in target_col:
                self.target_feat_idx = 0  # 南侧_计算BTP位置
            else:
                self.target_feat_idx = 4  # 北侧_计算BTP位置
            self.speed_feat_idx = 8       # 机速检测值
            self.pressure_feat_idx = 10   # 南侧风箱负压
        else:
            # 无拟合衍生特征时的特征布局 (baseline_core_cols):
            # [南侧风箱温度x9, 北侧风箱温度x9, 南侧BTP位置, 北侧BTP位置, 机速, 料层厚度, 风箱负压]
            if "南" in target_col:
                self.target_feat_idx = 18  # 南侧_计算BTP位置 (9+9=18)
            else:
                self.target_feat_idx = 19  # 北侧_计算BTP位置 (9+9+1=19)
            self.speed_feat_idx = 20       # 机速检测值 (9+9+2=20)
            self.pressure_feat_idx = 22    # 南侧风箱负压 (9+9+2+2=22)
        
        logger.info(f"[EnhancedTransformer] enable_fitting_module={self.enable_fitting_module}")
        logger.info(f"[EnhancedTransformer] Feature Indices: target={self.target_feat_idx}, speed={self.speed_feat_idx}, pressure={self.pressure_feat_idx}")

        # ========== 输入投影 (通道融合) ==========
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # ========== 时间编码 ==========
        if self.enable_time2vec:
            # Time2Vec: 可学习的时间编码 (ICLR 2020)
            self.time_encoding = Time2Vec(d_model=hidden_dim, seq_len=self.raw_seq_len)
            logger.info(f"[EnhancedTransformer] Time2Vec 已启用 (d_model={hidden_dim}, seq_len={self.raw_seq_len})")
        else:
            # 固定正弦位置编码 (原始 Transformer)
            self.time_encoding = PositionalEncoding(d_model=hidden_dim, max_len=self.raw_seq_len + 100)
            logger.info(f"[EnhancedTransformer] 使用固定正弦位置编码")

        # Transformer 编码器 (可配置层数、FFN维度、头数)
        ffn_dim = int(hidden_dim * ffn_multiplier)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ========== 输出头 ==========
        self.head = QuantileDeltaHead(
            in_dim=hidden_dim + 3,
            steps=output_steps,
            dropout=dropout
        )
        
        # 参数统计与配置日志
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"[EnhancedTransformer] Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"[EnhancedTransformer] 配置: layers={num_layers}, heads={nhead}, ffn_dim={ffn_dim}, dropout={dropout}")
        logger.info(f"[EnhancedTransformer] 模块: RevIN={self.enable_revin}, Time2Vec={self.enable_time2vec}")
    
    def forward(self, x: torch.Tensor, mc_samples: int = None, show_progress: bool = False):
        """
        前向传播
        Args:
            x: 输入 (B, T, D)
            mc_samples: 兼容旧接口（当前忽略）
            show_progress: 兼容旧接口（当前忽略）
        Returns:
            (B, output_steps, num_quantiles)
        """
        return self._single_forward(x)
    
    def _single_forward(self, x: torch.Tensor):
        """单次前向传播"""
        B, T, D = x.shape
        idx = self.target_feat_idx
        
        # 1. Physics Path — 在 RevIN 之前提取，保留绝对水平信息
        speed_idx = self.speed_feat_idx
        pressure_idx = self.pressure_feat_idx
        phys = torch.cat([
            x[:, -1, idx:idx+1],                     # 目标特征 (BTP位置)
            x[:, -1, speed_idx:speed_idx+1],        # 机速检测值
            x[:, -1, pressure_idx:pressure_idx+1]   # 风箱负压
        ], dim=-1)  # (B, 3)
        
        if phys.shape[-1] != 3:
            phys = torch.zeros((B, 3), device=x.device)
        
        # 2. RevIN Normalize — 消除实例级分布漂移
        target_mean, target_std = None, None
        if self.enable_revin:
            x, mean, std = self.revin(x)
            target_mean = mean[:, :, idx]  # (B, 1)
            target_std = std[:, :, idx]    # (B, 1)
        
        # 3. 输入投影 (通道融合)
        h_proj = self.input_proj(x)  # (B, T, H)
        
        # 4. 时间编码 (Time2Vec 或固定正弦)
        h_proj = self.time_encoding(h_proj) + h_proj  # 残差连接

        # 5. Transformer 编码
        h_enc = self.transformer_encoder(h_proj)
        if not torch.isfinite(h_enc).all():
            logger.warning("[EnhancedTransformer] Encoder输出含NaN/Inf: finite=%s", torch.isfinite(h_enc).all().item())
        h = h_enc[:, -1, :]

        # 6. 拼接物理特征 + Quantile输出
        preds = self.head(h, phys)   # (B, output_steps, num_quantiles)
        if not torch.isfinite(preds).all():
            logger.warning("[EnhancedTransformer] 预测输出含NaN/Inf: finite=%s", torch.isfinite(preds).all().item())

        # 7. RevIN 反归一化
        if self.enable_revin and target_mean is not None:
            preds = preds * target_std.unsqueeze(-1) + target_mean.unsqueeze(-1)

        preds = torch.nan_to_num(preds, nan=0.0)
        
        return preds


# =========================================================
# 4. 对比基线模型 (Baselines)
# =========================================================

class BaselineQuantileModel(nn.Module):
    """
    基线对比模型 (Baseline Models for Ablation Study)
    
    与 EnhancedTransformer 的关键区别：
    1. 禁用 RevIN 模块 (不做实例归一化)
    2. 禁用拟合模块衍生特征 (由 main.py 传入不含衍生特征的数据)
    3. 禁用 CQR 校准 (由 main.py 控制)
    """
    def __init__(self, cfg: Any, model_type: str, input_dim: int):
        super().__init__()
        self.cfg = cfg
        self.model_name = model_type.lower()
        
        # [基线模型] 强制禁用 RevIN，不受配置影响
        self.enable_revin = False
        logger.info(f"[BaselineQuantileModel/{model_type}] RevIN 已禁用 (基线模型不使用实例归一化)")
        
        hidden_dim = getattr(cfg, "hidden_size", 128)
        dropout = getattr(cfg, "dropout", 0.5)

        # [基线模型] 目标特征索引需要根据输入特征重新计算
        # 基线模型输入不含衍生特征，所以索引不同于 EnhancedTransformer
        # 这里使用一个简化的物理路径，只用最后时刻的特征均值
        self.target_feat_idx = 0  # 基线模型使用第一个特征作为物理路径参考
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        if 'lstm' in self.model_name:
            self.core = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                                num_layers=2, batch_first=True, dropout=dropout if dropout > 0 else 0)
        elif 'gru' in self.model_name:
            self.core = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                                num_layers=2, batch_first=True, dropout=dropout if dropout > 0 else 0)
        elif 'transformer' in self.model_name:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
                dropout=dropout, batch_first=True
            )
            self.core = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.head = QuantileDeltaHead(
            in_dim=hidden_dim + 3,
            steps=int(getattr(cfg, "forecast_steps", 5)),
            dropout=dropout
        )

    def forward(self, x_raw: torch.Tensor):
        """
        基线模型前向传播
        
        注意：基线模型输入是原始风箱温度数据，不含衍生特征
        物理路径使用输入特征的统计量作为辅助信息
        """
        B, T, D = x_raw.shape
        
        # 1. Physics Path - 基线模型使用简化的物理路径
        # 由于输入不含衍生特征，使用最后时刻的前3个特征作为物理信息
        if D >= 3:
            phys = x_raw[:, -1, :3]  # 使用前3个特征
        else:
            phys = torch.zeros((B, 3), device=x_raw.device)
        
        # 2. 直接投影 (无 RevIN)
        x = self.input_projection(x_raw)
        
        # 3. 序列编码
        if 'lstm' in self.model_name or 'gru' in self.model_name:
            out, _ = self.core(x)
            z = out[:, -1, :]
        else:
            out = self.core(x)
            z = out[:, -1, :]

        # 4. Quantile Output (直接输出，无 RevIN 反归一化)
        preds = self.head(z, phys)
             
        return preds

# =========================================================
# 6. 模型构建工厂
# =========================================================

def build_model(config, input_dim, model_type=None, **kwargs):
    """
    模型工厂：统一接口
    
    注意:
    - GranulationTransformer 已移除，旧的 granulation_transformer 类型会自动映射到 enhanced_transformer
    - 基线模型 (baseline_*) 强制禁用 RevIN 和 CQR，用于消融实验对比
    """
    m_type = model_type if model_type else getattr(config, "model_type", "enhanced_transformer")
    revin_flag = "ON" if getattr(config, "enable_revin", False) else "OFF"
    
    # 兼容性映射：旧的粒化模型类型映射到 enhanced_transformer
    if m_type in ["granulation_refusion", "cnn_lstm", "granulation_transformer"]:
        logging.warning(f"[Model] {m_type} 已废弃，自动映射到 enhanced_transformer")
        m_type = "enhanced_transformer"
    
    # 判断是否为基线模型
    is_baseline = m_type.startswith("baseline_")
    
    if m_type == "enhanced_transformer":
        logging.info(f">>> [Model] Enhanced: Transformer + MC Dropout (Dim={input_dim}, RevIN={revin_flag}, FittingModule=ON)")
        return EnhancedTransformer(cfg=config, input_dim=input_dim)
    else:
        # 基线模型：强制禁用 RevIN、CQR、拟合模块
        logging.info(f">>> [Model] Baseline: {m_type.upper()} (Dim={input_dim}, RevIN=OFF, FittingModule=OFF, CQR=OFF)")
        return BaselineQuantileModel(cfg=config, model_type=m_type, input_dim=input_dim)
