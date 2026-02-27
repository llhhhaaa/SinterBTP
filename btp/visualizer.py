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

# visualizer.py
# =========================================================
# 论文级可视化模块 V2 - MIMO 多步预测完整版
# 
# 核心升级：
# 1. 严格的时序对齐验证
# 2. 多步预测渐进式展开
# 3. 背景感知的真值分布展示
# 4. 形态学指标 (Skewness/Kurtosis/Entropy) 可视化
# 5. 保留所有旧版功能（校准分析、健康度、CV汇总等）
# =========================================================

import os
import logging
from typing import Optional, Dict, List, Tuple, Union, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import pandas as pd
from scipy.stats import norm
# =========================================================
# 字体与风格设置
# =========================================================

def setup_cn_fonts_and_style():
    """修复中文显示 + 统一论文风格"""
    preferred_fonts = [
        "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred_fonts:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            break

    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 12
    matplotlib.rcParams["axes.titlesize"] = 14
    matplotlib.rcParams["axes.labelsize"] = 12
    matplotlib.rcParams["legend.fontsize"] = 10
    matplotlib.rcParams["lines.linewidth"] = 1.6
    matplotlib.rcParams["grid.alpha"] = 0.25
    matplotlib.rcParams["axes.grid"] = True

setup_cn_fonts_and_style()

# =========================================================
# 工具函数
# =========================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _savefig(fig: plt.Figure, save_path: str, dpi: int = 220):
    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[Plot] 保存: {save_path}")

def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """滚动平均"""
    if window <= 1: return x.copy()
    x = np.asarray(x, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(x_pad, kernel, mode="valid")
    return y[:len(x)]

def _to_1d(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """强制展平为 1D"""
    if x is None: return None
    return np.asarray(x).ravel()

def _compute_pit_bins(y_true_scalar: np.ndarray, q: np.ndarray) -> np.ndarray:
    """计算 PIT Bin (0..5)"""
    bins = np.zeros(len(y_true_scalar), dtype=int)
    for i in range(5):
        bins += (y_true_scalar > q[:, i]).astype(int)
    return bins

def _weights_entropy(weights: np.ndarray, eps: float = 1e-12, normalize: bool = True) -> np.ndarray:
    """计算权重熵 (用于衡量 Attention/Retrieval 的确定性)"""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[1] <= 0:
        return np.zeros(w.shape[0])
    w = np.clip(w, eps, 1.0)
    ent = -np.sum(w * np.log(w), axis=1)
    if normalize:
        k = w.shape[1]
        ent = ent / max(np.log(max(2, k)), eps)
    return ent

def _select_case_windows(y_cal: np.ndarray, y_true: np.ndarray, window_len: int = 240, top_k: int = 3) -> List[Tuple[str, int, int]]:
    """智能选择典型案例窗口"""
    n = len(y_cal)
    if n <= window_len + 5: return [("full", 0, n)]
    
    true_target = y_true[:, 2]
    cal_w_out = (y_cal[:, 4] - y_cal[:, 0])
    cal_mid_err = np.abs(y_cal[:, 2] - true_target)

    def get_best(score, reason):
        roll = _rolling_mean(score, window_len)
        end = int(np.argmax(roll))
        s = max(0, end - window_len // 2)
        e = min(n, s + window_len)
        s = max(0, e - window_len)
        return (reason, s, e)

    picks = [
        get_best(cal_w_out, "High_Uncertainty"),
        get_best(cal_mid_err, "High_Error")
    ]
    
    dedup = []
    for r, s, e in picks:
        if not any(abs(s - s2) < window_len//2 for _, s2, _ in dedup):
            dedup.append((r, s, e))
    return dedup[:top_k]

# =========================================================
# 核心对齐验证函数
# =========================================================

def validate_alignment(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_obs_win: Optional[np.ndarray] = None,
    context: str = ""
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    🔍 严格验证并对齐预测值和真值的维度
    
    输入规范：
    - y_pred: (N, Steps, 5) 或 (N, 5) - 预测分位数
    - y_true: (N, Steps) 或 (N, 5) 或 (N,) - 真实值
    - y_obs_win: (N, Win) - 观测窗（可选）
    
    输出规范：
    - y_pred_aligned: (N, 5) - 提取最后一步
    - y_true_aligned: (N, 5) - 构建真值分布
    - y_obs_win: (N, Win) - 原样返回
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    
    # Step 1: 处理预测值
    if y_pred.ndim == 3:
        # (N, Steps, 5) -> 提取最后一步
        N, Steps, Q = y_pred.shape
        if Q != 5:
            raise ValueError(f"[{context}] y_pred 最后一维必须是 5 个分位数，当前为 {Q}")
        y_pred_aligned = y_pred[:, -1, :]  # Shape: (N, 5)
        logging.info(f"[{context}] 预测值维度: {y_pred.shape} -> 提取 T+{Steps} (最后一步) -> {y_pred_aligned.shape}")
    elif y_pred.ndim == 2:
        if y_pred.shape[1] != 5:
            raise ValueError(f"[{context}] y_pred 必须是 (N, 5)，当前为 {y_pred.shape}")
        y_pred_aligned = y_pred
        logging.info(f"[{context}] 预测值维度: {y_pred.shape} (单步模式)")
    else:
        raise ValueError(f"[{context}] y_pred 维度错误: {y_pred.shape}")
    
    # Step 2: 处理真值
    N = y_pred_aligned.shape[0]
    
    if y_true.ndim == 1:
        # (N,) -> 构建 (N, 5) 分布
        y_true_scalar = y_true
    elif y_true.ndim == 2:
        if y_true.shape[1] == 5:
            # 已经是 (N, 5) 分布格式
            y_true_aligned = y_true
            logging.info(f"[{context}] 真值维度: {y_true.shape} (已对齐)")
            return y_pred_aligned, y_true_aligned, y_obs_win
        else:
            # (N, Steps) -> 提取最后一步
            y_true_scalar = y_true[:, -1]
            logging.info(f"[{context}] 真值维度: {y_true.shape} -> 提取最后一步 -> ({len(y_true_scalar)},)")
    else:
        raise ValueError(f"[{context}] y_true 维度错误: {y_true.shape}")
    
    # Step 3: 构建真值分布 (利用观测窗)
    if y_obs_win is not None and y_obs_win.ndim == 2:
        # 计算历史波动范围
        q10 = np.nanpercentile(y_obs_win, 10, axis=1)
        q90 = np.nanpercentile(y_obs_win, 90, axis=1)
        
        q25 = np.nanpercentile(y_obs_win, 25, axis=1)
        q75 = np.nanpercentile(y_obs_win, 75, axis=1)

        y_true_aligned = np.zeros((N, 5))
        y_true_aligned[:, 0] = q10
        y_true_aligned[:, 1] = q25
        y_true_aligned[:, 2] = y_true_scalar
        y_true_aligned[:, 3] = q75
        y_true_aligned[:, 4] = q90
        
        logging.info(f"[{context}] ✅ 使用观测窗构建真值分布: {y_true_aligned.shape}")
    else:
        # 简单平铺
        y_true_aligned = np.tile(y_true_scalar.reshape(-1, 1), (1, 5))
        logging.info(f"[{context}] ⚠️ 无观测窗，使用简单平铺: {y_true_aligned.shape}")
    
    return y_pred_aligned, y_true_aligned, y_obs_win

# =========================================================
# Visualizer 类
# =========================================================

class Visualizer:
    def __init__(self, save_dir: str, config=None):
        self.save_dir = save_dir
        self.cfg = config
        _ensure_dir(save_dir)
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        self.q_labels = ["Q10", "Q25", "Q50", "Q75", "Q90"]

    def _savefig(self, fig: plt.Figure, fname: str):
        save_path = os.path.join(self.save_dir, fname)
        _savefig(fig, save_path)

# 在 Visualizer 类中添加

    def plot_residual_diagnostic(self, res_data: Dict, fname: str):
        """绘制残差全景诊断图 (使用全量数据，ACF使用5分钟重采样)"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2)
        
        residuals = res_data['series']
        acf_vals = res_data['acf']
        # 从 res_data 获取采样间隔（秒），默认5秒
        sampling_sec = res_data.get('sampling_sec', 5.0)
        
        # 1. 残差时序图
        ax1 = fig.add_subplot(gs[0, :])
        # 将 x 轴转换为分钟
        time_minutes = np.arange(len(residuals)) * sampling_sec / 60.0
        ax1.plot(time_minutes, residuals, color='#2c3e50', lw=0.8, alpha=0.7)
        ax1.axhline(0, color='red', ls='--')
        ax1.set_title(f"残差时序分布 (共 {len(residuals)} 个样本, 采样间隔 {sampling_sec}秒)", fontweight='bold')
        ax1.set_xlabel("时间 (分钟)")
        ax1.set_ylabel("Error")
        
        # 2. 残差直方图
        ax2 = fig.add_subplot(gs[1, 0])
        sns.histplot(residuals, kde=True, ax=ax2, color='#3498db', stat="density")
        mu, std = np.mean(residuals), np.std(residuals)
        x = np.linspace(mu - 4*std, mu + 4*std, 100)
        ax2.plot(x, norm.pdf(x, mu, std), 'r-', lw=2, label='Normal Fit')
        ax2.set_title(f"残差概率密度分布 (μ={mu:.3f}, σ={std:.3f})", fontweight='bold')
        ax2.legend()
        
        # 3. ACF 图 (每个lag代表5分钟)
        ax3 = fig.add_subplot(gs[1, 1])
        # ACF已经是5分钟重采样后的数据，每个lag代表5分钟
        lags_minutes = np.arange(len(acf_vals)) * 5  # 0, 5, 10, 15, ... 分钟
        
        # 使用stem图（球线图）
        ax3.stem(lags_minutes, acf_vals)
        ax3.axhline(0.2, color='gray', ls=':', alpha=0.5)
        ax3.axhline(-0.2, color='gray', ls=':', alpha=0.5)
        ax3.set_title("残差自相关函数 (ACF - 5分钟间隔)", fontweight='bold', color='red')
        ax3.set_xlabel("滞后时间 (分钟)")
        ax3.set_ylabel("ACF")
        
        plt.tight_layout()
        self._savefig(fig, fname)


    def plot_robustness_stress_test(self, df_robust: pd.DataFrame, fname: str):
        """绘制鲁棒性压力测试结果"""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 绘制 MAE 增长
        sns.barplot(data=df_robust, x="噪声水平", y="MAE", ax=ax1, palette="Reds_r", alpha=0.7)
        ax1.set_title("模型抗干扰鲁棒性压力测试 (MAE vs Noise Level)", fontweight='bold', fontsize=14)
        
        # 绘制性能保持率折线
        ax2 = ax1.twinx()
        ax2.plot(range(len(df_robust)), df_robust["性能保持率"], color='blue', marker='o', lw=2, label="保持率")
        ax2.set_ylabel("性能保持率 (0-1)")
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc='upper right')
        
        ax1.grid(True, axis='y', ls='--', alpha=0.3)
        plt.tight_layout()
        self._savefig(fig, fname)




    # =========================================================
    # 🆕 多步预测渐进式展开图
    # =========================================================
    
    def plot_multistep_progression(
        self,
        y_pred_full: np.ndarray,
        y_true_full: np.ndarray,
        fname: str
    ):
        """
        展示 T+1 到 T+5 的预测误差演化
        
        Args:
            y_pred_full: (N, Steps, 5)
            y_true_full: (N, Steps)
        """
        if y_pred_full.ndim != 3:
            logging.warning("⚠️ 需要完整的 (N, Steps, 5) 数据才能绘制多步展开图")
            return
        
        N, Steps, _ = y_pred_full.shape
        
        # 计算每一步的 MAE
        step_maes = []
        for s in range(Steps):
            pred_mid = y_pred_full[:, s, 2]  # Q50
            true_val = y_true_full[:, s]
            mae = np.mean(np.abs(pred_mid - true_val))
            step_maes.append(mae)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        grain = getattr(self.cfg, "window_minutes", 1.0)
        steps_labels = [f"T+{(s+1)*grain}min" for s in range(Steps)]
        
        # 绘制折线
        ax.plot(steps_labels, step_maes, marker='o', color='#e74c3c', lw=2.5, ms=10, label='MAE 演化')
        
        # 填充渐变
        ax.fill_between(range(Steps), step_maes, alpha=0.2, color='#e74c3c')
        
        # 标注数值
        for i, mae in enumerate(step_maes):
            ax.text(i, mae + max(step_maes)*0.03, f"{mae:.4f}", ha='center', fontweight='bold', fontsize=10)
        
        ax.set_title("多步预测误差渐进式演化 (T+1 → T+5)", fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel("平均绝对误差 (MAE)")
        ax.set_xlabel("预测步长")
        ax.legend(fontsize=12)
        ax.grid(True, ls='--', alpha=0.4)
        
        self._savefig(fig, fname)

    # =========================================================
    # 🆕 背景感知的 BTP 全景图
    # =========================================================
    def plot_btp_panorama_enhanced(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        y_obs_win: Optional[np.ndarray],
        title: str,
        fname: str
    ):
        """
        增强版 BTP 全景图：展示真值的历史波动背景
        
        Args:
            y_pred: (N, 5) - 预测分位数
            y_true: (N, 5) - 真值分布（已构建）
            y_obs_win: (N, Win) - 观测窗
        """
        fig, ax = plt.subplots(figsize=(18, 7))
        x = np.arange(len(y_pred))
        
        # 1. 真实背景波动范围 (极浅灰色)
        ax.fill_between(x, y_true[:, 0], y_true[:, 4], 
                        color='#ecf0f1', alpha=0.8, label="真实历史波动范围 (GT Q10-Q90)", zorder=1)
        
        # 2. 真实中心线
        ax.plot(x, y_true[:, 2], color='#34495e', lw=1.5, alpha=0.7, label="真实值 (Ground Truth)", zorder=2)
        
        # 3. 预测区间 (80% 覆盖)
        ax.fill_between(x, y_pred[:, 0], y_pred[:, 4], 
                        color='#3498db', alpha=0.15, label="预测区间 80% (Q10-Q90)", zorder=3)
        
        # 4. 预测区间 (50% 覆盖)
        ax.fill_between(x, y_pred[:, 1], y_pred[:, 3], 
                        color='#e67e22', alpha=0.30, label="预测区间 50% (Q25-Q75)", zorder=4)
        
        # 5. 预测中心线
        ax.plot(x, y_pred[:, 2], color='#e74c3c', lw=2.5, ls='--', label="预测中心 (Q50)", zorder=5)
        
        # 6. 控制限
        if self.cfg:
            ax.axhline(self.cfg.btp_L_up, color='#c0392b', ls=':', lw=2, alpha=0.6, label="正常上界")
            ax.axhline(self.cfg.btp_L_low, color='#2980b9', ls=':', lw=2, alpha=0.6, label="正常下界")
        
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.set_ylabel("BTP 位置")
        ax.set_xlabel("样本索引")
        ax.legend(loc="upper right", frameon=True, facecolor='white', framealpha=0.9, ncol=3, fontsize=9)
        ax.grid(True, alpha=0.25)
        
        self._savefig(fig, fname)

    # =========================================================
    # 🆕 形态学指标诊断图
    # =========================================================
    
    def plot_morphology_indicators(self, health_res: Dict, fname: str):
        """
        可视化 MDPHI 三维分量：H_pos, H_stab, H_trend
        """
        H_pos = _to_1d(health_res.get("H_pos_series", []))
        H_stab = _to_1d(health_res.get("H_stab_series", []))
        H_trend = _to_1d(health_res.get("H_trend_series", []))
        
        if H_pos is None or len(H_pos) == 0:
            logging.warning("⚠️ 无 MDPHI 分量数据，跳过绘图")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        x = np.arange(len(H_pos))
        
        # 1. 静态偏离度 H_pos
        axes[0].plot(x, H_pos, color='#3498db', alpha=0.6, label="原始 H_pos")
        axes[0].plot(x, _rolling_mean(H_pos, 50), color='#2980b9', lw=2, label="平滑趋势 (MA50)")
        axes[0].axhline(1.0, color='green', ls='--', alpha=0.4, label="理想值")
        axes[0].set_title("静态偏离度 H_pos (基于非对称高斯核)", fontweight='bold')
        axes[0].set_ylabel("H_pos (0-1)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.25)
        
        # 2. 动态稳定性 H_stab
        axes[1].plot(x, H_stab, color='#e67e22', alpha=0.6, label="原始 H_stab")
        axes[1].plot(x, _rolling_mean(H_stab, 50), color='#d35400', lw=2, label="平滑趋势 (MA50)")
        axes[1].axhline(1.0, color='green', ls='--', alpha=0.4, label="理想值")
        axes[1].set_title("动态稳定性 H_stab (基于滚动波动率)", fontweight='bold')
        axes[1].set_ylabel("H_stab (0-1)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.25)
        
        # 3. 趋势风险度 H_trend
        axes[2].plot(x, H_trend, color='#9b59b6', alpha=0.6, label="原始 H_trend")
        axes[2].plot(x, _rolling_mean(H_trend, 50), color='#8e44ad', lw=2, label="平滑趋势 (MA50)")
        axes[2].axhline(1.0, color='green', ls='--', alpha=0.4, label="理想值")
        axes[2].set_title("趋势风险度 H_trend (相空间势能惩罚)", fontweight='bold')
        axes[2].set_ylabel("H_trend (0-1)")
        axes[2].set_xlabel("样本索引")
        axes[2].legend()
        axes[2].grid(True, alpha=0.25)
        
        plt.tight_layout()
        self._savefig(fig, fname)

    # =========================================================
    # 以下保留所有旧版功能 (略作调整以适配新对齐逻辑)
    # =========================================================
    def plot_predictions_5panel(self, y_pred: np.ndarray, y_true: np.ndarray, title: str, fname: str):
        """
        5分位数分面图
        修正说明：所有子图的'真实值'黑线都必须是标量真值 (Index 2)，
        不能随 i 变化而绘制成历史分位数。
        """
        fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
        x = np.arange(len(y_pred))
        
        # 确保 y_true 有 5 列，取中间列 (Index 2) 作为唯一的标量真值
        # validate_alignment 保证了 y_true 是 (N, 5)，且 Index 2 永远是 Scalar Truth
        true_scalar = y_true[:, 2] 

        for i, ax in enumerate(axes):
            # [修正] 这里 y_true 改为固定的 true_scalar，不再随 i 变化
            ax.plot(x, true_scalar, color='black', alpha=0.6, label="真实值 (Scalar Truth)")
            
            # 预测值依然随 i 变化 (Q10, Q25, Q50, Q75, Q90)
            ax.plot(x, y_pred[:, i], color='#1f77b4', alpha=0.8, ls="--", label=f"预测 {self.q_labels[i]}")
            
            ax.set_title(f"{title} - {self.q_labels[i]}", fontweight="bold", fontsize=10)
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.25)

        axes[-1].set_xlabel("样本索引")
        self._savefig(fig, fname)    
   
    def plot_variant_q50_comparison(self, y_pred_dict: Dict[str, np.ndarray], y_true: np.ndarray, fname: str):
        """对比不同模型的 Q50 输出"""
        fig, ax = plt.subplots(figsize=(16, 6))
        x = np.arange(len(y_true))
        ax.plot(x, y_true[:, 2], color="black", alpha=0.5, label="真实值")

        styles = ["-", "--", "-.", ":"]
        for idx, (name, y_p) in enumerate(y_pred_dict.items()):
            ls = styles[idx % len(styles)]
            ax.plot(x, y_p[:, 2], ls=ls, alpha=0.9, label=f"{name} Q50")

        ax.set_title("不同模型配置对比 (Q50 中心线)", fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        self._savefig(fig, fname)

    def plot_calibration_comparison(self, y_raw: np.ndarray, y_calib: np.ndarray, y_true: np.ndarray, corrections: np.ndarray, fname: str):
        """校准对比图（三联图）"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 15), sharex=True, 
                                            gridspec_kw={'height_ratios': [1, 1, 0.6], 'hspace': 0.15})
        x = np.arange(len(y_true))
        y_true_mid = y_true[:, 2]

        def draw_panel(ax, y_pred, title_str, is_raw=True):
            ax.fill_between(x, y_true[:, 0], y_true[:, 4], color='#ecf0f1', alpha=0.8, label="真实波动范围", zorder=1)
            ax.plot(x, y_true_mid, color='#7f8c8d', lw=1, alpha=0.6, label="真实中心线", zorder=2)
            ax.fill_between(x, y_pred[:, 0], y_pred[:, 4], color='#3498db', alpha=0.15, label="预测区间", zorder=3)
            line_style = '--' if is_raw else '-'
            ax.plot(x, y_pred[:, 0], color='#d35400' if not is_raw else '#e67e22', ls=line_style, lw=1.5, label="下界", zorder=4)
            ax.plot(x, y_pred[:, 4], color='#1e8449' if not is_raw else '#27ae60', ls=line_style, lw=1.5, label="上界", zorder=5)
            ax.plot(x, y_pred[:, 2], color='#2980b9', lw=2, label="中心", zorder=6)
            ax.set_title(title_str, fontsize=15, fontweight="bold", loc='left')
            ax.legend(loc='upper right', ncol=3, fontsize=9)
            ax.set_ylabel("BTP 位置")
            ax.grid(True, alpha=0.2)

        draw_panel(ax1, y_raw, "Panel A: 原始预测 (Raw)", is_raw=True)
        draw_panel(ax2, y_calib, "Panel B: CQR 校准后 (Calibrated)", is_raw=False)

        if corrections is not None:
            ax3.plot(x, corrections[:, 0], color='#3498db', lw=1.5, label="Q-Factor (Inner)", alpha=0.7)
            ax3.plot(x, corrections[:, 1], color='#e67e22', lw=1.5, label="Q-Factor (Outer)", alpha=0.7)
            ax3.plot(x, _rolling_mean(corrections[:, 0], 100), color='#2980b9', lw=2, label="Trend (Inner)")
            ax3.plot(x, _rolling_mean(corrections[:, 1], 100), color='#d35400', lw=2, label="Trend (Outer)")

        ax3.axhline(0, color='black', lw=1, zorder=0)
        ax3.set_title("Panel C: 动态校准因子趋势", fontsize=14, fontweight="bold", loc='left')
        ax3.set_ylabel("校准偏移量")
        ax3.set_xlabel("样本索引")
        ax3.legend(loc='upper right', ncol=2, fontsize=9)
        ax3.grid(True, alpha=0.2)

        self._savefig(fig, fname)

    def plot_reliability_curve(self, y_raw: np.ndarray, y_cal: np.ndarray, y_true: np.ndarray, fname: str):
        """可靠性曲线"""
        taus = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
        y_t = y_true[:, 2]

        def get_prob(y_p):
            return np.array([np.mean(y_t <= y_p[:, j]) for j in range(5)])

        e_raw = get_prob(y_raw)
        e_cal = get_prob(y_cal)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(taus, e_raw, marker="o", ls="--", label="Raw")
        ax.plot(taus, e_cal, marker="s", ls="-", label="Calibrated")
        ax.plot([0, 1], [0, 1], "k:", label="Ideal")
        
        err_raw = np.mean(np.abs(e_raw - taus))
        err_cal = np.mean(np.abs(e_cal - taus))
        ax.text(0.05, 0.85, f"MAE Raw: {err_raw:.3f}\nMAE Cal: {err_cal:.3f}", transform=ax.transAxes)
        
        ax.set_title("Reliability Curve", fontweight="bold")
        ax.set_xlabel("Nominal Quantile")
        ax.set_ylabel("Empirical Probability")
        ax.legend()
        ax.grid(True)
        self._savefig(fig, fname)

    def plot_health_fusion_panorama(self, y_pred: np.ndarray, y_true: np.ndarray, health_res: Dict, title: str, fname: str):
        """
        联动图: BTP vs Health (简化版 - 仅支持 MDPHI 核心指标)
        """
        H_pred = _to_1d(health_res.get("health_scores", []))
        if H_pred is None or len(H_pred) == 0: 
            logging.warning("⚠️ 无健康度数据，跳过融合全景图")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})
        x = np.arange(len(y_pred))

        # Panel 1: BTP 预测全景
        ax1.plot(x, y_true[:, 2], color="black", alpha=0.5, label="Truth Q50")
        ax1.fill_between(x, y_pred[:, 0], y_pred[:, 4], color='orange', alpha=0.15, label="Pred [Q10-Q90]")
        ax1.plot(x, y_pred[:, 2], color='red', label="Pred Q50")
        if self.cfg:
            ax1.axhline(self.cfg.btp_L_low, ls=":", color='green')
            ax1.axhline(self.cfg.btp_L_up, ls=":", color='green')
        ax1.set_title(f"{title} - BTP 预测全景", fontweight="bold", fontsize=14)
        ax1.set_ylabel("BTP 位置")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.25)

        # Panel 2: 健康度
        H_pred = _to_1d(health_res.get("health_scores", []))
        
        # 如果有真值健康度
        if "true_health_scores" in health_res:
            H_true = _to_1d(health_res["true_health_scores"])
            if H_true is not None and len(H_true) == len(x):
                ax2.plot(x, H_true, color='#2c3e50', lw=1.5, alpha=0.6, ls='--', label="真实健康度")
        
        ax2.axhline(90, color='#e74c3c', ls=':', lw=2, alpha=0.6, label="健康阈值 (90分)")
        ax2.fill_between(x, 90, 100, color='#27ae60', alpha=0.1)
        ax2.fill_between(x, 0, 90, color='#e74c3c', alpha=0.1)
        
        ax2.set_title("健康度演化趋势 (0-100分)", fontweight="bold", fontsize=13)
        ax2.set_ylabel("健康度分数")
        ax2.set_xlabel("样本索引")
        ax2.set_ylim(-5, 105)
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.25)

        plt.tight_layout()
        self._savefig(fig, fname)
    

    def plot_pit_histogram(self, y_cal: np.ndarray, y_true: np.ndarray, fname: str):
        """PIT 直方图 (概率积分变换)"""
        y_t = y_true[:, 2]
        bins = _compute_pit_bins(y_t, y_cal)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        counts, _, _ = ax.hist(bins, bins=np.arange(7)-0.5, color='#3498db', alpha=0.7, edgecolor='black', density=True)
        
        expected = 1.0 / 6.0
        ax.axhline(expected, color='red', ls='--', lw=2, label=f"理想均匀分布 ({expected:.3f})")
        
        # 卡方检验
        from scipy.stats import chisquare
        chi2, p = chisquare(counts)
        ax.text(0.05, 0.95, f"χ² = {chi2:.2f}\np-value = {p:.4f}", 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title("PIT 直方图 (Probability Integral Transform)", fontweight="bold", fontsize=14)
        ax.set_xlabel("PIT Bin")
        ax.set_ylabel("密度")
        ax.set_xticks(range(6))
        ax.set_xticklabels(["[0, Q10)", "[Q10, Q25)", "[Q25, Q50)", "[Q50, Q75)", "[Q75, Q90)", "[Q90, 1]"])
        ax.legend()
        ax.grid(True, alpha=0.25, axis='y')
        
        self._savefig(fig, fname)

    def plot_interval_width_analysis(self, y_raw: np.ndarray, y_cal: np.ndarray, fname: str):
        """区间宽度分析 (校准前后对比)"""
        w_raw_inner = y_raw[:, 3] - y_raw[:, 1]
        w_cal_inner = y_cal[:, 3] - y_cal[:, 1]
        w_raw_outer = y_raw[:, 4] - y_raw[:, 0]
        w_cal_outer = y_cal[:, 4] - y_cal[:, 0]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        x = np.arange(len(w_raw_inner))
        
        # Panel A: Inner Width 时序
        axes[0, 0].plot(x, w_raw_inner, color='#e67e22', alpha=0.4, label="Raw")
        axes[0, 0].plot(x, w_cal_inner, color='#2980b9', alpha=0.7, label="Calibrated")
        axes[0, 0].plot(x, _rolling_mean(w_raw_inner, 100), color='#d35400', lw=2, label="Raw Trend")
        axes[0, 0].plot(x, _rolling_mean(w_cal_inner, 100), color='#1e8449', lw=2, label="Cal Trend")
        axes[0, 0].set_title("Inner Width (Q25-Q75) 时序演化", fontweight="bold")
        axes[0, 0].set_ylabel("区间宽度")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.25)
        
        # Panel B: Outer Width 时序
        axes[0, 1].plot(x, w_raw_outer, color='#e67e22', alpha=0.4, label="Raw")
        axes[0, 1].plot(x, w_cal_outer, color='#2980b9', alpha=0.7, label="Calibrated")
        axes[0, 1].plot(x, _rolling_mean(w_raw_outer, 100), color='#d35400', lw=2, label="Raw Trend")
        axes[0, 1].plot(x, _rolling_mean(w_cal_outer, 100), color='#1e8449', lw=2, label="Cal Trend")
        axes[0, 1].set_title("Outer Width (Q10-Q90) 时序演化", fontweight="bold")
        axes[0, 1].set_ylabel("区间宽度")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.25)
        
        # Panel C: Inner Width 分布对比
        axes[1, 0].hist(w_raw_inner, bins=50, alpha=0.5, color='#e67e22', label="Raw", density=True)
        axes[1, 0].hist(w_cal_inner, bins=50, alpha=0.5, color='#2980b9', label="Calibrated", density=True)
        axes[1, 0].axvline(np.median(w_raw_inner), color='#d35400', ls='--', lw=2, label=f"Raw Median: {np.median(w_raw_inner):.3f}")
        axes[1, 0].axvline(np.median(w_cal_inner), color='#1e8449', ls='--', lw=2, label=f"Cal Median: {np.median(w_cal_inner):.3f}")
        axes[1, 0].set_title("Inner Width 分布", fontweight="bold")
        axes[1, 0].set_xlabel("区间宽度")
        axes[1, 0].set_ylabel("密度")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.25)
        
        # Panel D: Outer Width 分布对比
        axes[1, 1].hist(w_raw_outer, bins=50, alpha=0.5, color='#e67e22', label="Raw", density=True)
        axes[1, 1].hist(w_cal_outer, bins=50, alpha=0.5, color='#2980b9', label="Calibrated", density=True)
        axes[1, 1].axvline(np.median(w_raw_outer), color='#d35400', ls='--', lw=2, label=f"Raw Median: {np.median(w_raw_outer):.3f}")
        axes[1, 1].axvline(np.median(w_cal_outer), color='#1e8449', ls='--', lw=2, label=f"Cal Median: {np.median(w_cal_outer):.3f}")
        axes[1, 1].set_title("Outer Width 分布", fontweight="bold")
        axes[1, 1].set_xlabel("区间宽度")
        axes[1, 1].set_ylabel("密度")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.25)
        
        plt.tight_layout()
        self._savefig(fig, fname)

    def plot_case_windows_deep_dive(self, y_cal: np.ndarray, y_true: np.ndarray, health_res: Dict, fname_prefix: str):
        """典型案例窗口深度分析"""
        windows = _select_case_windows(y_cal, y_true, window_len=240, top_k=3)
        
        for reason, s, e in windows:
            fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
            x = np.arange(s, e)
            idx_slice = slice(s, e)
            
            # Panel 1: BTP 预测
            y_c = y_cal[idx_slice]
            y_t = y_true[idx_slice]
            
            axes[0].fill_between(x, y_t[:, 0], y_t[:, 4], color='#ecf0f1', alpha=0.8, label="真实波动")
            axes[0].plot(x, y_t[:, 2], color='#34495e', lw=1.5, alpha=0.7, label="真实中心")
            axes[0].fill_between(x, y_c[:, 0], y_c[:, 4], color='#3498db', alpha=0.15, label="预测区间")
            axes[0].plot(x, y_c[:, 2], color='#e74c3c', lw=2.5, ls='--', label="预测中心")
            
            axes[0].set_title(f"案例窗口: {reason} [样本 {s}-{e}] - BTP 轨迹", fontweight="bold", fontsize=13)
            axes[0].set_ylabel("BTP 位置")
            axes[0].legend(loc="upper right")
            axes[0].grid(True, alpha=0.25)
            
            # Panel 2: 健康度
            H_pred = _to_1d(health_res.get("H", []))
            if H_pred is not None and len(H_pred) >= e:
                axes[1].plot(x, H_pred[idx_slice], color='#27ae60', lw=2, label="预测健康度")
                axes[1].axhline(0.9, color='#e74c3c', ls=':', lw=2, alpha=0.6, label="健康阈值")
                axes[1].fill_between(x, 0.9, 1.0, color='#27ae60', alpha=0.1)
                axes[1].set_title("健康度动态", fontweight="bold", fontsize=12)
                axes[1].set_ylabel("健康度")
                axes[1].set_ylim(-0.05, 1.05)
                axes[1].legend()
                axes[1].grid(True, alpha=0.25)
            
            # Panel 3: 动态稳定性 (替代熵)
            H_stab = _to_1d(health_res.get("H_stab_series", []))
            if H_stab is not None and len(H_stab) >= e:
                axes[2].plot(x, H_stab[idx_slice], color='#9b59b6', lw=2, label="动态稳定性")
                axes[2].plot(x, _rolling_mean(H_stab[idx_slice], 20), color='#8e44ad', lw=2.5, label="MA20")
                axes[2].set_title("动态稳定性演化 (H_stab)", fontweight="bold", fontsize=12)
                axes[2].set_ylabel("H_stab (0-1)")
                axes[2].set_xlabel("样本索引")
                axes[2].legend()
                axes[2].grid(True, alpha=0.25)
            
            plt.tight_layout()
            self._savefig(fig, f"{fname_prefix}_{reason}.png")

    def plot_cross_validation_summary(self, cv_results: List[Dict], fname: str):
        """交叉验证汇总热力图"""
        if not cv_results or len(cv_results) == 0:
            logging.warning("⚠️ 无交叉验证数据")
            return
        
        metrics = ["mae", "mape", "rmse", "coverage_50", "coverage_80", "sharpness"]
        n_folds = len(cv_results)
        
        # 提取数据
        data = np.zeros((len(metrics), n_folds))
        for i, m in enumerate(metrics):
            for j, res in enumerate(cv_results):
                data[i, j] = res.get(m, 0)
        
        # 归一化 (按行)
        data_norm = (data - data.min(axis=1, keepdims=True)) / (data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True) + 1e-8)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(n_folds))
        ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics)
        
        # 标注数值
        for i in range(len(metrics)):
            for j in range(n_folds):
                text = ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", 
                              color="black" if data_norm[i, j] > 0.5 else "white", fontsize=9, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label="归一化分数 (0-1)")
        ax.set_title("交叉验证性能热力图 (颜色越绿越好)", fontweight="bold", fontsize=14)
        
        self._savefig(fig, fname)

    def plot_quantile_correlation_matrix(self, y_pred: np.ndarray, fname: str):
        """分位数相关性矩阵"""
        corr_matrix = np.corrcoef(y_pred.T)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        ax.set_xticks(range(5))
        ax.set_xticklabels(self.q_labels)
        ax.set_yticks(range(5))
        ax.set_yticklabels(self.q_labels)
        
        # 标注相关系数
        for i in range(5):
            for j in range(5):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center",
                              color="white" if abs(corr_matrix[i, j]) > 0.7 else "black", fontsize=11, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label="Pearson 相关系数")
        ax.set_title("预测分位数相关性矩阵", fontweight="bold", fontsize=14)
        
        self._savefig(fig, fname)

    def plot_error_decomposition(self, y_pred: np.ndarray, y_true: np.ndarray, fname: str):
        """误差分解分析 (Bias vs Variance)"""
        err = y_pred[:, 2] - y_true[:, 2]  # Q50 误差
        
        # 计算偏差和方差
        bias = np.mean(err)
        variance = np.var(err)
        mse = np.mean(err**2)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel A: 误差时序
        x = np.arange(len(err))
        axes[0, 0].plot(x, err, color='#e74c3c', alpha=0.4, label="逐点误差")
        axes[0, 0].plot(x, _rolling_mean(err, 100), color='#c0392b', lw=2.5, label="MA100")
        axes[0, 0].axhline(0, color='black', ls='--', alpha=0.5)
        axes[0, 0].axhline(bias, color='#2980b9', ls=':', lw=2, label=f"全局偏差: {bias:.4f}")
        axes[0, 0].fill_between(x, 0, err, where=(err>0), color='#e74c3c', alpha=0.1)
        axes[0, 0].fill_between(x, err, 0, where=(err<0), color='#3498db', alpha=0.1)
        axes[0, 0].set_title("误差时序演化", fontweight="bold")
        axes[0, 0].set_ylabel("预测误差 (Pred - True)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.25)
        
        # Panel B: 误差分布
        axes[0, 1].hist(err, bins=50, color='#3498db', alpha=0.7, edgecolor='black', density=True)
        axes[0, 1].axvline(0, color='black', ls='--', lw=2, label="零误差")
        axes[0, 1].axvline(bias, color='#e74c3c', ls=':', lw=2.5, label=f"偏差: {bias:.4f}")
        axes[0, 1].set_title("误差分布直方图", fontweight="bold")
        axes[0, 1].set_xlabel("预测误差")
        axes[0, 1].set_ylabel("密度")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.25, axis='y')
        
        # Panel C: Bias-Variance 分解
        components = ['Bias²', 'Variance', 'MSE']
        values = [bias**2, variance, mse]
        colors_bar = ['#e74c3c', '#3498db', '#95a5a6']
        
        bars = axes[1, 0].bar(components, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[1, 0].set_title("误差分解 (Bias² + Variance ≈ MSE)", fontweight="bold")
        axes[1, 0].set_ylabel("数值")
        
        # 标注数值
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                          f'{val:.5f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        axes[1, 0].grid(True, alpha=0.25, axis='y')
        
        # Panel D: QQ Plot
        from scipy.stats import probplot
        probplot(err, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("QQ图 (正态性检验)", fontweight="bold")
        axes[1, 1].grid(True, alpha=0.25)
        
        plt.tight_layout()
        self._savefig(fig, fname)
    def _plot_training_history(self, history: Dict, fname: str):
        """绘制训练过程曲线（Loss、学习率等）"""
        if not history:
            logging.warning("⚠️ 训练历史为空，跳过曲线绘制")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Panel 1: Loss 曲线
        if "train_loss" in history:
            axes[0].plot(history["train_loss"], label="Train Loss", color='#3498db', lw=2)
        if "val_loss" in history:
            axes[0].plot(history["val_loss"], label="Val Loss", color='#e74c3c', lw=2)
        
        axes[0].set_title("训练过程 Loss 演化", fontweight="bold", fontsize=14)
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.25)
        
        # Panel 2: 学习率曲线
        if "lr" in history:
            axes[1].plot(history["lr"], label="Learning Rate", color='#27ae60', lw=2)
            axes[1].set_title("学习率调度", fontweight="bold", fontsize=14)
            axes[1].set_ylabel("Learning Rate")
            axes[1].set_xlabel("Epoch")
            axes[1].legend()
            axes[1].grid(True, alpha=0.25)
        else:
            axes[1].axis('off')  # 如果没有 lr 数据，隐藏第二个子图
        
        plt.tight_layout()
        self._savefig(fig, fname)

    def plot_diagnosis_confusion_matrix(self, y_true_states: np.ndarray, y_pred_states: np.ndarray, fname: str):
        """
        绘制 5分类 混淆矩阵热力图 (含模糊准确率计算)
        States: 0:过烧, 1:疑似过烧, 2:正常, 3:疑似欠烧, 4:欠烧
        """
        labels = ["过烧", "疑似过烧", "正常", "疑似欠烧", "欠烧"]
        
        # 1. 计算基础指标
        acc_exact = accuracy_score(y_true_states, y_pred_states)
        
        # 2. 计算模糊准确率 (允许误差 ±1 级)
        diff = np.abs(y_true_states - y_pred_states)
        acc_fuzzy = np.mean(diff <= 1)
        
        # 3. 构建混淆矩阵
        cm = confusion_matrix(y_true_states, y_pred_states, labels=[0, 1, 2, 3, 4])
        
        # 归一化 (按真值行归一化，显示召回率)
        # 避免除以0
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        # 4. 绘图
        fig, ax = plt.subplots(figsize=(10, 9))
        
        # 使用 Seaborn 绘制热力图
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax, square=True,
                    annot_kws={"size": 11, "weight": "bold"})
        
        # 在格子中填入原始数量
        for i in range(5):
            for j in range(5):
                text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j + 0.5, i + 0.7, f"({cm[i, j]})", 
                        ha="center", va="center", color=text_color, fontsize=9)
        
        ax.set_title(f"工况状态诊断混淆矩阵\n精确准确率: {acc_exact:.2%} | 模糊准确率(±1级): {acc_fuzzy:.2%}", 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("预测状态 (Predicted)", fontsize=12, fontweight='bold')
        ax.set_ylabel("真实状态 (Ground Truth)", fontsize=12, fontweight='bold')
        
        # 标注方向箭头
        ax.text(-0.5, 0.5, "← 数值偏小 (过烧侧)", rotation=90, va='center', fontsize=10, color='red')
        ax.text(-0.5, 4.5, "数值偏大 (欠烧侧) →", rotation=90, va='center', fontsize=10, color='red')
        
        self._savefig(fig, fname)
        # [在 visualizer.py 的 Visualizer 类中添加以下两个方法]

    def plot_btp_health_panorama(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 health_res: Dict, fname: str, 
                                 mu: float = 22.6):
        """
        绘制 BTP全景 + 健康度全景 (上下对齐)
        Top: BTP 真实 vs 预测
        Bottom: Health 真实 vs 预测
        """
        # 数据准备
        if y_pred.ndim > 1 and y_pred.shape[1] == 5:
            # 如果是分位数预测，取中位数(Q50)作为主线
            btp_pred = y_pred[:, 2]
        else:
            btp_pred = y_pred.flatten()
            
        if y_true.ndim > 1 and y_true.shape[1] == 5:
            btp_true = y_true[:, 2]
        else:
            btp_true = y_true.flatten()
            
        h_pred = health_res['health_scores']
        h_true = health_res.get('true_health_scores', None)
        
        # 绘图设置
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True, 
                                       gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})
        
        # --- 子图1: BTP 数值 ---
        x = np.arange(len(btp_true))
        ax1.plot(x, btp_true, label='BTP True', color='black', alpha=0.6, linewidth=1)
        ax1.plot(x, btp_pred, label='BTP Pred', color='#1f77b4', linewidth=1.2)
        # 绘制目标线
        ax1.axhline(y=mu, color='green', linestyle='--', alpha=0.5, label='Target $\mu$')
        
        ax1.set_ylabel("BTP Value", fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.set_title("BTP Prediction & Health Diagnosis Panorama", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # --- 子图2: 健康度数值 ---
        if h_true is not None:
            ax2.plot(x, h_true, label='Health True', color='green', alpha=0.7, linewidth=1.5)
        ax2.plot(x, h_pred, label='Health Pred', color='#d62728', alpha=0.9, linewidth=1.5, linestyle='-')
        
        # 绘制健康/故障阈值带
        ax2.axhline(y=85, color='green', linestyle=':', alpha=0.5)
        ax2.axhline(y=65, color='orange', linestyle=':', alpha=0.5)
        ax2.fill_between(x, 85, 100, color='green', alpha=0.05, label='Healthy Zone')
        ax2.fill_between(x, 0, 65, color='red', alpha=0.05, label='Fault Zone')
        
        ax2.set_ylabel("Health Score (0-100)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Time Step", fontsize=12, fontweight='bold')
        ax2.set_ylim(-5, 105)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        self._savefig(fig, fname)

    def plot_health_correlation(self, health_res: Dict, fname: str):
        """
        绘制健康度相关性分析图 (True vs Pred)
        """
        h_true = health_res.get('true_health_scores')
        h_pred = health_res['health_scores']
        
        if h_true is None:
            return

        # 计算相关系数
        corr_matrix = np.corrcoef(h_true, h_pred)
        r_value = corr_matrix[0, 1]
        
        # 绘图
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 散点
        ax.scatter(h_true, h_pred, alpha=0.5, color='#1f77b4', s=15, label='Samples')
        
        # 对角线 (理想线)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal (y=x)')
        
        # 拟合线
        m, b = np.polyfit(h_true, h_pred, 1)
        ax.plot(h_true, m*h_true + b, color='red', alpha=0.8, linewidth=2, 
                label=f'Fit: y={m:.2f}x+{b:.2f}')
        
        ax.set_aspect('equal')
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_xlabel('True Health Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Health Score', fontsize=12, fontweight='bold')
        ax.set_title(f"Health Score Correlation Analysis\nPearson R = {r_value:.4f}", 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        self._savefig(fig, fname)


    # =========================================================
    # 🚀 一键生成所有图表 (调度器 - 已修复参数名对齐)
    # =========================================================
    
    def generate_all_plots(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        y_obs_win: Optional[np.ndarray] = None,
        health_res: Optional[Dict] = None,
        cv_metrics_list: Optional[List[Dict]] = None,
        y_raw: Optional[np.ndarray] = None,
        corrections: Optional[np.ndarray] = None,
        config=None,
        history: Optional[Dict] = None,
        enable_case_slices: bool = True,
        prefix: str = "val"
    ):
        """自动对齐数据并生成精简版可视化报告"""
        logging.info(f"开始生成可视化报告: {prefix}...")
        
        if config is not None:
            self.cfg = config

        # 1. 严格对齐数据
        y_pred_aligned, y_true_aligned, _ = validate_alignment(
            y_pred, y_true, y_obs_win, context=prefix
        )

        # 2. 基础趋势与全景图
        self.plot_btp_panorama_enhanced(
            y_pred_aligned, y_true_aligned, y_obs_win, 
            title=f"{prefix.upper()} - BTP Prediction Panorama", 
            fname=f"{prefix}_btp_panorama.png"
        )
        
        self.plot_predictions_5panel(
            y_pred_aligned, y_true_aligned, 
            title=f"{prefix.upper()} Quantiles", 
            fname=f"{prefix}_5panel.png"
        )

        # 3. 校准分析 (只有在传入 y_raw 时才执行)
        if y_raw is not None:
            y_raw_aligned, _, _ = validate_alignment(y_raw, y_true, context=f"{prefix}_raw")
            self.plot_calibration_comparison(
                y_raw_aligned, y_pred_aligned, y_true_aligned, corrections,
                f"{prefix}_calibration_comparison.png"
            )
            self.plot_reliability_curve(y_raw_aligned, y_pred_aligned, y_true_aligned, f"{prefix}_reliability.png")
            self.plot_pit_histogram(y_pred_aligned, y_true_aligned, f"{prefix}_pit_hist.png")
            self.plot_interval_width_analysis(y_raw_aligned, y_pred_aligned, f"{prefix}_width_analysis.png")

        # 4. 误差分解与统计
        self.plot_error_decomposition(y_pred_aligned, y_true_aligned, f"{prefix}_error_analysis.png")

        # 5. 训练曲线
        if history:
            self._plot_training_history(history, f"{prefix}_training_curves.png")

        logging.info(f"[Plot] {prefix} 报告生成成功！")

    # =========================================================
    # 🆕 跨模型对比分析 (论文对比实验专用)
    # =========================================================

# --- visualizer.py ---

# --- visualizer.py ---

    def plot_model_comparison_boxplots(self, experiment_root: str, fname: str = "model_error_comparison.png"):
        """
        [全折聚合修复版] 跨模型误差对比分析
        """
        import pandas as pd
        import seaborn as sns
        import glob
        
        all_model_dfs = []
        # 搜索所有以 Compare_ 开头的文件夹
        model_dirs = [d for d in os.listdir(experiment_root) if os.path.isdir(os.path.join(experiment_root, d)) and d.startswith("Compare_")]
        
        logging.info(f"🔍 正在扫描对比模型，找到: {model_dirs}")

        for m_dir in model_dirs:
            m_label = m_dir.replace("Compare_", "").upper()
            model_base_path = os.path.join(experiment_root, m_dir)
            
            # 使用 glob 递归搜索该模型目录下所有的 residual_analysis.csv
            # 这会同时抓取 final_test_results 和所有 Fold_X 下的数据
            search_pattern = os.path.join(model_base_path, "**", "residual_analysis.csv")
            csv_files = glob.glob(search_pattern, recursive=True)
            
            if not csv_files:
                logging.warning(f"⚠️ 模型 {m_label} 未找到任何残差数据文件！")
                continue
            
            logging.info(f"汇总模型 {m_label}: 发现 {len(csv_files)} 个残差文件 (含多折数据)")
            
            model_residuals = []
            for f in csv_files:
                try:
                    df_tmp = pd.read_csv(f)
                    if 'Residual' in df_tmp.columns:
                        model_residuals.append(np.abs(df_tmp['Residual'].values))
                except Exception as e:
                    logging.error(f"读取文件 {f} 失败: {e}")

            if model_residuals:
                # 合并该模型的所有误差数据
                all_abs_errors = np.concatenate(model_residuals)
                temp_df = pd.DataFrame({
                    'Absolute Error': all_abs_errors,
                    'Model': m_label
                })
                all_model_dfs.append(temp_df)

        if not all_model_dfs:
            logging.error("❌ 严重警告: 未能抓取到任何残差数据，无法生成 Boxplot！")
            return

        # 合并所有模型数据
        combined_df = pd.concat(all_model_dfs, ignore_index=True)
        
        # 绘图逻辑
        fig, ax = plt.subplots(figsize=(14, 8))
        # 使用 hue 映射避免 FutureWarning
        sns.violinplot(data=combined_df, x='Model', y='Absolute Error', hue='Model', ax=ax, inner=None, alpha=0.3, palette="muted", legend=False)
        sns.boxplot(data=combined_df, x='Model', y='Absolute Error', hue='Model', ax=ax, width=0.3, showfliers=False, palette="muted", legend=False)
        
        # 添加均值标注
        means = combined_df.groupby('Model')['Absolute Error'].mean().to_dict()
        for i, label in enumerate(ax.get_xticklabels()):
            m_val = means.get(label.get_text(), 0)
            ax.text(i, m_val, f'Mean:{m_val:.4f}', ha='center', va='bottom', fontweight='bold', color='darkred', fontsize=10)

        ax.set_title(f"跨模型预测误差分布对比 (全 {len(csv_files)} 折数据聚合)", fontsize=16, fontweight='bold')
        ax.set_ylabel("绝对误差 (MAE Distribution)", fontsize=12)
        ax.grid(True, axis='y', ls='--', alpha=0.4)
        
        self._savefig(fig, fname)


# =========================================================
# 使用示例
# =========================================================

def example_usage():
    """完整使用示例"""
    
    # 1. 创建 Visualizer
    viz = Visualizer(save_dir="./outputs/plots_v2")
    
    # 2. 准备数据
    N, Steps = 1000, 5
    y_pred_full = np.random.randn(N, Steps, 5).cumsum(axis=1)  # (N, Steps, 5)
    y_true_full = np.random.randn(N, Steps).cumsum(axis=1)      # (N, Steps)
    y_obs_win = np.random.randn(N, 20)                           # (N, 20)
    
    # 3. 数据对齐
    y_pred_aligned, y_true_aligned, _ = validate_alignment(
        y_pred=y_pred_full,
        y_true=y_true_full,
        y_obs_win=y_obs_win,
        context="Test"
    )
    
    # 4. 绘制图表
    viz.plot_multistep_progression(y_pred_full, y_true_full, "multistep_mae.png")
    viz.plot_btp_panorama_enhanced(y_pred_aligned, y_true_aligned, y_obs_win, 
                                   "增强版 BTP 全景图", "btp_enhanced.png")
    viz.plot_predictions_5panel(y_pred_aligned, y_true_aligned, "5分位数分析", "5panel.png")
    
    # 5. 形态学指标 (假设有健康度数据)
    health_res = {
        "skewness": np.random.randn(N) * 0.5,
        "kurtosis_ratio": np.random.randn(N) * 0.3 + 3,
        "entropy": np.random.rand(N) * 0.8,
        "H": np.random.rand(N) * 0.3 + 0.7
    }
    viz.plot_morphology_indicators(health_res, "morphology.png")
    
    logging.info("✅ 所有图表生成完毕!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    example_usage()
