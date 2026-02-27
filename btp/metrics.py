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

# metrics.py
import numpy as np
import logging
from typing import Dict, Optional

def compute_winkler_score(
    y_true: np.ndarray, 
    lower: np.ndarray, 
    upper: np.ndarray, 
    alpha: float
) -> float:
    """
    ✅ 标准指标：温克勒分数 (Winkler Score / Interval Score)
    越小越好。综合衡量了区间的覆盖率和宽度。
    """
    # 1. 基础宽度惩罚
    width = upper - lower
    
    # 2. 违规惩罚 (Penalty for missing the target)
    left_penalty = np.maximum(0.0, lower - y_true) * (2.0 / alpha)
    right_penalty = np.maximum(0.0, y_true - upper) * (2.0 / alpha)
    
    score = width + left_penalty + right_penalty
    return float(np.mean(score))

def evaluate_quantile_regression(
    y_true: np.ndarray,       # (N,) 或 (N, 5) 或 (N, Steps)
    y_pred: np.ndarray,       # (N, 5) 或 (N, Steps, 5)
    scaler_y=None,
    **kwargs
) -> Dict[str, float]:
    """
    ✅ 全能评估入口：自动处理单步或多步预测
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    # --- A. 维度适配 ---
    # 如果是 MIMO (N, Steps, 5)，取最后一步
    if y_pred.ndim == 3:
        y_pred = y_pred[:, -1, :] 
    
    # 真值处理：如果是 (N, Steps) 取最后一步；如果是 (N, 5) 分布取中间
    if y_true.ndim == 2:
        if y_true.shape[1] == 5:
            y_true = y_true[:, 2] # 取 Q50 作为标量真值
        elif y_true.shape[1] == y_pred.shape[0]: # 可能的转置情况
             pass 
        else:
            y_true = y_true[:, -1] # 取最后一步 Time Step
            
    if y_true.ndim > 1:
        y_true = y_true.ravel()

    # --- B. 反标准化 ---
    if scaler_y is not None and y_pred.ndim == 2:
        y_pred = scaler_y.inverse_transform(y_pred)

    # --- C. 提取分位数 ---
    q10, q25, q50, q75, q90 = [y_pred[:, i] for i in range(5)]
    
    # --- D. 计算指标 ---
    mae = np.mean(np.abs(y_true - q50))
    rmse = np.sqrt(np.mean((y_true - q50)**2))
    mape = np.mean(np.abs((y_true - q50) / (np.abs(y_true) + 1e-6))) * 100
    
    # Coverage
    picp_inner = np.mean((y_true >= q25) & (y_true <= q75)) * 100.0
    picp_outer = np.mean((y_true >= q10) & (y_true <= q90)) * 100.0
    
    # Width
    mpiw_inner = np.mean(q75 - q25)
    mpiw_outer = np.mean(q90 - q10)
    
    # Winkler Score
    winkler_inner = compute_winkler_score(y_true, q25, q75, alpha=0.5)
    winkler_outer = compute_winkler_score(y_true, q10, q90, alpha=0.2)
    
    return {
        "MAE": mae, "RMSE": rmse, "MAPE": mape,
        "PICP_50": picp_inner, "PICP_80": picp_outer,
        "MPIW_50": mpiw_inner, "MPIW_80": mpiw_outer,
        "Winkler_50": winkler_inner, "Winkler_80": winkler_outer
    }

def print_metrics(metrics: Dict[str, float], title: str = "Evaluation"):
    logging.info(f"\n{'='*60}")
    logging.info(f"  📝 {title} Report (Standard Metrics)")
    logging.info(f"{'='*60}")
    
    # 安全获取，防止 key error
    def get(k): return metrics.get(k, 0.0)

    logging.info(f"🎯 准确度 (Accuracy):")
    logging.info(f"  • MAE (Q50):      {get('MAE'):.4f}")
    logging.info(f"  • RMSE:           {get('RMSE'):.4f}")
    logging.info(f"  • MAPE:           {get('MAPE'):.2f}%")

    logging.info(f"\n🛡️ 区间质量 (Interval Quality):")
    logging.info(f"  [Inner 50%] Cov: {get('PICP_50'):.1f}% | Width: {get('MPIW_50'):.4f} | Winkler: {get('Winkler_50'):.4f}")
    logging.info(f"  [Outer 80%] Cov: {get('PICP_80'):.1f}% | Width: {get('MPIW_80'):.4f} | Winkler: {get('Winkler_80'):.4f}")
    logging.info(f"{'='*60}\n")
