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

# calibrator.py
# =========================================================
# 在线预测校准器 V5 - 滑动窗口中位数偏差平移（简化版）
#
# 设计原则：
# 1. Bias = 固定大小滑动窗口内误差的中位数（稳定且能跟上漂移）
# 2. 每步更新 bias（中位数天然鲁棒，不会追踪噪声）
# 3. 消融实验表明仅平移效果最佳，已移除宽度缩放功能
# =========================================================

import numpy as np
import logging
from collections import deque


class OnlineGradientCalibrator:
    """
    在线校准器 V5：滑动窗口中位数偏差平移（简化版）。
    
    核心设计：
    - 用固定大小 deque 做滑动窗口，不让早期数据稀释近期偏差
    - 每步更新 bias（中位数本身就够稳定，不需要阶梯式更新）
    - 消融实验表明仅平移效果最佳，已移除宽度缩放功能
    """
    def __init__(
        self,
        bias_window: int = 60,       # 滑动窗口大小
        min_samples: int = 20,       # 最少校准样本数
    ):
        self.bias_window = bias_window
        self.min_samples = min_samples
        
        logging.info(f"[Calibrator V5] Init: bias_window={self.bias_window}, min_samples={self.min_samples}")

    def apply(self, y_pred_abs, y_true_abs, delay_steps=0):
        """
        因果在线校准：严格遵守 delay_steps 延迟约束，仅做偏差平移。
        
        因果约束：
            在时刻 t 做校准时，只能使用 t - delay_steps 及更早的真值作为反馈。
            delay_steps 应 = ceil(forecast_steps / stride)，由调用方根据
            粒化窗口大小和预测步长计算后传入。
        
        Args:
            y_pred_abs: (N, 5) [Q10, Q25, Q50, Q75, Q90]，粒化后序列
            y_true_abs: (N,) 或 (N, 5)，粒化后序列
            delay_steps: int, 因果延迟步数（粒化步为单位）
            
        Returns:
            y_calib: (N, 5) 校准后
            corrections: (N, 2) [bias, _]（第二列保留兼容，恒为1.0）
            diag_list: list
        """
        assert delay_steps >= 1, (
            f"[Calibrator] delay_steps={delay_steps} 必须 >= 1，否则存在数据泄漏风险！"
            f"请检查 forecast_steps 和 stride 的计算。"
        )
        n = len(y_pred_abs)
        y_calib = y_pred_abs.copy().astype(np.float64)
        corrections = np.zeros((n, 2))
        corrections[:, 1] = 1.0  # 保留兼容，恒为1.0
        
        # 提取真值标量
        if y_true_abs.ndim == 2:
            y_true = y_true_abs[:, 2].copy()
        else:
            y_true = y_true_abs.copy()
        
        # 固定大小滑动窗口
        error_ring = deque(maxlen=self.bias_window)
        
        # 当前偏差
        current_bias = 0.0
        
        diag_list = []
        
        for t in range(n):
            # =========================================================
            # Phase 1: Apply — 用当前偏差校准
            # =========================================================
            
            # 全局偏差平移
            y_calib[t, :] += current_bias
            
            # 强制排序确保分位数单调性
            y_calib[t] = np.sort(y_calib[t])
            
            corrections[t, 0] = current_bias
            
            # =========================================================
            # Phase 2: 收集反馈 + 更新偏差
            # =========================================================
            feedback_t = t - delay_steps
            
            if feedback_t >= 0:
                # 误差 = 真值 - 原始预测中位数（不是校准后的）
                error = y_true[feedback_t] - y_pred_abs[feedback_t, 2]
                error_ring.append(error)
            
            # 每步更新 bias（中位数天然鲁棒，不会追踪噪声）
            if len(error_ring) >= self.min_samples:
                current_bias = float(np.median(error_ring))
        
        logging.info(f"[Calibrator V5] 完成: final_bias={current_bias:.4f}")
        
        return y_calib, corrections, diag_list


# 保留旧类以兼容
class RobustCalibrator:
    pass

class AdaptiveConformalCalibrator:
    pass
