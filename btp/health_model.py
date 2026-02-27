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

# health_model.py
"""
烧结BTP多维动态势能健康度体系 (MDPHI) - v2.0
Multi-Dimensional Potential Health Index for Sintering BTP

核心特性：
1. 同构性 (Isomorphism)：统一处理标量真值和分位数分布
2. 三维正交分量：静态偏离度 + 动态稳定性 + 趋势风险度
3. 二阶临界阻尼平滑 (Second-Order Critical Damping)
4. 状态迟滞判定 (Hysteresis State Machine)
"""

import numpy as np
import logging
from typing import Union, List, Dict, Tuple, Optional
from collections import deque


class HealthModel:
    """
    🏥 BTP健康度评估模型 + 工况状态诊断 (增强平滑版)
    """
    
    # 定义状态常量映射
    STATE_MAP = {
        0: "过烧",       # Over-burn (Severe)
        1: "疑似过烧",   # Suspected Over-burn
        2: "正常",       # Normal
        3: "疑似欠烧",   # Suspected Under-burn
        4: "欠烧"        # Under-burn (Severe)
    }
    
    def __init__(self, config):
        self.cfg = config
        
        # ========== 目标值与参数 ==========
        self.mu = getattr(config, 'health_mu', 22.5)
        self.sigma_left = getattr(config, 'health_sigma_left', 0.3)
        self.sigma_right = getattr(config, 'health_sigma_right', 0.5)
        self.width_tolerance = getattr(config, 'health_width_tol', 1.5)         
        
        # ========== 阈值体系 (基准) ==========
        self.thresh_normal = 65.0    # 健康阈值 (0-100分制)
        self.thresh_fault = 38.0     # 故障/疑似分界线
        self.hysteresis_band = 2.0   # [新增] 迟滞带宽，防止状态闪烁
        
        # ========== 权重参数 ==========
        self.history_window = getattr(config, 'volatility_window_size', 10)
        self.sigma_limit = getattr(config, 'health_sigma_limit', 0.15)
        self.k_stab = getattr(config, 'health_k_stab', 2.0)
        self.alpha_trend = getattr(config, 'health_alpha_trend', 0.8)
        self.W_pos = getattr(config, 'health_W_pos', 1.2)
        self.W_stab = getattr(config, 'health_W_stab', 0.8)
        self.W_trend = getattr(config, 'health_W_trend', 1.0)
        
        # ========== 平滑参数 (二阶滤波) ==========
        # beta 越大越平滑 (0~1)。二阶滤波建议 0.6~0.8，比一阶需要更小一点的值就能达到同样的平滑度
        raw_beta = getattr(config, 'health_beta_ewma', 0.9) 
        # 内部微调：二阶滤波因为串联了两次，如果沿用一阶的参数会变得过于迟钝
        # 所以这里做一个映射，或者直接使用。通常 sqrt(beta) 是个不错的起点，但保持原值能获得更强平滑。
        self.beta = raw_beta 
        
        # ========== 运行时状态 ==========
        self.history_queue = deque(maxlen=self.history_window)
        self.r_prev = None
        
        # [新增] 滤波器状态向量 [Stage1_Output, Stage2_Output]
        # 初始设为 1.0 (100分)，避免冷启动从0爬升
        self.filter_state = {'s1': 1.0, 's2': 1.0} 
        
        # [新增] 上一时刻的状态 ID (用于迟滞判断)
        self.last_state_idx = 2 # 默认为正常
        
        logging.info(f"[HealthModel] v2.0 初始化 | 二阶平滑β={self.beta} | 迟滞带=±{self.hysteresis_band}")
    
    # =========================================================
    # 核心组件 A：静态偏离度 H_pos
    # =========================================================
    def _compute_H_pos(self, input_data: Union[float, np.ndarray]) -> float:
        """
        计算静态偏离度（主分-罚分机制）
        """
        if isinstance(input_data, (int, float)):
            q50 = float(input_data)
            width = 0.0 
        else:
            Q = np.asarray(input_data).flatten()
            if len(Q) == 5:
                q50 = float(Q[2])       
                width = float(Q[4] - Q[0]) 
            else:
                q50 = float(np.mean(Q))
                width = 0.0 
        
        # 基准分
        sigma = self.sigma_left if q50 < self.mu else self.sigma_right
        deviation = (q50 - self.mu) ** 2
        base_score = np.exp(-deviation / (2 * sigma ** 2))
        
        # 罚分系数
        penalty = 0.0
        if width > 0:
            ratio = width / self.width_tolerance
            max_penalty = 0.3
            penalty = max_penalty * (1.0 - np.exp(-0.5 * ratio))
            
        H_pos = base_score * (1.0 - penalty)
        return float(np.clip(H_pos, 0.0, 1.0))


    # =========================================================
    # 核心组件 B：动态稳定性 H_stab
    # =========================================================
    def _compute_H_stab(self, input_data: Union[float, np.ndarray]) -> float:
        """
        计算动态稳定性
        """
        if isinstance(input_data, (int, float)):
            r_t = float(input_data)
        else:
            Q = np.asarray(input_data).flatten()
            r_t = float(Q[2])
        
        self.history_queue.append(r_t)
        
        if len(self.history_queue) < 2:
            return 1.0
        
        sigma_roll = float(np.std(self.history_queue))
        
        ratio = sigma_roll / self.sigma_limit
        H_stab = 1.0 / (1.0 + ratio ** self.k_stab)
        
        return float(np.clip(H_stab, 0.0, 1.0))
    
    # =========================================================
    # 核心组件 C：趋势风险度 H_trend
    # =========================================================
    def _compute_H_trend(self, input_data: Union[float, np.ndarray]) -> float:
        """
        计算趋势风险度
        """
        if isinstance(input_data, (int, float)):
            r_t = float(input_data)
        else:
            Q = np.asarray(input_data).flatten()
            r_t = float(Q[2])
        
        if self.r_prev is None:
            self.r_prev = r_t
            return 1.0
        
        v_t = r_t - self.r_prev
        bias = r_t - self.mu
        D = np.sign(bias) * v_t
        
        H_trend = np.exp(-self.alpha_trend * np.maximum(0, D))
        self.r_prev = r_t
        
        return float(np.clip(H_trend, 0.0, 1.0))
    
    # =========================================================
    # [升级] 融合与二阶平滑
    # =========================================================
    def _fuse_and_smooth_v2(
        self, 
        H_pos: float, 
        H_stab: float, 
        H_trend: float
    ) -> float:
        """
        [算法升级] 乘性融合 + 二阶临界阻尼平滑 (Cascaded EWMA)
        
        数学原理：
        相当于两个一阶RC低通滤波器串联。
        Stage 1: y1[t] = β * y1[t-1] + (1-β) * x[t]
        Stage 2: y2[t] = β * y2[t-1] + (1-β) * y1[t]
        
        特性：
        1. 相比一阶，它对高频噪声（毛刺）的衰减是指数级的。
        2. 没有超调（Overshoot），不会因为平滑导致分数冲出 0~1 范围。
        """
        # 1. 瞬时融合 (Instantaneous Fusion)
        H_inst = (H_pos ** self.W_pos)* \
                 ( H_stab ** self.W_stab) * \
                 ( H_trend ** self.W_trend)
        
        # 2. 读取上一时刻滤波器状态
        s1_prev = self.filter_state['s1']
        s2_prev = self.filter_state['s2']
        
        # 3. 第一级平滑 (Stage 1 Filter)
        s1_curr = self.beta * s1_prev + (1 - self.beta) * H_inst
        
        # 4. 第二级平滑 (Stage 2 Filter) - 输入是上一级的输出
        s2_curr = self.beta * s2_prev + (1 - self.beta) * s1_curr
        
        # 5. 更新状态
        self.filter_state['s1'] = s1_curr
        self.filter_state['s2'] = s2_curr
        
        return float(np.clip(s2_curr, 0.0, 1.0))

    # =========================================================
    # [升级] 状态判定 (带迟滞)
    # =========================================================
    def _determine_state_hysteresis(self, health_score_val: float, current_val: float) -> int:
        """
        基于施密特触发器逻辑的状态判定
        防止在 65分 或 38分 附近反复跳变
        """
        # health_score_val 是 0~100 的值
        
        # 1. 确定方向 (Direction)
        # 偏大(欠烧方向) vs 偏小(过烧方向)
        is_under_burn_side = (current_val > self.mu)
        
        # 2. 动态调整阈值 (Hysteresis Logic)
        # 如果上一刻是正常(2)，那么进入异常需要更低的分数 (thresh - band)
        # 如果上一刻是异常(<2 or >2)，那么回到正常需要更高的分数 (thresh + band)
        
        eff_thresh_normal = self.thresh_normal
        eff_thresh_fault = self.thresh_fault
        
        if self.last_state_idx == 2:
            # 当前是正常，变坏稍微难一点
            eff_thresh_normal -= self.hysteresis_band
        else:
            # 当前是异常，变好稍微难一点
            eff_thresh_normal += self.hysteresis_band
            
        # 同样的逻辑应用于 疑似 vs 严重
        # 这里简化逻辑：主要防止 正常 <-> 疑似 的跳变
        
        # 3. 判定逻辑
        new_state = 2
        
        if health_score_val >= eff_thresh_normal:
            new_state = 2 # 正常
        else:
            # 进入非正常区域
            is_suspected = (health_score_val >= eff_thresh_fault)
            
            if is_under_burn_side:
                new_state = 3 if is_suspected else 4
            else:
                new_state = 1 if is_suspected else 0
        
        self.last_state_idx = new_state
        return new_state

    # =========================================================
    # 主接口：批量分析
    # =========================================================
    def analyze(self, y_pred: np.ndarray, y_true: np.ndarray = None) -> Dict:
        """
        批量分析
        """
        # 重置状态
        self.history_queue.clear()
        self.r_prev = None
        self.filter_state = {'s1': 1.0, 's2': 1.0} # 默认满分起步
        self.last_state_idx = 2
        
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
        N = len(y_pred)
        
        health_scores = np.zeros(N)
        state_indices = np.zeros(N, dtype=int)
        
        H_pos_s, H_stab_s, H_trend_s = np.zeros(N), np.zeros(N), np.zeros(N)
        
        for t in range(N):
            # 1. 数据解析
            if y_pred.shape[1] == 5:
                current_input = y_pred[t, :]
                representative_val = float(current_input[2]) # Q50
            else:
                current_input = float(y_pred[t, 0])
                representative_val = current_input
            
            # 2. 计算基础分量
            H_pos = self._compute_H_pos(current_input)
            H_stab = self._compute_H_stab(current_input)
            H_trend = self._compute_H_trend(current_input)
            
            # 3. 二阶平滑融合
            H_final = self._fuse_and_smooth_v2(H_pos, H_stab, H_trend)
            score_val = H_final * 100
            
            # 4. 带迟滞的状态判定
            state_idx = self._determine_state_hysteresis(score_val, representative_val)
            
            # 5. 存储
            health_scores[t] = score_val
            state_indices[t] = state_idx
            H_pos_s[t] = H_pos
            H_stab_s[t] = H_stab
            H_trend_s[t] = H_trend
            
        results = {
            'health_scores': health_scores,
            'pred_states': state_indices,
            'H_pos_series': H_pos_s,
            'H_stab_series': H_stab_s,
            'H_trend_series': H_trend_s,
            'mean_health': float(np.mean(health_scores)),
            'min_health': float(np.min(health_scores)),
        }
        
        # =========================================================
        # [修正版] 处理真值：引入历史窗口计算分位数，实现与预测端完全对齐
        # =========================================================
        if y_true is not None:
            # 重置内部状态，确保评价独立
            self.history_queue.clear()
            self.r_prev = None
            self.filter_state = {'s1': 1.0, 's2': 1.0}
            self.last_state_idx = 2
            
            if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
            
            # [关键修正] 确定真值所在的列索引
            # 如果是 (N, 5) 分布矩阵，真值在 index 2；如果是 (N, 1)，真值在 index 0
            true_col_idx = 2 if y_true.shape[1] == 5 else 0
            
            true_health = np.zeros(N)
            true_states = np.zeros(N, dtype=int)
            
            # [关键设置] 历史窗口大小
            # 使用 forecast_steps 作为回溯窗口大小
            LOOKBACK_STEPS = max(5, int(getattr(self.cfg, 'forecast_steps', 3) * 4))
            
            for t in range(N):
                # 1. 动态切片：获取包含当前时刻的过去窗口数据
                # 比如 t=10, step=2 -> 取索引 9, 10 两个点
                start_idx = max(0, t - LOOKBACK_STEPS + 1)
                end_idx = t + 1 
                
                # [关键修正] 始终取真值列 (Scalar Truth) 来构建历史分布
                window_vals = y_true[start_idx : end_idx, true_col_idx]
                
                # 2. 计算分位数 (Q10, Q25, Q50, Q75, Q90)
                if len(window_vals) > 0:
                    t_in = np.percentile(window_vals, [10, 25, 50, 75, 90])
                else:
                    val = float(y_true[t, true_col_idx])
                    t_in = np.array([val, val, val, val, val])

                # 提取中位数用于状态机判断
                t_val = float(t_in[2])
                
                # 3. 传入模型
                hp = self._compute_H_pos(t_in)
                hs = self._compute_H_stab(t_in)
                ht = self._compute_H_trend(t_in)
                hf = self._fuse_and_smooth_v2(hp, hs, ht)
                
                true_health[t] = hf * 100
                true_states[t] = self._determine_state_hysteresis(hf * 100, t_val)
                
            results['true_health_scores'] = true_health
            results['true_states'] = true_states

        return results


    def get_diagnostics(self) -> Dict:
        return {
            'mu': self.mu,
            'last_score': self.filter_state['s2'] * 100,
            'filter_stage1': self.filter_state['s1'],
            'current_state': self.STATE_MAP.get(self.last_state_idx, "Unknown")
        }

# =========================================================
# 效果对比测试
# =========================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import matplotlib.pyplot as plt
    
    class MockConfig:
        volatility_window_size = 10
        health_beta_ewma = 0.7 # 0.7 的二阶平滑非常强力
    
    config = MockConfig()
    model = HealthModel(config)
    
    # 构造一个带噪声的阶跃信号
    # 0-30: 正常
    # 30-70: 故障 (带大量噪声)
    # 70-100: 恢复
    t = np.arange(100)
    signal = np.full(100, 22.5) # Target
    
    # 故障区
    signal[30:70] = 23.5 # 偏离
    # 加噪声
    noise = np.random.normal(0, 0.3, 100)
    signal += noise
    
    # 运行模型
    res = model.analyze(signal)
    
    # 绘图
    plt.figure(figsize=(12, 6))
    
    # 原始信号
    ax1 = plt.gca()
    ax1.plot(signal, 'k-', alpha=0.3, label='Raw Signal (BTP)')
    ax1.set_ylabel('BTP Value')
    
    # 健康度
    ax2 = ax1.twinx()
    ax2.plot(res['health_scores'], 'b-', linewidth=2, label='Health Score (2nd Order)')
    
    # 画阈值线
    ax2.axhline(65, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(38, color='r', linestyle='--', alpha=0.5)
    
    ax2.set_ylim(0, 110)
    ax2.set_ylabel('Health Score')
    
    plt.title(f"2nd Order Smoothing Test (Beta={config.health_beta_ewma})")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Test finished. Check the plot for smoothness.")
