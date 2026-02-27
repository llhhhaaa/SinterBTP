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

"""
验证集可预测性诊断工具
Author: Your Name
Date: 2026-01-15
"""

import numpy as np
import logging
from scipy.stats import pearsonr
from typing import Dict

def diagnose_val_predictability(
    y_val: np.ndarray,
    config_name: str = "unknown"
) -> Dict[str, float]:
    """
    诊断验证集的可预测性（基于标签本身的特性）
    
    Args:
        y_val: (N, 3) 验证集标签 [q20, q50, q80]（原始尺度）
        config_name: 配置名称（如 "0.20" 或 "0.25"）
    
    Returns:
        包含各项指标的字典
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"开始诊断验证集可预测性 - 配置: {config_name}")
    logging.info(f"{'='*60}")
    
    # 提取中位数序列（主要分析对象）
    y_mid = y_val[:, 1]
    
    # ========== 🔍 指标1：自相关系数（趋势强度）==========
    if len(y_mid) > 1:
        autocorr, p_value = pearsonr(y_mid[:-1], y_mid[1:])
    else:
        autocorr, p_value = 0.0, 1.0
    
    # ========== 🔍 指标2：变异系数（噪声水平）==========
    mean_val = np.mean(y_mid)
    std_val = np.std(y_mid)
    cv = std_val / (np.abs(mean_val) + 1e-8)
    
    # ========== 🔍 指标3：平均跳变幅度（平滑度）==========
    if len(y_mid) > 1:
        smoothness = np.mean(np.abs(np.diff(y_mid)))
    else:
        smoothness = 0.0
    
    # ========== 🔍 指标4：信噪比（SNR）==========
    signal_power = np.var(y_mid)
    # 用一阶差分估计噪声
    noise_estimate = np.var(np.diff(y_mid)) / 2 if len(y_mid) > 1 else 1e-8
    snr = 10 * np.log10(signal_power / (noise_estimate + 1e-8))
    
    # ========== 🔍 指标5：趋势线性度（R²）==========
    if len(y_mid) > 2:
        x_time = np.arange(len(y_mid))
        # 线性拟合
        coeffs = np.polyfit(x_time, y_mid, deg=1)
        y_fit = np.polyval(coeffs, x_time)
        ss_res = np.sum((y_mid - y_fit) ** 2)
        ss_tot = np.sum((y_mid - np.mean(y_mid)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
    else:
        r_squared = 0.0
    
    # ========== 🔍 指标6：可预测性综合评分 ==========
    # 评分逻辑：高自相关 + 低噪声 + 高平滑度 = 高可预测性
    predictability_score = (
        autocorr * 0.4 +                          # 趋势权重40%
        (1 - min(cv, 1.0)) * 0.3 +               # 低噪声权重30%
        (1 - min(smoothness / 10, 1.0)) * 0.3    # 平滑度权重30%
    )
    
    report = {
        '自相关系数': float(autocorr),              # ✅ 转换
        '自相关p值': float(p_value),                # ✅ 转换
        '变异系数': float(cv),                      # ✅ 转换
        '平均跳变幅度': float(smoothness),          # ✅ 转换
        '信噪比SNR_dB': float(snr),                 # ✅ 转换
        '趋势线性度R2': float(r_squared),           # ✅ 转换
        '可预测性评分': float(predictability_score) # ✅ 转换
    }
    
    # 打印报告
    logging.info(f"\n{'─'*60}")
    logging.info(f"📊 验证集质量报告 - {config_name}")
    logging.info(f"{'─'*60}")
    for metric, value in report.items():
        # 根据指标类型添加emoji和评价
        if '自相关' in metric and 'p值' not in metric:
            emoji = '🟢' if value > 0.7 else '🟡' if value > 0.4 else '🔴'
            comment = '(强趋势)' if value > 0.7 else '(中等)' if value > 0.4 else '(弱趋势)'
        elif '变异系数' in metric:
            emoji = '🟢' if value < 0.2 else '🟡' if value < 0.5 else '🔴'
            comment = '(低噪声)' if value < 0.2 else '(中等)' if value < 0.5 else '(高噪声)'
        elif '跳变' in metric:
            emoji = '🟢' if value < 3 else '🟡' if value < 7 else '🔴'
            comment = '(平滑)' if value < 3 else '(中等)' if value < 7 else '(剧烈)'
        elif 'SNR' in metric:
            emoji = '🟢' if value > 10 else '🟡' if value > 0 else '🔴'
            comment = '(高)' if value > 10 else '(中)' if value > 0 else '(低)'
        elif '可预测性' in metric:
            emoji = '🟢' if value > 0.7 else '🟡' if value > 0.4 else '🔴'
            comment = '(易预测)' if value > 0.7 else '(中等)' if value > 0.4 else '(难预测)'
        else:
            emoji = '📌'
            comment = ''
        
        if isinstance(value, float):
            logging.info(f"{emoji} {metric:30s}: {value:8.4f} {comment}")
        else:
            logging.info(f"{emoji} {metric:30s}: {value} {comment}")
    
    logging.info(f"{'─'*60}\n")
    
    return report


def compare_val_quality(
    report_1: Dict, 
    report_2: Dict, 
    name_1: str = "Config 1", 
    name_2: str = "Config 2"
):
    """
    对比两个配置的验证集质量
    
    Args:
        report_1: 第一个配置的诊断报告
        report_2: 第二个配置的诊断报告
        name_1: 第一个配置名称
        name_2: 第二个配置名称
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"🔬 验证集质量对比：{name_1} vs {name_2}")
    logging.info(f"{'='*80}")
    
    # 打印对比表格
    logging.info(f"\n{'指标':<30s} | {name_1:^12s} | {name_2:^12s} | {'差异':^12s} | {'结论':^15s}")
    logging.info(f"{'-'*30}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}")
    
    for metric in report_1.keys():
        if metric == '自相关p值':  # 跳过p值
            continue
            
        val_1 = report_1[metric]
        val_2 = report_2[metric]
        
        if not isinstance(val_1, (int, float)) or not isinstance(val_2, (int, float)):
            continue
            
        diff = val_2 - val_1
        
        # 判断哪个更好
        if '自相关' in metric or '可预测性' in metric or 'SNR' in metric or 'R²' in metric:
            better = name_1 if val_1 > val_2 else name_2
            diff_emoji = '📈' if diff > 0 else '📉'
        elif '变异系数' in metric or '跳变' in metric:
            better = name_1 if val_1 < val_2 else name_2
            diff_emoji = '📉' if diff > 0 else '📈'
        else:
            better = '-'
            diff_emoji = '➡️'
        
        logging.info(
            f"{metric:<30s} | {val_1:12.4f} | {val_2:12.4f} | "
            f"{diff_emoji}{diff:11.4f} | {'✅ '+better:^15s}"
        )
    
    logging.info(f"{'-'*80}\n")
    
    # 总结性判断（兼容不同key）
    score_key_candidates = ["可预测性评分", "可预测性评分 (0-1)"]
    def _get_score(rep):
        for k in score_key_candidates:
            if k in rep:
                return rep[k]
        return None

    score_1 = _get_score(report_1)
    score_2 = _get_score(report_2)

    logging.info("🎯 诊断结论:")
    if score_1 is None or score_2 is None:
        logging.info("   ⚠️ 未找到可预测性评分字段，跳过综合结论。")
        return
    if score_1 > score_2 * 1.2:
        logging.info(f"   ✅ {name_1} 的验证集明显更容易预测（评分高 {(score_1/score_2-1)*100:.1f}%）")
        logging.info(f"   ⚠️  这解释了为什么模型在 {name_1} 上表现更好")
    elif score_2 > score_1 * 1.2:
        logging.info(f"   ✅ {name_2} 的验证集明显更容易预测（评分高 {(score_2/score_1-1)*100:.1f}%）")
        logging.info(f"   ⚠️  这解释了为什么模型在 {name_2} 上表现更好")
    else:
        logging.info(f"   ➡️ 两个验证集的可预测性接近（差异 < 20%）")
        logging.info(f"   💡 性能差异可能来自其他因素（如训练集质量）")
    
    logging.info(f"{'='*80}\n")
