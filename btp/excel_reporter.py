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
import logging
import pandas as pd
import numpy as np
from btp.utils import pretty_title

class ExcelReporter:
    """
    Excel 报告生成器（修正版）
    变更点：
    1. 移除无意义的 "真实_Qxx" 分布列。
    2. 仅保留唯一的 "真实值" 与 "预测分布" 进行对比。
    3. 增加是否命中区间的直观判断列。
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _extract_scalar_truth(self, y_true: np.ndarray) -> np.ndarray:
        """
        辅助函数：从 y_true 中提取标量真值。
        main.py 中为了 shape 对齐可能构造了 (N, 5) 的 y_true，
        其中 index=2 通常存放的是真实标量值。
        """
        if y_true.ndim == 2 and y_true.shape[1] == 5:
            # 约定：中间列是真实值
            return y_true[:, 2]
        elif y_true.ndim == 2 and y_true.shape[1] == 1:
            return y_true.ravel()
        elif y_true.ndim == 1:
            return y_true
        else:
            # 兜底：如果是其他形状，强制拉平取第一个或报错，这里假设是 (N,)
            return y_true.ravel()

    def export_comparison_results(self, full_data: dict, config):
        """
        导出校准前后对比报告 (Validation Calibration Report)
        """
        excel_path = os.path.join(self.save_dir, "validation_calibration_report.xlsx")
        
        y_raw = full_data["y_pred_raw"]    # (N, 5)
        y_calib = full_data["y_pred_calib"] # (N, 5)
        y_true_in = full_data["y_true"]     # (N, 5) 或 (N,)
        corrections = full_data["corrections"] 
        
        # 1. 提取唯一的标量真值
        y_true_scalar = self._extract_scalar_truth(y_true_in)

        # 2. 构建数据字典
        data_dict = {
            "样本ID": range(len(y_raw)),
            # --- 核心真值 ---
            "真实值": y_true_scalar,
            
            # --- 原始预测分布 ---
            "原始_Q10_下界": y_raw[:, 0],
            "原始_Q50_中位": y_raw[:, 2],
            "原始_Q90_上界": y_raw[:, 4],
            "原始_误差(True-Median)": y_true_scalar - y_raw[:, 2],
            
            # --- 校准后预测分布 ---
            "校准_Q10_下界": y_calib[:, 0],
            "校准_Q25": y_calib[:, 1],
            "校准_Q50_中位": y_calib[:, 2],
            "校准_Q75": y_calib[:, 3],
            "校准_Q90_上界": y_calib[:, 4],
            
            # --- 直观判断 ---
            "校准_误差(True-Median)": y_true_scalar - y_calib[:, 2],
            "是否命中_80%区间": ((y_true_scalar >= y_calib[:, 0]) & (y_true_scalar <= y_calib[:, 4])).astype(int),
            "是否命中_50%区间": ((y_true_scalar >= y_calib[:, 1]) & (y_true_scalar <= y_calib[:, 3])).astype(int),
        }
        
        # 添加校准因子详情
        if corrections.shape[1] >= 2:
            data_dict["校准因子_Inner"] = corrections[:, 0]
            data_dict["校准因子_Outer"] = corrections[:, 1]
        
        df_detail = pd.DataFrame(data_dict)
        
        # --- Sheet 2: 指标对比 ---
        metrics_raw = full_data["metrics_raw"]
        metrics_calib = full_data["metrics_calib"]
        
        keys = [k for k in metrics_raw.keys() if isinstance(metrics_raw[k], (int, float))]
        
        df_metrics = pd.DataFrame({
            "指标名称": keys,
            "原始模型": [metrics_raw.get(k, 0) for k in keys],
            "校准后模型": [metrics_calib.get(k, 0) for k in keys],
            "提升值 (校准-原始)": [metrics_calib.get(k, 0) - metrics_raw.get(k, 0) for k in keys]
        })

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_detail.to_excel(writer, sheet_name="逐样本对比", index=False)
            df_metrics.to_excel(writer, sheet_name="指标对比", index=False)
            
        logging.info(f"[Export] 校准对比报告已生成: {excel_path}")
        
    def export_results(
        self,
        eval_results: dict,
        health_results: dict,
        config,
    ):
        """
        导出最终测试的完整分析结果
        """
        pretty_title("Step 10  导出 Excel 报告 / Export Results")

        excel_path = os.path.join(self.save_dir, "analysis_report.xlsx")

        # ========== Sheet 1: 预测详情 ==========
        y_pred = eval_results["y_pred_orig"] # 这里通常是校准后的最终输出
        y_true_in = eval_results["y_true_orig"]
        
        # 1. 提取唯一的标量真值
        y_true_scalar = self._extract_scalar_truth(y_true_in)
        
        # 计算点预测误差
        errors = y_true_scalar - y_pred[:, 2]

        df_pred = pd.DataFrame({
            "样本ID": range(len(y_pred)),
            # 核心修正：只保留一列真实值
            "真实值": y_true_scalar,
            
            "预测_Q10": y_pred[:, 0],
            "预测_Q25": y_pred[:, 1],
            "预测_Q50": y_pred[:, 2],
            "预测_Q75": y_pred[:, 3],
            "预测_Q90": y_pred[:, 4],
            
            "点预测误差(Abs)": np.abs(errors),
            "区间覆盖_Inner(50%)": ((y_true_scalar >= y_pred[:, 1]) & (y_true_scalar <= y_pred[:, 3])).astype(int),
            "区间覆盖_Outer(80%)": ((y_true_scalar >= y_pred[:, 0]) & (y_true_scalar <= y_pred[:, 4])).astype(int),
        })

        # ========== Sheet 2: 健康度详情 ==========
        state_names = ["正常", "疑似偏热", "偏热", "疑似偏冷", "偏冷"]
        
        health_scores = health_results.get("health_scores", [])
        
        if len(health_scores) == 0:
            df_health = pd.DataFrame({"提示": ["无健康度数据"]})
        else:
            # 同样，在健康度表中也展示真实值，方便对照
            # 确保长度一致
            current_len = len(health_scores)
            sliced_true = y_true_scalar[-current_len:] if len(y_true_scalar) >= current_len else y_true_scalar
            
            df_health = pd.DataFrame({
                "样本ID": range(len(health_scores)),
                "真实值_BTP": sliced_true, # 方便对照
                "综合健康度_H": health_scores,
                "预测状态": [state_names[s] if 0 <= s < 5 else f"未知({s})" for s in health_results.get("pred_states", [])],
                "真实状态": [state_names[s] if 0 <= s < 5 else f"未知({s})" for s in health_results.get("true_states", [])],
                "静态偏离_H_pos": health_results.get("H_pos_series", [0]*len(health_scores)),
                "动态稳定_H_stab": health_results.get("H_stab_series", [0]*len(health_scores)),
                "趋势风险_H_trend": health_results.get("H_trend_series", [0]*len(health_scores)),
            })


        # ========== Sheet 3: 整体指标汇总 ==========
        # 安全提取统计量
        avg_h = float(np.mean(health_scores)) if len(health_scores) > 0 else 0.0
        min_h = health_results.get("min_health", 0.0)
        max_h = health_results.get("max_health", 0.0)

        # 尝试提取诊断准确率
        diag_acc = eval_results.get("diagnosis_acc", np.nan)

        metrics_data = {
            "指标类别": [
                "回归误差", "回归误差", "回归误差", "回归误差", "回归误差", "回归误差", "回归误差", "回归误差",
                "区间质量", "区间质量", "区间质量", "区间质量",
                "健康度分析", "健康度分析", "健康度分析", "健康度分析",
                "工况诊断"
            ],
            "指标名称": [
                "MAE_Q10 (Lower)", "MAE_Q25", "MAE_Q50 (Median)", "MAE_Q75", "MAE_Q90 (Upper)", "MAE_Avg_Quantiles",
                "RMSE_Q50", "MAPE_Q50",
                "IoU_Inner", "IoU_Outer",
                "Coverage_Inner (Target 50%)", "Coverage_Outer (Target 80%)",
                "Avg_Health_Score", "Min_Health_Score", "Max_Health_Score", "Health_Std",
                "诊断准确率 (Accuracy)"
            ],
            "数值": [
                eval_results.get("mae_q10", 0.0),
                eval_results.get("mae_q25", 0.0),
                eval_results.get("mae_q50", 0.0),
                eval_results.get("mae_q75", 0.0),
                eval_results.get("mae_q90", 0.0),
                eval_results.get("mae_avg", 0.0),
                eval_results.get("rmse_q50", 0.0),
                eval_results.get("mape_q50", 0.0),
                eval_results.get("iou_inner", 0.0),
                eval_results.get("iou_outer", 0.0),
                eval_results.get("coverage_inner", 0.0),
                eval_results.get("coverage_outer", 0.0),
                avg_h, min_h, max_h, health_results.get("health_std", 0.0),
                diag_acc
            ]
        }
        
        df_metrics = pd.DataFrame(metrics_data)

        # 添加状态分布统计
        counts = health_results.get("state_counts", {})
        total_samples = len(health_scores)
        if total_samples > 0:
            state_stats = []
            for name, count in counts.items():
                pct = count / total_samples * 100
                state_stats.append({
                    "指标类别": "状态分布",
                    "指标名称": f"状态_{name}_占比", 
                    "数值": f"{count} ({pct:.1f}%)"
                })
            df_state_stats = pd.DataFrame(state_stats)
            df_metrics = pd.concat([df_metrics, df_state_stats], ignore_index=True)

        # ========== Sheet 4: 配置参数 ==========
        config_data = {
            "参数名称": list(config.to_dict().keys()),
            "参数值": [str(v) for v in config.to_dict().values()],
        }
        df_config = pd.DataFrame(config_data)

        # ========== 写入 Excel ==========
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_pred.to_excel(writer, sheet_name="预测详情", index=False)
            df_health.to_excel(writer, sheet_name="健康度与工况", index=False)
            df_metrics.to_excel(writer, sheet_name="整体指标", index=False)
            df_config.to_excel(writer, sheet_name="配置参数", index=False)

        logging.info(f"[Export] 修正版 Excel 报告已保存: {excel_path}")
