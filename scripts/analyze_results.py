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

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# ==========================================
# 配置区域
# ==========================================
# 填入你想要分析的实验根目录文件夹名
EXPERIMENT_ROOT = "实验记录\TrendAwareLite_comparison\TrendAwareLite_full_20260218_225118"
TARGET_MODEL = "enhanced_transformer" # 基准模型名称 (granulation_transformer 已废弃)

# 日志设置 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def collect_model_residuals(model_dir):
    """递归收集该模型目录下所有 Fold 的残差数据"""
    # 路径通配符：搜索 cv_results 下所有 Fold 文件夹及其子目录中的 residual_analysis.csv
    search_pattern = os.path.join(model_dir, "cv_results", "Fold_*", "**", "residual_analysis.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        return None, 0

    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if 'Residual' in df.columns:
                all_dfs.append(df)
        except Exception as e:
            logging.warning(f"读取文件失败 {f}: {e}")
    
    if not all_dfs:
        return None, 0
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df, len(csv_files)

def run_offline_analysis(root_dir, target_model_name):
    if not os.path.exists(root_dir):
        logging.error(f"实验目录不存在: {root_dir}")
        return

    logging.info(f"开始离线统计分析: {root_dir}")
    
    # 1. 寻找所有 Compare_ 目录
    model_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("Compare_")]
    
    results_summary = []
    plot_data_list = []
    
    # 首先加载主模型数据作为基准
    target_dir = os.path.join(root_dir, f"Compare_{target_model_name}")
    df_target, n_folds_target = collect_model_residuals(target_dir)
    
    if df_target is None:
        logging.error(f"找不到主模型 {target_model_name} 的残差数据，请检查路径！")
        return

    target_errors = np.abs(df_target['Residual'].values)
    mae_target = np.mean(target_errors)
    logging.info(f"主模型 {target_model_name} 已载入: {len(target_errors)} 样本, 来自 {n_folds_target} 个 Fold 文件")

    # 2. 遍历其他模型进行对比
    for m_dir_name in model_dirs:
        m_name = m_dir_name.replace("Compare_", "")
        m_path = os.path.join(root_dir, m_dir_name)
        
        df_comp, n_folds = collect_model_residuals(m_path)
        if df_comp is None:
            continue
            
        comp_errors = np.abs(df_comp['Residual'].values)
        mae_comp = np.mean(comp_errors)
        
        # 准备绘图数据
        plot_data_list.append(pd.DataFrame({
            'Absolute Error': comp_errors,
            'Model': m_name.upper()
        }))

        # 显著性检验 (Wilcoxon)
        # 对齐长度（防止由于异常中断导致的不同模型样本数不一致）
        min_len = min(len(target_errors), len(comp_errors))
        a, b = target_errors[:min_len], comp_errors[:min_len]
        
        stat, p_value = wilcoxon(a, b)
        improvement = (mae_comp - mae_target) / mae_comp * 100
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "N.S."

        results_summary.append({
            "Model": m_name,
            "Folds": n_folds,
            "Total_Samples": len(comp_errors),
            "MAE_Base": round(mae_comp, 4),
            "MAE_Proposed": round(mae_target, 4),
            "Improvement(%)": f"{improvement:.2f}%",
            "P-Value": f"{p_value:.4e}",
            "Sig": sig_marker
        })
        logging.info(f"对比完成: {m_name.upper()} | P-Val: {p_value:.4e} ({sig_marker})")

    # 3. 保存 CSV 报告
    report_df = pd.DataFrame(results_summary)
    report_save_path = os.path.join(root_dir, "offline_statistical_report.csv")
    report_df.to_csv(report_save_path, index=False)
    
    # 4. 生成对比图
    if plot_data_list:
        plt.figure(figsize=(14, 8))
        all_plot_df = pd.concat(plot_data_list, ignore_index=True)
        
        # 设置风格
        sns.set_theme(style="whitegrid")
        sns.violinplot(data=all_plot_df, x='Model', y='Absolute Error', hue='Model', inner=None, alpha=0.3, palette="muted", legend=False)
        sns.boxplot(data=all_plot_df, x='Model', y='Absolute Error', hue='Model', width=0.3, showfliers=False, palette="muted", legend=False)
        
        # 标注均值
        for i, row in report_df.iterrows():
            plt.text(i, row['MAE_Base'], f"Mean:{row['MAE_Base']:.4f}", ha='center', va='bottom', fontweight='bold', color='darkred')
        
        plt.title(f"全折聚合误差分布对比 (N={len(target_errors)} samples)", fontsize=15, fontweight='bold')
        plt.ylabel("Absolute Error (MAE)")
        
        plot_path = os.path.join(root_dir, "offline_model_comparison_boxplot.png")
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        
    print("\n" + "="*60)
    print(f"✅ 分析完成！")
    print(f"📊 报告已更新: {report_save_path}")
    print(f"🖼️ 图表已更新: {plot_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_offline_analysis(EXPERIMENT_ROOT, TARGET_MODEL)
