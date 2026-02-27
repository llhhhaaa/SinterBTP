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
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import logging

def perform_significance_test(experiment_root: str, target_model: str = "enhanced_transformer"):
    """
    自动汇总所有折(Folds)的残差并执行显著性检验
    """
    # --- stats.py ---

    def collect_cv_residuals(model_name):
        import glob
        # 修正后的搜索路径：Compare_xxx/cv_results/Fold_X/Standard/diagnostics/residual_analysis.csv
        # 我们直接用通配符解决层级不确定的问题
        model_path = os.path.join(experiment_root, f"Compare_{model_name}", "cv_results")
        
        # 递归查找所有 Fold 下的残差文件
        search_pattern = os.path.join(model_path, "Fold_*", "**", "residual_analysis.csv")
        res_files = glob.glob(search_pattern, recursive=True)
        
        all_residuals = []
        for f in res_files:
            try:
                df = pd.read_csv(f)
                all_residuals.append(df)
                # logging.info(f"成功载入折数据: {f}")
            except:
                continue
                
        if not all_residuals:
            return None
            
        return pd.concat(all_residuals, ignore_index=True)

    # 1. 提取主模型数据
    df_target = collect_cv_residuals(target_model)
    if df_target is None:
        print(f"❌ 未找到主模型 {target_model} 的 CV 残差数据。")
        return
    
    target_errors = np.abs(df_target['Residual'].values)
    mae_target_all = np.mean(target_errors)

    print(f"\n" + "="*60)
    print(f"📊 统计显著性检验报告 (基于全折汇总, 对比基准: {target_model})")
    print(f"主模型全折平均 MAE: {mae_target_all:.4f}")
    print("="*60)

    results = []
    # 2. 自动寻找其他 Compare_ 文件夹
    model_dirs = [d for d in os.listdir(experiment_root) if d.startswith("Compare_")]

    for m_dir in model_dirs:
        m_name = m_dir.replace("Compare_", "")
        if m_name == target_model: continue
        
        df_comp = collect_cv_residuals(m_name)
        if df_comp is None: continue
        
        comp_errors = np.abs(df_comp['Residual'].values)
        
        # 对齐长度（防止异常情况）
        min_len = min(len(target_errors), len(comp_errors))
        a, b = target_errors[:min_len], comp_errors[:min_len]
        
        # 执行 Wilcoxon 检验
        stat, p_value = wilcoxon(a, b)
        mae_comp = np.mean(b)
        improvement = (mae_comp - mae_target_all) / mae_comp * 100
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "N.S."
        
        results.append({
            "Model": m_name,
            "MAE_Base": round(mae_comp, 4),
            "MAE_Proposed": round(mae_target_all, 4),
            "Improvement(%)": f"{improvement:.2f}%",
            "P-Value": f"{p_value:.4e}",
            "Significance": sig_marker
        })
        
        print(f"[Stats] {target_model} vs {m_name.upper()}:")
        print(f"   - 基线 MAE: {mae_comp:.4f} | 提升: {improvement:.2f}%")
        print(f"   - P-Value: {p_value:.4e} ({sig_marker})")

    # 3. 保存最终汇总表
    res_df = pd.DataFrame(results)
    save_path = os.path.join(experiment_root, "statistical_comparison_report.csv")
    res_df.to_csv(save_path, index=False)
    print(f"\n[Stats] 汇总统计报告已保存: {save_path}")
    return res_df
