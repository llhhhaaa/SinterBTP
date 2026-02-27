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

"""消融实验结果汇总脚本"""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

DIR_A = "data/ablation_20260222_140916"
DIR_B = "实验记录/3.消融实验/ablation_20260222_145440"
OUTPUT_DIR = "data/ablation_merged"

VARIANT_DISPLAY = {"base": "Full Model", "no_time2vec": "w/o Time2Vec", 
                   "no_fitting": "w/o Fitting", "no_revin": "w/o RevIN"}
VARIANT_COLORS = {"base": "#4C72B0", "no_time2vec": "#55A868", 
                  "no_fitting": "#C44E52", "no_revin": "#DD8452"}

def _aggregate_metrics(df_cv):
    df_test = df_cv[df_cv["split"] == "test"] if "split" in df_cv.columns else df_cv
    if len(df_test) == 0: df_test = df_cv
    metrics = ["MAE", "RMSE", "MAPE", "PICP_50", "PICP_80", "MPIW_50", "MPIW_80", 
               "Winkler_50", "Winkler_80", "mae_avg", "robustness_score"]
    agg = {}
    for col in metrics:
        if col in df_test.columns:
            agg[f"{col}_mean"] = float(df_test[col].mean())
            agg[f"{col}_std"] = float(df_test[col].std())
    return agg

def load_and_merge():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_a = pd.read_csv(os.path.join(DIR_A, "ablation_results_summary.csv"))
    print(f"[目录A] 加载 {len(df_a)} 个变体: {df_a['variant'].tolist()}")
    
    df_no_fitting_cv = pd.read_csv(os.path.join(DIR_B, "no_fitting/cv_results/cv_summary_report.csv"))
    print(f"[目录B] 加载 no_fitting CV数据: {len(df_no_fitting_cv)} 折")
    
    no_fitting_agg = _aggregate_metrics(df_no_fitting_cv)
    no_fitting_agg["variant"] = "no_fitting"
    no_fitting_agg["display_name"] = VARIANT_DISPLAY["no_fitting"]
    
    df_merged = pd.concat([df_a, pd.DataFrame([no_fitting_agg])], ignore_index=True)
    order = ["base", "no_time2vec", "no_fitting", "no_revin"]
    df_merged["_s"] = df_merged["variant"].apply(lambda x: order.index(x) if x in order else 99)
    df_merged = df_merged.sort_values("_s").drop(columns=["_s"])
    
    merged_path = os.path.join(OUTPUT_DIR, "ablation_results_merged.csv")
    df_merged.to_csv(merged_path, index=False)
    print(f"\n[汇总] 已保存: {merged_path}")
    return df_merged

def generate_charts(df):
    charts_dir = os.path.join(OUTPUT_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    metrics = [("MAE_mean", "MAE_std", "MAE", "lower"), ("RMSE_mean", "RMSE_std", "RMSE", "lower"),
               ("PICP_50_mean", "PICP_50_std", "PICP@50%", "higher"), 
               ("PICP_80_mean", "PICP_80_std", "PICP@80%", "higher"),
               ("robustness_score_mean", "robustness_score_std", "Robustness", "higher")]
    
    for metric, std_col, title, better in metrics:
        if metric not in df.columns: continue
        fig, ax = plt.subplots(figsize=(10, 6))
        variants = df["variant"].tolist()
        labels = [VARIANT_DISPLAY.get(v, v) for v in variants]
        colors = [VARIANT_COLORS.get(v, "#4C72B0") for v in variants]
        values = df[metric].tolist()
        yerr = df[std_col].tolist() if std_col in df.columns else None
        
        bars = ax.bar(labels, values, color=colors, yerr=yerr, capsize=5, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.002, f"{val:.4f}", 
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        best_idx = values.index(min(values) if better=="lower" else max(values))
        bars[best_idx].set_edgecolor("#2ca02c")
        bars[best_idx].set_linewidth(3)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(title); ax.set_xlabel("Model Variant"); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"ablation_{metric.replace('_mean','')}.png"), dpi=150)
        plt.close()
        print(f"  [图表] {title}")
    
    # 综合对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, (metric, std_col, title, better) in enumerate(metrics):
        if i >= 5 or metric not in df.columns: continue
        ax = axes[i]
        variants = df["variant"].tolist()
        labels = [VARIANT_DISPLAY.get(v, v) for v in variants]
        colors = [VARIANT_COLORS.get(v, "#4C72B0") for v in variants]
        values = df[metric].tolist()
        yerr = df[std_col].tolist() if std_col in df.columns else None
        bars = ax.bar(labels, values, color=colors, yerr=yerr, capsize=3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis="y", alpha=0.3)
    axes[5].axis("off")
    plt.suptitle("消融实验综合对比", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "ablation_overview.png"), dpi=150)
    plt.close()
    print(f"  [图表] 综合对比图")

def print_report(df):
    print("\n" + "="*80)
    print("消融实验结果汇总报告")
    print("="*80)
    print(f"\n{'变体':<15} {'MAE':<20} {'RMSE':<20} {'PICP@50%':<12} {'PICP@80%':<12} {'鲁棒性':<10}")
    print("-"*80)
    for _, r in df.iterrows():
        name = VARIANT_DISPLAY.get(r["variant"], r["variant"])
        mae = f"{r['MAE_mean']:.4f}±{r.get('MAE_std',0):.4f}"
        rmse = f"{r['RMSE_mean']:.4f}±{r.get('RMSE_std',0):.4f}"
        picp50 = f"{r['PICP_50_mean']:.2f}%"
        picp80 = f"{r['PICP_80_mean']:.2f}%"
        robust = f"{r['robustness_score_mean']:.4f}"
        print(f"{name:<15} {mae:<20} {rmse:<20} {picp50:<12} {picp80:<12} {robust:<10}")
    
    print("\n" + "="*80)
    print("分析结论")
    print("="*80)
    base = df[df["variant"]=="base"].iloc[0]
    no_fit = df[df["variant"]=="no_fitting"].iloc[0]
    mae_diff = (no_fit["MAE_mean"] - base["MAE_mean"]) / base["MAE_mean"] * 100
    rmse_diff = (no_fit["RMSE_mean"] - base["RMSE_mean"]) / base["RMSE_mean"] * 100
    
    print(f"\n1. no_fitting vs Full Model:")
    print(f"   - MAE: {no_fit['MAE_mean']:.4f} vs {base['MAE_mean']:.4f} ({mae_diff:+.2f}%)")
    print(f"   - RMSE: {no_fit['RMSE_mean']:.4f} vs {base['RMSE_mean']:.4f} ({rmse_diff:+.2f}%)")
    print(f"   - PICP@80%: {no_fit['PICP_80_mean']:.2f}% vs {base['PICP_80_mean']:.2f}%")
    print(f"   - 鲁棒性: {no_fit['robustness_score_mean']:.4f} vs {base['robustness_score_mean']:.4f}")
    
    best_mae = df.loc[df["MAE_mean"].idxmin()]
    best_robust = df.loc[df["robustness_score_mean"].idxmax()]
    print(f"\n2. 最佳MAE: {VARIANT_DISPLAY.get(best_mae['variant'], best_mae['variant'])} ({best_mae['MAE_mean']:.4f})")
    print(f"3. 最佳鲁棒性: {VARIANT_DISPLAY.get(best_robust['variant'], best_robust['variant'])} ({best_robust['robustness_score_mean']:.4f})")

if __name__ == "__main__":
    df = load_and_merge()
    generate_charts(df)
    print_report(df)
    print(f"\n输出目录: {OUTPUT_DIR}")
