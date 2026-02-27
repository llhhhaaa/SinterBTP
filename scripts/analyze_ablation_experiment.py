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
消融实验分析脚本 — 生成对比图表和汇总表格

适配 4 个变体:
  base         → "Base (w/o CQR)"
  no_revin     → "w/o RevIN"
  no_fitting   → "w/o Fitting"
  full         → "Full (w/ CQR)"
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import os
import json
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 变体显示名称映射
VARIANT_DISPLAY = {
    "base": "Base (w/o CQR)",
    "no_revin": "w/o RevIN",
    "no_fitting": "w/o Fitting",
    "full": "Full (w/ CQR)",
}

# 柱状图配色
VARIANT_COLORS = {
    "base": "#4C72B0",
    "no_revin": "#DD8452",
    "no_fitting": "#C44E52",
    "full": "#8172B3",
}


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_summary(exp_root: str) -> pd.DataFrame:
    summary_path = os.path.join(exp_root, "ablation_results_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"未找到汇总文件: {summary_path}")
    df = pd.read_csv(summary_path)
    return df


def _add_display_names(df: pd.DataFrame) -> pd.DataFrame:
    """为 variant 列添加显示名称"""
    df = df.copy()
    if "display_name" not in df.columns:
        df["display_name"] = df["variant"].map(
            lambda v: VARIANT_DISPLAY.get(v, v)
        )
    return df


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    """提取论文表格所需的关键指标列"""
    cols = [
        "variant",
        "display_name",
        "MAE_mean", "MAE_std",
        "RMSE_mean", "RMSE_std",
        "PICP_50_mean",
        "PICP_80_mean",
        "MPIW_50_mean",
        "MPIW_80_mean",
        "Winkler_50_mean",
        "Winkler_80_mean",
        "robustness_score_mean",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


def _plot_bar(df: pd.DataFrame, metric: str, save_path: str, title: str):
    """绘制单指标柱状图对比"""
    if metric not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    variants = df["variant"].tolist()
    labels = [VARIANT_DISPLAY.get(v, v) for v in variants]
    colors = [VARIANT_COLORS.get(v, "#4C72B0") for v in variants]
    values = df[metric].tolist()

    # 如果有 std 列，绘制误差棒
    std_col = metric.replace("_mean", "_std")
    yerr = df[std_col].tolist() if std_col in df.columns else None

    bars = ax.bar(labels, values, color=colors, yerr=yerr,
                  capsize=4, edgecolor="white", linewidth=0.8)

    # 在柱子上方标注数值
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(metric.replace("_mean", ""))
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_grouped_bar(df: pd.DataFrame, metrics: list, save_path: str, title: str):
    """绘制多指标分组柱状图"""
    existing = [m for m in metrics if m in df.columns]
    if not existing:
        return

    variants = df["variant"].tolist()
    labels = [VARIANT_DISPLAY.get(v, v) for v in variants]

    import numpy as np
    x = np.arange(len(labels))
    width = 0.8 / len(existing)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(existing):
        offset = (i - len(existing) / 2 + 0.5) * width
        vals = df[metric].tolist()
        ax.bar(x + offset, vals, width, label=metric.replace("_mean", ""))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_cqr_comparison(df: pd.DataFrame, output_dir: str):
    """CQR 前后对比图（base vs full）"""
    cqr_df = df[df["variant"].isin(["base", "full"])].copy()
    if len(cqr_df) < 2:
        return

    cqr_df = cqr_df.set_index("variant")
    for metric in ["MAE_mean", "RMSE_mean", "PICP_80_mean", "MPIW_80_mean"]:
        if metric not in cqr_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = [VARIANT_DISPLAY.get(v, v) for v in cqr_df.index]
        colors = ["#DD8452", "#55A868"]
        ax.bar(labels, cqr_df[metric].values, color=colors)
        ax.set_ylabel(metric.replace("_mean", ""))
        ax.set_title(f"CQR 对比 — {metric.replace('_mean', '')}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"cqr_compare_{metric}.png"), dpi=200)
        plt.close(fig)


def analyze_ablation(exp_root: str, output_dir: Optional[str] = None):
    """消融实验分析主入口"""
    df = _load_summary(exp_root)
    df = _add_display_names(df)

    if output_dir is None:
        output_dir = os.path.join(exp_root, "analysis")
    _ensure_dir(output_dir)

    # 1. 格式化表格并保存
    table = _format_table(df)
    table.to_csv(os.path.join(output_dir, "ablation_table.csv"),
                 index=False, encoding="utf-8-sig")
    try:
        table.to_excel(os.path.join(output_dir, "ablation_table.xlsx"), index=False)
    except Exception:
        pass

    # 2. 单指标柱状图
    bar_metrics = [
        ("MAE_mean", "MAE 对比"),
        ("RMSE_mean", "RMSE 对比"),
        ("PICP_80_mean", "PICP_80 覆盖率对比"),
        ("MPIW_80_mean", "MPIW_80 区间宽度对比"),
        ("robustness_score_mean", "鲁棒性得分对比"),
    ]
    for metric, title in bar_metrics:
        save_path = os.path.join(output_dir, f"bar_{metric}.png")
        _plot_bar(df, metric, save_path, title)

    # 3. 分组柱状图
    _plot_grouped_bar(
        df,
        ["MAE_mean", "RMSE_mean"],
        os.path.join(output_dir, "grouped_mae_rmse.png"),
        "MAE & RMSE 对比",
    )
    _plot_grouped_bar(
        df,
        ["PICP_80_mean", "MPIW_80_mean"],
        os.path.join(output_dir, "grouped_picp_mpiw.png"),
        "PICP_80 & MPIW_80 对比",
    )

    # 4. CQR 前后对比
    _plot_cqr_comparison(df, output_dir)

    # 5. 元信息
    meta_path = os.path.join(output_dir, "analysis_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"exp_root": exp_root, "variants": df["variant"].tolist()},
                  f, ensure_ascii=False, indent=2)

    print(f"[Ablation Analysis] 分析完成，输出目录: {output_dir}")
    return output_dir


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = "data/ablation_latest"
    analyze_ablation(root)
