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
消融实验脚本 — 验证各组件对 BTP 预测性能的贡献

变体定义:
  base         → 没有CQR的主模型（所有其他组件开启）
  no_revin     → 没有CQR和RevIN
  no_fitting   → 没有CQR和拟合模块
  full         → 完整模型（含CQR偏差校准）
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import os
import copy
import json
import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd

from btp.config import TrainConfig, make_timestamp


# ── 变体定义 ──
VARIANT_DEFINITIONS: Dict[str, Dict] = {
    'base': {
        # 基准模型：RevIN + Fitting + Time2Vec（完整模型）
        'enable_revin': True,
        'enable_fitting_module': True,
        'enable_time2vec': True,
        'enable_online_cqr': False,
        'enable_cv_cqr': False,
    },
    'no_revin': {
        # 消融 RevIN
        'enable_revin': False,
        'enable_fitting_module': True,
        'enable_time2vec': True,
        'enable_online_cqr': False,
        'enable_cv_cqr': False,
    },
    'no_fitting': {
        # 消融拟合模块
        'enable_revin': True,
        'enable_fitting_module': False,
        'enable_time2vec': True,
        'enable_online_cqr': False,
        'enable_cv_cqr': False,
    },
    'no_time2vec': {
        # 消融 Time2Vec（使用固定正弦位置编码）
        'enable_revin': True,
        'enable_fitting_module': True,
        'enable_time2vec': False,
        'enable_online_cqr': False,
        'enable_cv_cqr': False,
    },
}

# 变体显示名称（用于图表）
VARIANT_DISPLAY_NAMES: Dict[str, str] = {
    'base': "Full Model",
    'no_revin': "w/o RevIN",
    'no_fitting': "w/o Fitting",
    'no_time2vec': "w/o Time2Vec",
}


def build_ablation_variants(base_config: TrainConfig) -> Dict[str, TrainConfig]:
    """根据 config.ablation_variants 列表构建消融变体配置"""
    variants = {}

    for name in base_config.ablation_variants:
        if name not in VARIANT_DEFINITIONS:
            logging.warning(f"[Ablation] 未知变体 '{name}'，跳过")
            continue

        cfg = copy.deepcopy(base_config)
        cfg.enable_ablation_study = False   # 防止递归
        cfg.enable_model_comparison = False
        cfg.enable_model_diagnostics = False
        cfg.exp_name = f"Ablation_{name}"

        for attr, val in VARIANT_DEFINITIONS[name].items():
            setattr(cfg, attr, val)

        variants[name] = cfg

    return variants


def _load_cv_summary(output_dir: str) -> pd.DataFrame:
    """从变体运行目录中加载 CV 汇总指标"""
    summary_path = os.path.join(output_dir, "cv_results", "cv_summary_report.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"未找到 CV 汇总: {summary_path}")
    return pd.read_csv(summary_path)


def _aggregate_metrics(df_cv: pd.DataFrame) -> Dict:
    """从 CV 汇总表中聚合关键指标（均值 ± 标准差）"""
    # 优先使用 test split
    if "split" in df_cv.columns:
        df_test = df_cv[df_cv["split"] == "test"].copy()
        if len(df_test) == 0:
            df_test = df_cv.copy()
    else:
        df_test = df_cv.copy()

    metrics_cols = [
        "MAE", "RMSE", "MAPE",
        "PICP_50", "PICP_80",
        "MPIW_50", "MPIW_80",
        "Winkler_50", "Winkler_80",
        "mae_avg", "robustness_score",
    ]
    agg = {}
    for col in metrics_cols:
        if col in df_test.columns:
            agg[f"{col}_mean"] = float(df_test[col].mean())
            agg[f"{col}_std"] = float(df_test[col].std())
    return agg


def run_ablation_experiment(base_config: TrainConfig):
    """
    消融实验主入口

    1. 构建变体配置
    2. 逐个运行完整流水线
    3. 汇总指标并生成对比图表
    """
    from main import run_full_pipeline

    timestamp = make_timestamp()
    exp_root = os.path.join("data", f"ablation_{timestamp}")
    os.makedirs(exp_root, exist_ok=True)

    variants = build_ablation_variants(base_config)

    print("\n" + "=" * 70)
    print(f"[Ablation] 启动消融实验 ({len(variants)} 个变体)")
    print(f"输出目录: {exp_root}")
    print(f"变体列表: {list(variants.keys())}")
    print("=" * 70 + "\n")

    results: List[Dict] = []

    for idx, (name, cfg) in enumerate(variants.items(), 1):
        display = VARIANT_DISPLAY_NAMES.get(name, name)
        print(f"\n>>> [{idx}/{len(variants)}] 运行变体: {name} ({display})")

        # 设置独立输出目录
        cfg.output_dir = os.path.join(exp_root, name)

        try:
            run_full_pipeline(cfg)
            print(f"    [Success] {name} 运行成功")

            # 加载 CV 汇总指标
            df_cv = _load_cv_summary(cfg.output_dir)
            agg = _aggregate_metrics(df_cv)

            results.append({
                "variant": name,
                "display_name": display,
                **agg,
            })

        except Exception as e:
            print(f"    [Failed] {name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "variant": name,
                "display_name": display,
                "error": str(e),
            })
            continue

    # ── 汇总结果 ──
    df_results = pd.DataFrame(results)
    summary_csv = os.path.join(exp_root, "ablation_results_summary.csv")
    df_results.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    summary_xlsx = os.path.join(exp_root, "ablation_results_summary.xlsx")
    try:
        df_results.to_excel(summary_xlsx, index=False)
    except Exception:
        pass  # openpyxl 可能未安装

    # ── 生成对比图表 ──
    try:
        from analyze_ablation_experiment import analyze_ablation
        analyze_ablation(exp_root)
    except Exception as e:
        print(f"[Warning] 分析图表生成失败: {e}")

    print("\n" + "=" * 70)
    print("[Ablation] 消融实验完成！")
    print(f"汇总表: {summary_csv}")
    print(f"输出目录: {exp_root}")
    print("=" * 70 + "\n")

    return exp_root, summary_csv


if __name__ == "__main__":
    # 独立运行入口
    cfg = TrainConfig(
        excel_path="data/raw/20130503ycz.xlsx",
        target_column="北侧_计算BTP位置",
        enable_ablation_study=True,
    )
    run_ablation_experiment(cfg)
