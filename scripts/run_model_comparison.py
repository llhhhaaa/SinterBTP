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
import json
import copy
import time
import logging
import argparse
from typing import Dict, List

import pandas as pd

from btp.config import TrainConfig, make_timestamp
from scripts.main import run_full_pipeline


MODELS_TO_COMPARE = [
    "enhanced_transformer",
    "baseline_gru",
    "baseline_lstm",
    "baseline_transformer",
]


def _load_cv_summary(output_dir: str) -> pd.DataFrame:
    summary_path = os.path.join(output_dir, "cv_results", "cv_summary_report.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"未找到CV汇总: {summary_path}")
    return pd.read_csv(summary_path)


def _aggregate_metrics(df_cv: pd.DataFrame) -> Dict[str, float]:
    df_test = df_cv[df_cv.get("split", "test") == "test"].copy()
    if len(df_test) == 0:
        df_test = df_cv.copy()

    metrics_cols = [
        "MAE",
        "RMSE",
        "PICP_80",
        "PICP_50",
        "MPIW_80",
        "MPIW_50",
        "Winkler_80",
        "Winkler_50",
        "mae_avg",
        "mae_q50",
        "coverage_80",
        "coverage_50",
        "interval_width_mean",
        "robustness_score",
    ]
    agg = {}
    for col in metrics_cols:
        if col in df_test.columns:
            agg[f"{col}_mean"] = float(df_test[col].mean())
            agg[f"{col}_std"] = float(df_test[col].std())
    return agg


def _build_base_config() -> TrainConfig:
    cfg = TrainConfig(
        excel_path="20130503ycz.xlsx",
        target_column="北侧_计算BTP位置",
    )
    cfg.PREGENERATE_ONLY = False  # 确保运行完整流程
    cfg.enable_ablation_study = False
    cfg.enable_model_comparison = False
    cfg.enable_model_diagnostics = False
    cfg.enable_cv_cqr = False
    cfg.enable_online_cqr = False
    return cfg


def run_model_comparison(
    quick_run: bool = True,
    cv_splits: int = 2,
    epochs: int = 3,
    output_root: str = "实验记录/Model_comparison",
    models: List[str] = None,
    raw_seq_len: int = None,
):
    models = models or MODELS_TO_COMPARE
    timestamp = make_timestamp()
    run_tag = "quick" if quick_run else "full"
    exp_root = os.path.join(output_root, f"ModelComparison_{run_tag}_{timestamp}")
    os.makedirs(exp_root, exist_ok=True)

    base_config = _build_base_config()
    if raw_seq_len is not None:
        base_config.raw_seq_len = int(raw_seq_len)

    if quick_run:
        base_config.cv_n_splits = cv_splits
        base_config.epochs = epochs
        base_config.patience = max(1, min(2, epochs))
        base_config.batch_size = min(128, base_config.batch_size)
        base_config.mc_samples = 1
    else:
        base_config.cv_n_splits = max(5, base_config.cv_n_splits)
        base_config.epochs = max(10, base_config.epochs)
        if base_config.mc_samples <= 1:
            base_config.mc_samples = 10

    results = []

    for idx, model_name in enumerate(models, 1):
        logging.info("=" * 80)
        logging.info("[%s/%s] 运行模型: %s", idx, len(models), model_name)

        cfg = copy.deepcopy(base_config)
        cfg.model_type = model_name
        cfg.output_dir = os.path.join(exp_root, f"Compare_{model_name}")
        cfg.exp_name = f"Compare_{model_name}"

        start_time = time.time()
        run_full_pipeline(cfg)
        elapsed = time.time() - start_time

        df_cv = _load_cv_summary(cfg.output_dir)
        agg = _aggregate_metrics(df_cv)

        results.append(
            {
                "model": model_name,
                "elapsed_sec": round(elapsed, 2),
                "config": json.dumps(
                    {
                        "cv_splits": cfg.cv_n_splits,
                        "epochs": cfg.epochs,
                        "seed": cfg.seed,
                        "mc_samples": cfg.mc_samples,
                        "raw_seq_len": cfg.raw_seq_len,
                        "forecast_steps": cfg.forecast_steps,
                        "prediction_offset": cfg.prediction_offset,
                    },
                    ensure_ascii=False,
                ),
                **agg,
            }
        )

    df_results = pd.DataFrame(results)
    results_csv = os.path.join(exp_root, "model_comparison_summary.csv")
    df_results.to_csv(results_csv, index=False, encoding="utf-8-sig")

    results_xlsx = os.path.join(exp_root, "model_comparison_summary.xlsx")
    df_results.to_excel(results_xlsx, index=False)

    print("=" * 80)
    print("模型对比实验完成")
    print(f"输出目录: {exp_root}")
    print(f"汇总表: {results_csv}")
    print("=" * 80)

    return exp_root, results_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="模型对比实验")
    parser.add_argument("--full", action="store_true", help="运行完整对比实验(5折、10 epochs)")
    parser.add_argument("--cv-splits", type=int, default=2, help="快速实验CV折数")
    parser.add_argument("--epochs", type=int, default=3, help="快速实验训练轮数")
    parser.add_argument("--seq-len", type=int, default=360, help="输入序列长度(raw_seq_len)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.full:
        run_model_comparison(quick_run=False, raw_seq_len=args.seq_len)
    else:
        run_model_comparison(quick_run=True, cv_splits=args.cv_splits, epochs=args.epochs, raw_seq_len=args.seq_len)
