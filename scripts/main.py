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

import matplotlib
matplotlib.use('Agg')  
import logging
import os
import copy
import json
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Any
from btp.calibrator import OnlineGradientCalibrator
from btp.model_diagnostics import ModelDiagnostics  
from btp.config import TrainConfig, make_timestamp
from btp.data_loader import DataLoader as CustomDataLoader
from btp.preprocessor import DataPreprocessor
from btp.model import build_model
from btp.trainer import Trainer
from btp.metrics import evaluate_quantile_regression
from btp.health_model import HealthModel
from btp.visualizer import Visualizer
from btp.utils import setup_logging

def restore_absolute_values(
    y_scaled: np.ndarray,
    anchor: np.ndarray,
    scaler_y,
    enable_delta: bool
) -> np.ndarray:
    """
    将模型输出还原为绝对物理值。
    """
    y_scaled = np.asarray(y_scaled)

    # 1) 逆标准化
    if y_scaled.ndim == 3:
        B, S, Q = y_scaled.shape
        mean = np.asarray(scaler_y.mean_, dtype=float)
        scale = np.asarray(scaler_y.scale_, dtype=float)
        if mean.size == 1:
            mean = mean.reshape(1, 1, 1)
            scale = scale.reshape(1, 1, 1)
        elif mean.size == S:
            mean = mean.reshape(1, S, 1)
            scale = scale.reshape(1, S, 1)
        y_phys = y_scaled * scale + mean
    elif y_scaled.ndim == 2:
        B, Q = y_scaled.shape
        mean = np.asarray(scaler_y.mean_, dtype=float)
        scale = np.asarray(scaler_y.scale_, dtype=float)
        if mean.size == 1:
            mean = mean.reshape(1, 1)
            scale = scale.reshape(1, 1)
        y_phys = y_scaled * scale + mean
    else:
        raise ValueError(f"Unsupported y_scaled ndim={y_scaled.ndim}")

    # 2) 加上 Delta Anchor
    if enable_delta:
        if anchor is None: return y_phys
        if hasattr(anchor, "values"): anchor = anchor.values
        anchor = np.asarray(anchor)

        if y_phys.ndim == 3:
            if anchor.ndim == 1: anchor = anchor[:, None, None]
            elif anchor.ndim == 2: anchor = anchor[:, :, None]
            return y_phys + anchor
        if anchor.ndim == 1: anchor = anchor[:, None]
        return y_phys + anchor

    return y_phys


def run_cqr_simulation(
    y_pred_abs: np.ndarray,
    y_true_abs: np.ndarray,
    config: TrainConfig,
    sampling_sec: float,
    tag: str,
    fold_dir: str
):
    """ 执行在线校准：偏差修正 + 区间宽度自适应 """
    if not config.enable_online_cqr:
        return y_pred_abs, np.zeros((len(y_pred_abs), 2)), {}

    logging.info(f"--- [{tag}] 执行在线校准 (Bias + Width) ---")
    # ── 因果延迟计算（防数据泄漏） ──
    #
    # 数据流说明：
    #   preprocessor._internal_parallel_build 用 arange(start, end) 遍历每个原始行索引 i，
    #   对每个 i 构建一个样本，其预测目标在 i + forecast_steps × w_rows 处。
    #   因此相邻输出样本间距 ≈ 1 个原始行。
    #
    # 因果约束：
    #   在校准步 t，要使用步 (t - delay) 的反馈，需要该步的真值已可观测：
    #     t_raw >= (t_raw - delay) + forecast_steps
    #   即 delay >= forecast_steps
    #
    # 不再使用粒化窗口，delay 直接等于 forecast_steps
    #
    f_steps = int(getattr(config, "forecast_steps", 5))
    prediction_offset = int(getattr(config, "prediction_offset", 0))
    # [修复] CQR 因果性：延迟至少覆盖 prediction_offset + forecast_steps
    min_delay = prediction_offset + f_steps
    delay_steps = max(min_delay, getattr(config, "delay_steps", 0))
    if delay_steps < min_delay:
        raise ValueError(
            f"[Calibrator] delay_steps({delay_steps}) < prediction_offset + forecast_steps({min_delay})，存在数据泄漏风险"
        )
    logging.info(
        f"[Calibrator] delay_steps={delay_steps} (prediction_offset={prediction_offset}, forecast_steps={f_steps})"
    )
    
    calibrator = OnlineGradientCalibrator(
        bias_window=getattr(config, "cqr_bias_window", 60),
        min_samples=getattr(config, "cqr_min_samples", 20),
    )

    y_calib, corrections, diag_list = calibrator.apply(
        y_pred_abs, 
        y_true_abs, 
        delay_steps=delay_steps
    )
    
    final_diag = diag_list[-1] if diag_list else {}
    if fold_dir:
        os.makedirs(fold_dir, exist_ok=True)
        pd.DataFrame(diag_list).to_csv(os.path.join(fold_dir, "cqr_process_log.csv"), index=False)

    return y_calib, corrections, final_diag

def run_full_pipeline(config: TrainConfig):
    # ========== 0. 环境与日志设置 ==========
    if not config.output_dir:
        timestamp = make_timestamp()
        config.output_dir = os.path.join("outputs", f"run_{timestamp}")

    os.makedirs(config.output_dir, exist_ok=True)
    log_path = os.path.join(config.output_dir, "training.log")
    setup_logging(log_path, reset_handlers=True)

    config.save_json(os.path.join(config.output_dir, "config.json"))
    # [修复] 统一设置随机种子，保证可复现
    import random
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("=" * 80)
    logging.info("  🚀 BTP 预测系统 (原始高频序列)")
    logging.info("=" * 80)
    logging.info(f"[Config] Device: {device} | Model: {config.model_type} | SeqLen: {config.raw_seq_len}")

    # ========== 1. 数据加载与预处理 ==========
    data_loader = CustomDataLoader()
    df_raw, sampling_sec = data_loader.load_xlsx(config.excel_path, prefer_time_col="时间")

    preprocessor = DataPreprocessor(config)
    df_feat = preprocessor.build_features(df_raw)

    n_total = len(df_feat)
    n_test_holdout = int(n_total * config.test_split)
    df_cv_pool = df_feat.iloc[:-n_test_holdout].copy() if n_test_holdout > 0 else df_feat.copy()
    df_test_final = df_feat.iloc[-n_test_holdout:].copy() if n_test_holdout > 0 else None

    # 检查是否仅运行预生成逻辑 (对于 main 流程)
    if not config.enable_cv and getattr(config, "PREGENERATE_ONLY", False):
        logging.info("[Mode] PREGENERATE_ONLY 为真，正在预生成并缓存数据...")
        # 触发预处理和缓存
        _ = preprocessor.process_and_split(df_feat, sampling_sec)
        logging.info("[Mode] 数据已生成并缓存，程序退出。")
        return

    # [修复] 预处理器保存应在 process_and_split 之后，确保 scaler 已拟合

    # ==========================================
    # 🔄 Phase 1: 滚动交叉验证 (Rolling CV)
    # ==========================================
    cv_metrics_list: List[Dict] = []
    
    if config.enable_cv:
        cv_root_dir = os.path.join(config.output_dir, "cv_results")
        os.makedirs(cv_root_dir, exist_ok=True)
        
        # 检查是否仅运行预生成逻辑 (对于 CV 流程)
        if getattr(config, "PREGENERATE_ONLY", False):
            logging.info("[Mode] PREGENERATE_ONLY 为真，正在预生成并缓存所有 CV Folds 数据...")
            # 触发 yield_rolling_folds 以生成缓存
            for _ in preprocessor.yield_rolling_folds(df_cv_pool, sampling_sec):
                pass
            logging.info("[Mode] 所有 CV Folds 数据已生成并缓存，程序退出。")
            return

        # 使用时序版交叉验证（带独立 test 集）
        for fold_idx, data_dict_full in preprocessor.yield_rolling_folds(df_cv_pool, sampling_sec):
            fold_name = f"Fold_{fold_idx + 1}"
            logging.info(f"\n>>> 🟢 [CV] Processing {fold_name} (Train→Val→Test 时序划分)...")
            
            # [单一任务模式]
            task_dir = os.path.join(cv_root_dir, fold_name, "Standard")
            os.makedirs(task_dir, exist_ok=True)
            
            cfg_fold = copy.deepcopy(config)
            m_type = getattr(cfg_fold, "model_type", "enhanced_transformer")
            data_dict_task = data_dict_full.copy()
            
            # [数据分发] - 统一使用原始序列 (粒化功能已移除)
            data_dict_task["X_tr"] = data_dict_full["X_tr_raw"]
            data_dict_task["X_val"] = data_dict_full["X_val_raw"]
            data_dict_task["X_test"] = data_dict_full["X_test_raw"]
            input_dim = data_dict_full["raw_feat_dim"]
            
            # --- 构建模型 ---
            model = build_model(
                cfg_fold,
                input_dim=input_dim,
                model_type=m_type
            )
            
            # --- 训练（使用 train 数据，val 用于 early stopping）---
            trainer = Trainer(model, cfg_fold, device)
            trainer.train(data_dict_task, verbose=True)
            
            # --- 在 TEST 集上评估（最终结果）---
            logging.info(f"  🔮 [Predict] 在独立 TEST 集上评估...")
            X_test_tensor = torch.FloatTensor(data_dict_task["X_test"]).to(device)
            
            # MC Dropout 多次采样平均
            mc_samples = getattr(cfg_fold, 'mc_samples', 10)
            logging.info(f"  🔮 [MC Dropout] 使用 {mc_samples} 次采样平均...")
            preds_test_scaled = trainer.predict(X_test_tensor, mc_samples=mc_samples)
            logging.info(f"  🔮 [MC Dropout] 采样完成")
            
            y_pred_scaled = preds_test_scaled

            y_pred_abs_full = restore_absolute_values(
                y_pred_scaled, data_dict_full["anchor_test"],
                preprocessor.scaler_y, cfg_fold.enable_delta_forecast
            )
            
            # 取最后一步 T+5
            y_pred_abs_last = y_pred_abs_full[:, -1, :]
            anc = data_dict_full["anchor_test"].reshape(-1, 1) if cfg_fold.enable_delta_forecast else 0
            y_true_abs_full = data_dict_full["y_test_raw"] + anc
            y_true_abs_last = y_true_abs_full[:, -1]
            y_true_abs_tiled = np.tile(y_true_abs_last.reshape(-1, 1), (1, 5))

            # 计算 MAE (Raw — 无校准) - 使用 TEST 集结果
            step_maes = [np.mean(np.abs(y_pred_abs_full[:, s, 2] - y_true_abs_full[:, s])) for s in range(y_pred_abs_full.shape[1])]
            metrics_raw = evaluate_quantile_regression(y_true_abs_tiled, y_pred_abs_last, compute_width_stats=True)
            metrics_raw.update({'mae_avg': np.mean(step_maes), 'fold': fold_idx + 1, 'cqr': 'off', 'split': 'test'})

            # 健康度分析 (Raw)
            health_model = HealthModel(cfg_fold)
            h_res = health_model.analyze(y_pred_abs_last, y_true=y_true_abs_tiled)
            metrics_raw.update({f"health_{k}": v for k, v in h_res.items() if isinstance(v, (float, int))})

            cv_metrics_list.append(metrics_raw)

            # ── CQR 消融：在同一 fold 上跑校准版本 ──
            # [基线模型] 跳过 CQR 校准
            is_baseline = m_type.startswith("baseline_")
            if getattr(cfg_fold, "enable_cv_cqr", False) and not is_baseline:
                cqr_dir = os.path.join(task_dir, "cqr_ablation")
                y_val_calib, val_corr, cqr_diag = run_cqr_simulation(
                    y_pred_abs_last, y_true_abs_tiled, cfg_fold, sampling_sec,
                    f"CV_F{fold_idx+1}", cqr_dir
                )
                metrics_cqr = evaluate_quantile_regression(y_true_abs_tiled, y_val_calib, compute_width_stats=True)
                step_maes_cqr = [np.mean(np.abs(y_pred_abs_full[:, s, 2] - y_true_abs_full[:, s])) for s in range(y_pred_abs_full.shape[1])]
                metrics_cqr.update({'mae_avg': np.mean(step_maes_cqr), 'fold': fold_idx + 1, 'cqr': 'on'})
                # CQR 校准后的 Q50 MAE
                metrics_cqr['mae_q50_calib'] = float(np.mean(np.abs(y_val_calib[:, 2] - y_true_abs_tiled[:, 2])))
                h_res_cqr = health_model.analyze(y_val_calib, y_true=y_true_abs_tiled)
                metrics_cqr.update({f"health_{k}": v for k, v in h_res_cqr.items() if isinstance(v, (float, int))})
                cv_metrics_list.append(metrics_cqr)
                logging.info(f"  [CQR 消融] Fold {fold_idx+1}: "
                             f"Raw MAE={metrics_raw.get('mae_q50', 0):.4f} → CQR MAE={metrics_cqr['mae_q50_calib']:.4f}")

                # CQR 消融可视化
                vis_cqr_dir = os.path.join(task_dir, "visualizations_cqr")
                vis_cqr = Visualizer(vis_cqr_dir, cfg_fold)
                vis_cqr.generate_all_plots(y_val_calib, y_true_abs_tiled, cfg_fold,
                                           y_raw=y_pred_abs_last, corrections=val_corr,
                                           health_res=h_res_cqr, prefix="cqr")

            # 可视化 (Raw — 无校准)
            vis_dir = os.path.join(task_dir, "visualizations")
            visualizer = Visualizer(vis_dir, cfg_fold)
            visualizer.generate_all_plots(y_pred_abs_last, y_true_abs_tiled, cfg_fold, y_raw=y_pred_abs_last,
                                        corrections=np.zeros_like(y_pred_abs_last[:,:2]), health_res=h_res)
            
            # 保存残差
            diag_dir = os.path.join(task_dir, "diagnostics")
            os.makedirs(diag_dir, exist_ok=True)
            val_timestamps = df_cv_pool["时间"].iloc[-len(y_pred_abs_last):].values
            
            df_resid = pd.DataFrame({
                'Timestamp': val_timestamps,
                'True_Value': y_true_abs_tiled[:, 2], 
                'Pred_Value': y_pred_abs_last[:, 2],  
                'Residual': y_true_abs_tiled[:, 2] - y_pred_abs_last[:, 2]
            })
            df_resid.to_csv(os.path.join(diag_dir, "residual_analysis.csv"), index=False)

            # 诊断与鲁棒性
            diagnoser = ModelDiagnostics(diag_dir, cfg_fold)
            robust_df = diagnoser.perform_robustness_test(
                model=model,
                X_test=X_test_tensor,
                y_true_abs=y_true_abs_tiled,
                trainer=trainer,
                preprocessor=preprocessor,
                anchor=data_dict_task.get("anchor_val"),
                enable_delta=cfg_fold.enable_delta_forecast
            )
            metrics_raw['robustness_score'] = robust_df["性能保持率"].iloc[-1] 

        # 汇总 CV 结果（只报告 TEST 集结果）
        summary_path = os.path.join(cv_root_dir, "cv_summary_report.csv")
        df_cv_summary = pd.DataFrame(cv_metrics_list)
        df_cv_summary.to_csv(summary_path, index=False)
        
        # 计算 TEST 集结果的均值±标准差
        test_metrics = df_cv_summary[df_cv_summary['split'] == 'test'] if 'split' in df_cv_summary.columns else df_cv_summary
        if len(test_metrics) > 0:
            logging.info("\n" + "=" * 60)
            logging.info("  📊 [CV 汇总] TEST 集结果 (均值 ± 标准差)")
            logging.info("=" * 60)
            for col in ['mae_avg', 'mae_q50', 'coverage_80', 'coverage_50', 'interval_width_mean']:
                if col in test_metrics.columns:
                    mean_val = test_metrics[col].mean()
                    std_val = test_metrics[col].std()
                    logging.info(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")
            logging.info("=" * 60)

    # ==========================================
    # 🧪 Phase 2: 最终全量测试 (Refit)
    # ==========================================
    if df_test_final is not None:
        logging.info("\n" + "=" * 80 + "\n  🧪 最终测试阶段 (Refit)")
        final_test_root = os.path.join(config.output_dir, "final_test_results")
        os.makedirs(final_test_root, exist_ok=True)

        df_full = pd.concat([df_cv_pool, df_test_final])
        final_data = preprocessor.process_and_split(df_full, sampling_sec)
        preprocessor.save(os.path.join(config.output_dir, "preprocessor_meta"))
        
        m_type = getattr(config, "model_type", "enhanced_transformer")
        train_package = final_data.copy()
        
        # 统一使用原始序列 (粒化功能已移除)
        train_package["X_tr"] = final_data["X_tr_raw"]
        train_package["X_val"] = final_data["X_val_raw"]
        X_test_input = final_data["X_test_raw"]
        input_dim = final_data["raw_feat_dim"]

        final_model = build_model(config, input_dim=input_dim, model_type=m_type)
        final_trainer = Trainer(final_model, config, device)
        final_trainer.train(train_package, verbose=True)

        # MC Dropout 多次采样平均 (最终测试)
        mc_samples = getattr(config, 'mc_samples', 30)
        preds_test_scaled = final_trainer.predict(torch.FloatTensor(X_test_input), mc_samples=mc_samples)
        
        y_test_pred_abs_full = restore_absolute_values(preds_test_scaled, final_data["anchor_test"], preprocessor.scaler_y, config.enable_delta_forecast)
        
        y_test_pred_abs_last = y_test_pred_abs_full[:, -1, :]
        anc_test = final_data["anchor_test"].reshape(-1, 1) if config.enable_delta_forecast else 0
        y_test_true_abs_full = final_data["y_test_raw"] + anc_test
        y_test_true_tiled = np.tile(y_test_true_abs_full[:, -1].reshape(-1, 1), (1, 5))

        # [基线模型] 跳过 CQR 校准，直接使用原始预测
        is_baseline = m_type.startswith("baseline_")
        if is_baseline:
            logging.info("  [Baseline] 跳过 CQR 校准 (基线模型不使用 CQR)")
            y_test_calib = y_test_pred_abs_last
            test_corr = np.zeros_like(y_test_pred_abs_last[:, :2])
        else:
            y_test_calib, test_corr, _ = run_cqr_simulation(y_test_pred_abs_last, y_test_true_tiled, config, sampling_sec, "Final", final_test_root)

        # 最终绘图与保存
        vis_dir = os.path.join(final_test_root, "visualizations")
        vis_final = Visualizer(vis_dir, config)

        # 健康度计算结果 (如果需要传入，需要先计算)
        # 这里假设 y_test_calib 已经是最终输出，我们需要重新计算健康度以获取 health_results
        health_model = HealthModel(config)
        health_results = health_model.analyze(y_test_calib, y_true=y_test_true_tiled)

        vis_final.generate_all_plots(
            y_pred=y_test_calib,
            y_true=y_test_true_tiled,
            config=config,
            y_raw=y_test_pred_abs_last,
            corrections=test_corr,
            history=getattr(final_trainer, "history", None),
            y_obs_win=final_data.get("y_test_obs", None), # 尝试获取观测窗，如果没有则为 None
            health_res=health_results,
            prefix="final_test"
        )
        
        # 2. [新增] 专门调用混淆矩阵绘制
        if 'true_states' in health_results:
            vis_final.plot_diagnosis_confusion_matrix(
                y_true_states=health_results['true_states'],
                y_pred_states=health_results['pred_states'],
                fname="final_test_diagnosis_confusion_matrix.png"
            )

        # ==========================================
        # [新增] 1. 健康度相关性分析日志与绘图
        # ==========================================
        if 'true_health_scores' in health_results:
            h_true = health_results['true_health_scores']
            h_pred = health_results['health_scores']
            
            # 计算 Pearson 相关系数
            if len(h_true) > 1:
                corr_val = np.corrcoef(h_true, h_pred)[0, 1]
                # metrics_final['health_correlation'] = corr_val # metrics_final 未定义，跳过
                logging.info(f"  健康度预测相关系数 (Pearson R): {corr_val:.4f}")
                
                # 绘制相关性分析图
                vis_final.plot_health_correlation(
                    health_res=health_results,
                    fname="final_test_health_correlation.png"
                )
            
            # [新增] 2. 绘制 BTP + Health 上下对齐全景图
            # 需要传入 mu 参数供画参考线，从 config 获取
            mu_val = getattr(config, 'health_mu', 22.6)
            vis_final.plot_btp_health_panorama(
                y_true=y_test_true_tiled,
                y_pred=y_test_calib,
                health_res=health_results,
                fname="final_test_btp_health_panorama.png",
                mu=mu_val
            )
            logging.info("  已生成 BTP+健康度全景对比图 (final_test_btp_health_panorama.png)")

        # ==========================================
        # 🛡️ Step 11: Model Diagnostics (深度验证 - 分钟级)
        # ==========================================
        diag_dir = os.path.join(final_test_root, "diagnostics")
        diagnoser = ModelDiagnostics(diag_dir, config)
        
        # [修改点] 提取测试集对应的时间戳
        # 我们需要最后 N_test 个样本的时间戳，并跳过预处理的 buffer
        test_timestamps = df_full["时间"].iloc[-len(y_test_calib):].values
        
        # 1. 执行残差分析 (传入时间戳)
        res_info = diagnoser.perform_residual_analysis(
            y_test_true_tiled,
            y_test_calib,
            timestamps=test_timestamps
        )
        vis_final.plot_residual_diagnostic(res_info, "final_residual_analysis_1min.png")
        

        # 2. 执行鲁棒性压力测试 (传入 Anchor 以还原物理值)
        # 统一使用原始序列 (粒化功能已移除)
        X_test_input = final_data["X_test_raw"]
        
        robust_df = diagnoser.perform_robustness_test(
            final_model,
            torch.FloatTensor(X_test_input).to(device), # 转换为 Tensor 并移动到设备
            y_test_true_tiled,
            final_trainer,
            preprocessor,
            anchor=final_data["anchor_test"],           # <--- 新增
            enable_delta=config.enable_delta_forecast   # <--- 新增
        )
        vis_final.plot_robustness_stress_test(robust_df, "final_robustness_test.png")
        
        # 3. 导出 Excel
        diagnoser.export_to_excel()
        
        logging.info("🛡️ [Diagnostics] 残差分析与鲁棒性测试已完成。")

        logging.info("\n" + "=" * 80 + "\n  ✅ 最终测试完成！结果已保存至: " + final_test_root + "\n" + "=" * 80)


def run_ablation_study(base_config: TrainConfig = None):
    """
    运行消融实验，对比各模块的贡献
    
    Args:
        base_config: 基础配置，如果为 None 则使用默认配置
    """
    if base_config is None:
        base_config = TrainConfig(
            excel_path="20130503ycz.xlsx",
            target_column="北侧_计算BTP位置",
        )
    
    timestamp = make_timestamp()
    ablation_root = os.path.join("outputs", f"Ablation_Study_{timestamp}")
    os.makedirs(ablation_root, exist_ok=True)
    
    # 定义消融配置
    # [修复] 消融主体统一关闭 CQR，保证结构对比公平
    ablation_configs = [
        {"name": "full_model_no_cqr", "changes": {"enable_online_cqr": False}},
        {"name": "no_positional_encoding", "changes": {"enable_positional_encoding": False, "enable_online_cqr": False}},
        {"name": "no_mc_dropout", "changes": {"enable_mc_dropout": False, "enable_online_cqr": False}},
        {"name": "no_revin", "changes": {"enable_revin": False, "enable_online_cqr": False}},
        {"name": "layers_1", "changes": {"num_transformer_layers": 1, "enable_online_cqr": False}},
        {"name": "layers_3", "changes": {"num_transformer_layers": 3, "enable_online_cqr": False}},
        {"name": "heads_2", "changes": {"attn_heads": 2, "enable_online_cqr": False}},
        {"name": "heads_8", "changes": {"attn_heads": 8, "enable_online_cqr": False}},
        # [新增对照] 完整模型 + CQR
        {"name": "full_model_with_cqr", "changes": {"enable_online_cqr": True}},
    ]
    
    print("\n" + "=" * 70)
    print(f"[Ablation Study] 启动消融实验")
    print(f"总输出目录: {ablation_root}")
    print(f"消融配置数量: {len(ablation_configs)}")
    print("=" * 70 + "\n")
    
    # 存储所有实验结果
    all_results = []
    
    # 对每个配置运行交叉验证
    for i, ablation_cfg in enumerate(ablation_configs):
        exp_name = ablation_cfg["name"]
        changes = ablation_cfg["changes"]
        
        print(f"\n>>> [{i+1}/{len(ablation_configs)}] 运行消融配置: {exp_name}")
        if changes:
            print(f"    修改参数: {changes}")
        else:
            print(f"    (完整模型基准)")
        
        # 创建当前配置的副本并应用修改
        current_cfg = copy.deepcopy(base_config)
        for key, value in changes.items():
            if hasattr(current_cfg, key):
                setattr(current_cfg, key, value)
            else:
                print(f"    [Warning] 配置项 {key} 不存在，跳过")
        
        # 设置输出目录
        current_cfg.output_dir = os.path.join(ablation_root, f"Ablation_{exp_name}")
        current_cfg.exp_name = f"Ablation_{exp_name}"
        
        try:
            run_full_pipeline(current_cfg)
            print(f"    [Success] {exp_name} 运行成功")
            
            # 尝试读取结果
            result_path = os.path.join(current_cfg.output_dir, "analysis_report.xlsx")
            if os.path.exists(result_path):
                try:
                    df_result = pd.read_excel(result_path, sheet_name="CV_Summary")
                    all_results.append({
                        "name": exp_name,
                        "changes": str(changes),
                        "mae_mean": df_result["MAE"].mean() if "MAE" in df_result.columns else None,
                        "rmse_mean": df_result["RMSE"].mean() if "RMSE" in df_result.columns else None,
                        "coverage_mean": df_result["Coverage_80"].mean() if "Coverage_80" in df_result.columns else None,
                    })
                except Exception as e:
                    print(f"    [Warning] 读取结果失败: {e}")
                    all_results.append({"name": exp_name, "changes": str(changes), "error": str(e)})
            
        except Exception as e:
            print(f"    [Failed] {exp_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"name": exp_name, "changes": str(changes), "error": str(e)})
            continue
    
    # 汇总消融实验结果
    print("\n" + "=" * 70)
    print("[Ablation Study] 消融实验结果汇总")
    print("=" * 70)
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(ablation_root, "ablation_summary.xlsx")
        summary_df.to_excel(summary_path, index=False)
        print(f"\n结果已保存至: {summary_path}")
        
        # 打印结果表格
        print("\n" + "-" * 70)
        print(f"{'配置名称':<30} {'MAE':<12} {'RMSE':<12} {'Coverage':<12}")
        print("-" * 70)
        for r in all_results:
            if "error" not in r:
                mae = f"{r.get('mae_mean', 'N/A'):.4f}" if r.get('mae_mean') is not None else "N/A"
                rmse = f"{r.get('rmse_mean', 'N/A'):.4f}" if r.get('rmse_mean') is not None else "N/A"
                cov = f"{r.get('coverage_mean', 'N/A'):.2%}" if r.get('coverage_mean') is not None else "N/A"
                print(f"{r['name']:<30} {mae:<12} {rmse:<12} {cov:<12}")
            else:
                print(f"{r['name']:<30} {'ERROR':<12} {r.get('error', '')[:30]}")
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("[Ablation Study] 消融实验完成！")
    print(f"输出目录: {ablation_root}")
    print("=" * 70 + "\n")
    
    return all_results


def run_model_diagnostics(base_config: TrainConfig):
    """
    运行 Model Diagnostics 深度验证实验
    包含：残差分析、鲁棒性测试、超参数敏感性分析
    """
    from btp.model_diagnostics import ModelDiagnostics
    
    timestamp = make_timestamp()
    
    # 使用配置中指定的输出目录
    diagnostics_root = base_config.diagnostics_output_dir
    output_dir = os.path.join(diagnostics_root, f"Diagnostics_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("[Model Diagnostics] 启动深度验证实验")
    print(f"输出目录: {output_dir}")
    print("="*70 + "\n")
    
    # 保存配置
    base_config.save_json(os.path.join(output_dir, "config.json"))
    
    # 设置日志
    log_file = os.path.join(output_dir, "diagnostics.log")
    setup_logging(log_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 1. 加载数据
    print("[1/5] 加载数据...")
    loader = CustomDataLoader()
    df_raw, sampling_sec = loader.load_xlsx(base_config.excel_path, prefer_time_col="时间")
    
    # 2. 数据预处理
    print("[2/5] 数据预处理...")
    preprocessor = DataPreprocessor(base_config)
    df_feat = preprocessor.build_features(df_raw)
    
    # 使用 process_and_split 获取训练/验证/测试数据
    data_dict = preprocessor.process_and_split(df_feat, sampling_sec)
    
    # 提取数据
    X_train = data_dict["X_tr_raw"]
    X_val = data_dict["X_val_raw"]
    X_test = data_dict["X_test_raw"]
    y_train = data_dict["y_tr"]
    y_val = data_dict["y_val"]
    y_test = data_dict["y_test"]
    anchor_test = data_dict["anchor_test"]
    input_dim = data_dict["raw_feat_dim"]
    
    # 转换为 Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    # 3. 构建并训练基准模型
    print("[3/5] 训练基准模型...")
    model = build_model(base_config, input_dim)
    trainer = Trainer(model, base_config, device)
    
    # 训练模型 (使用 data_dict 格式)
    train_package = {
        "X_tr": X_train,
        "X_val": X_val,
        "y_tr": y_train,
        "y_val": y_val
    }
    trainer.train(train_package, verbose=True)
    
    # 4. 获取预测结果
    print("[4/5] 生成预测结果...")
    y_pred_scaled = trainer.predict(X_test_t, mc_samples=base_config.mc_samples)
    
    # 还原绝对值
    y_pred_abs = restore_absolute_values(
        y_pred_scaled,
        anchor_test,
        preprocessor.scaler_y,
        base_config.enable_delta_forecast
    )
    
    # 处理 y_pred_abs 形状: (N, 1, 5) -> (N, 5)
    if y_pred_abs.ndim == 3 and y_pred_abs.shape[1] == 1:
        y_pred_abs = y_pred_abs.squeeze(axis=1)
    
    # 还原真值
    y_test_last = y_test[:, -1] if y_test.ndim == 2 else y_test
    y_true_abs = restore_absolute_values(
        y_test_last.reshape(-1, 1) if y_test_last.ndim == 1 else y_test_last,
        anchor_test,
        preprocessor.scaler_y,
        base_config.enable_delta_forecast
    )
    
    # 扩展为 5 分位数格式 (如果需要)
    if y_true_abs.ndim == 1:
        y_true_abs = np.tile(y_true_abs.reshape(-1, 1), (1, 5))
    elif y_true_abs.shape[1] == 1:
        y_true_abs = np.tile(y_true_abs, (1, 5))
    
    # 5. 运行诊断分析
    print("[5/5] 运行诊断分析...")
    diagnostics = ModelDiagnostics(output_dir, base_config)
    
    # 获取时间戳
    timestamps = data_dict.get('timestamps_test', None)
    
    # 5.1 残差分析
    if base_config.enable_residual_analysis:
        print("  [5.1] 残差分析...")
        diagnostics.perform_residual_analysis(y_true_abs, y_pred_abs, timestamps)
    
    # 5.2 鲁棒性测试
    if base_config.enable_robustness_test:
        print("  [5.2] 鲁棒性测试...")
        diagnostics.perform_robustness_test(
            model, X_test_t, y_true_abs, trainer, preprocessor,
            anchor=anchor_test, enable_delta=base_config.enable_delta_forecast
        )
    
    # 5.3 超参数敏感性分析
    if base_config.enable_hyperparameter_sensitivity:
        print("  [5.3] 超参数敏感性分析...")
        diagnostics.perform_hyperparameter_sensitivity(
            X_train_t, y_train, X_val_t, y_val,
            build_model, preprocessor, device
        )
    
    # 6. 导出报告
    print("\n[Export] 导出诊断报告...")
    report_path = diagnostics.export_to_excel()
    
    print("\n" + "="*70)
    print("[Model Diagnostics] 深度验证实验完成！")
    print(f"报告路径: {report_path}")
    print("="*70 + "\n")
    
    return diagnostics.results


def main():
    # 1. 基础配置初始化
    base_config = TrainConfig(
        excel_path="data/raw/20130503ycz.xlsx",
        target_column="北侧_计算BTP位置",
    )
    
    # ==========================================
    # 实验模式分发
    # ==========================================
    
    # 优先级 1: Model Diagnostics 深度验证实验
    if base_config.enable_model_diagnostics:
        print("[Mode] Model Diagnostics 深度验证模式已启用")
        run_model_diagnostics(base_config)
        return
    
    # 优先级 2: 消融实验模式
    if base_config.enable_ablation_study:
        print("[Mode] 消融实验模式已启用")
        from scripts.run_ablation_experiment import run_ablation_experiment
        run_ablation_experiment(base_config)
        return
    
    # 优先级 3: 模型对比实验模式
    if base_config.enable_model_comparison:
        print("[Mode] 模型对比实验模式已启用")
        # 继续执行下面的模型对比逻辑
    else:
        # 默认：单模型训练
        print("[Mode] 单模型训练模式")
        timestamp = make_timestamp()
        base_config.output_dir = os.path.join("data", f"run_{timestamp}")
        base_config.exp_name = f"run_{timestamp}"
        run_full_pipeline(base_config)
        return
    
    # ==========================================
    # 模型对比实验逻辑
    # ==========================================
    timestamp = make_timestamp()
    experiment_root = os.path.join("outputs", f"Experiment_Significance_{timestamp}")
    os.makedirs(experiment_root, exist_ok=True)
    
    models_to_compare = base_config.comparison_models
    # 确保主模型在对比列表中
    main_model = base_config.model_type
    if main_model not in models_to_compare:
        models_to_compare.insert(0, main_model)

    print("\n" + "*"*60)
    print(f"[Start] 启动单尺度全自动统计显著性实验流水线")
    print(f"总输出目录: {experiment_root}")
    print("*"*60 + "\n")

    # 循环训练所有模型
    for m_name in models_to_compare:
        print(f"\n>>> [阶段 1/2] 正在运行模型: {m_name.upper()}")
        
        current_cfg = copy.deepcopy(base_config)
        current_cfg.model_type = m_name
        current_cfg.output_dir = os.path.join(experiment_root, f"Compare_{m_name}")
        current_cfg.exp_name = f"Compare_{m_name}"
        
        try:
            run_full_pipeline(current_cfg)
            print(f"[Success] {m_name} 运行成功。")
        except Exception as e:
            print(f"[Failed] {m_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 跨模型统计分析
    print("\n" + "*"*60)
    print("[Stats] [阶段 2/2] 启动统计检验与跨模型可视化")
    print("*"*60)

    try:
        from btp.stats import perform_significance_test
        stats_report = perform_significance_test(experiment_root, target_model=base_config.model_type)
        
        from btp.visualizer import Visualizer
        summary_viz = Visualizer(save_dir=experiment_root, config=base_config)
        summary_viz.plot_model_comparison_boxplots(experiment_root, fname="final_model_comparison_boxplots.png")
        
        print(f"\n[Info] 实验报告已生成：\nDirectory: {experiment_root}")
        
    except Exception as e:
        print(f"[Warning] 统计汇总阶段出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "*"*60)
    print("[Finish] 全部实验任务结束！")
    print("*"*60)

if __name__ == "__main__":
    main()
