# 健康度模型实验脚本
# 用于生成测试集数据和图像，支持论文初稿撰写

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib
matplotlib.use('Agg')
import logging
import os
import json
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime

from btp.config import TrainConfig
from btp.data_loader import DataLoader as CustomDataLoader
from btp.preprocessor import DataPreprocessor
from btp.model import build_model
from btp.trainer import Trainer
from btp.health_model import HealthModel
from btp.visualizer import Visualizer
from btp.metrics import evaluate_quantile_regression
from btp.utils import setup_logging


def restore_absolute_values(y_scaled, anchor, scaler_y, enable_delta):
    """将模型输出还原为绝对物理值"""
    y_scaled = np.asarray(y_scaled)
    
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
    
    if enable_delta:
        if anchor is None:
            return y_phys
        if hasattr(anchor, "values"):
            anchor = anchor.values
        anchor = np.asarray(anchor)
        
        if y_phys.ndim == 3:
            if anchor.ndim == 1:
                anchor = anchor[:, None, None]
            elif anchor.ndim == 2:
                anchor = anchor[:, :, None]
            return y_phys + anchor
        if anchor.ndim == 1:
            anchor = anchor[:, None]
        return y_phys + anchor
    
    return y_phys


def run_health_model_experiment(config: TrainConfig, output_dir: str):
    """
    运行健康度模型实验
    
    Args:
        config: 训练配置
        output_dir: 输出目录
    """
    # 设置日志
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "health_experiment.log")
    setup_logging(log_path, reset_handlers=True)
    
    logging.info("=" * 80)
    logging.info("  🏥 健康度模型实验")
    logging.info("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[Config] Device: {device}")
    
    # 1. 加载数据
    logging.info("[Step 1] 加载数据...")
    data_loader = CustomDataLoader()
    df_raw, sampling_sec = data_loader.load_xlsx(config.excel_path, prefer_time_col="时间")
    
    # 2. 数据预处理
    logging.info("[Step 2] 数据预处理...")
    preprocessor = DataPreprocessor(config)
    df_feat = preprocessor.build_features(df_raw)
    
    n_total = len(df_feat)
    n_test_holdout = int(n_total * config.test_split)
    df_train_pool = df_feat.iloc[:-n_test_holdout].copy() if n_test_holdout > 0 else df_feat.copy()
    df_test_final = df_feat.iloc[-n_test_holdout:].copy() if n_test_holdout > 0 else None
    
    # 3. 训练模型
    logging.info("[Step 3] 训练模型...")
    train_data = preprocessor.process_and_split(df_train_pool, sampling_sec)
    
    train_package = {
        "X_tr": train_data["X_tr_raw"],
        "X_val": train_data["X_val_raw"],
        "y_tr": train_data["y_tr"],
        "y_val": train_data["y_val"]
    }
    input_dim = train_data["raw_feat_dim"]
    
    model = build_model(config, input_dim=input_dim, model_type=config.model_type)
    trainer = Trainer(model, config, device)
    trainer.train(train_package, verbose=True)
    
    # 4. 测试集预测
    logging.info("[Step 4] 测试集预测...")
    if df_test_final is not None:
        df_full = pd.concat([df_train_pool, df_test_final])
        test_data = preprocessor.process_and_split(df_full, sampling_sec)
    else:
        test_data = train_data
    
    X_test = test_data["X_test_raw"]
    y_test = test_data["y_test_raw"]
    anchor_test = test_data.get("anchor_test", None)
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    mc_samples = getattr(config, 'mc_samples', 30)
    logging.info(f"  使用 MC Dropout 采样 {mc_samples} 次...")
    
    preds_scaled = trainer.predict(X_test_tensor, mc_samples=mc_samples)
    
    # 还原绝对值
    y_pred_abs = restore_absolute_values(
        preds_scaled, anchor_test, preprocessor.scaler_y, config.enable_delta_forecast
    )
    
    # 取最后一步 T+5
    y_pred_last = y_pred_abs[:, -1, :]
    anc = anchor_test.reshape(-1, 1) if config.enable_delta_forecast and anchor_test is not None else 0
    y_true_full = y_test + anc
    y_true_last = y_true_full[:, -1]
    y_true_tiled = np.tile(y_true_last.reshape(-1, 1), (1, 5))
    
    # 5. 健康度分析
    logging.info("[Step 5] 健康度分析...")
    health_model = HealthModel(config)
    health_results = health_model.analyze(y_pred_last, y_true=y_true_tiled)
    
    # 6. 计算预测指标
    logging.info("[Step 6] 计算预测指标...")
    metrics = evaluate_quantile_regression(y_true_tiled, y_pred_last, compute_width_stats=True)
    
    # 合并健康度指标
    all_metrics = {**metrics}
    all_metrics.update({
        f"health_{k}": v for k, v in health_results.items() 
        if isinstance(v, (float, int))
    })
    
    # 7. 生成可视化
    logging.info("[Step 7] 生成可视化...")
    vis_dir = os.path.join(output_dir, "figures")
    os.makedirs(vis_dir, exist_ok=True)
    visualizer = Visualizer(vis_dir, config)
    
    # 7.1 混淆矩阵
    if 'true_states' in health_results and 'pred_states' in health_results:
        logging.info("  生成混淆矩阵...")
        visualizer.plot_diagnosis_confusion_matrix(
            y_true_states=health_results['true_states'],
            y_pred_states=health_results['pred_states'],
            fname="health_confusion_matrix.png"
        )
    
    # 7.2 健康度曲线图
    if 'health_scores' in health_results:
        logging.info("  生成健康度曲线...")
        plot_health_curves(health_results, y_true_tiled, y_pred_last, vis_dir, config)
    
    # 7.3 BTP + 健康度全景图
    if 'true_health_scores' in health_results:
        logging.info("  生成BTP健康度全景图...")
        mu_val = getattr(config, 'health_mu', 22.5)
        visualizer.plot_btp_health_panorama(
            y_true=y_true_tiled,
            y_pred=y_pred_last,
            health_res=health_results,
            fname="btp_health_panorama.png",
            mu=mu_val
        )
        
        # 健康度相关性图
        visualizer.plot_health_correlation(
            health_res=health_results,
            fname="health_correlation.png"
        )
    
    # 8. 保存结果数据
    logging.info("[Step 8] 保存结果数据...")
    save_results(output_dir, health_results, all_metrics, y_true_tiled, y_pred_last)
    
    # 9. 生成摘要报告
    logging.info("[Step 9] 生成摘要报告...")
    generate_summary_report(output_dir, health_results, all_metrics)
    
    logging.info("=" * 80)
    logging.info("  ✅ 健康度模型实验完成！")
    logging.info(f"  结果保存至: {output_dir}")
    logging.info("=" * 80)
    
    return health_results, all_metrics


def plot_health_curves(health_results, y_true, y_pred, save_dir, config):
    """绘制健康度评分曲线图"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 子图1: BTP真值与预测
    ax1 = axes[0]
    t = np.arange(len(y_true))
    ax1.plot(t, y_true[:, 2], 'b-', alpha=0.7, label='BTP真值', linewidth=1)
    ax1.plot(t, y_pred[:, 2], 'r--', alpha=0.7, label='BTP预测', linewidth=1)
    mu_val = getattr(config, 'health_mu', 22.5)
    ax1.axhline(mu_val, color='g', linestyle=':', alpha=0.5, label=f'目标值 μ={mu_val}')
    ax1.set_ylabel('BTP位置')
    ax1.legend(loc='upper right')
    ax1.set_title('BTP预测对比')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 健康度评分
    ax2 = axes[1]
    ax2.plot(t, health_results['health_scores'], 'b-', linewidth=1.5, label='预测健康度')
    if 'true_health_scores' in health_results:
        ax2.plot(t, health_results['true_health_scores'], 'g--', alpha=0.7, linewidth=1, label='真实健康度')
    ax2.axhline(65, color='orange', linestyle='--', alpha=0.5, label='正常阈值 (65)')
    ax2.axhline(38, color='red', linestyle='--', alpha=0.5, label='故障阈值 (38)')
    ax2.set_ylabel('健康度评分')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper right')
    ax2.set_title('健康度评分曲线')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 三分量分解
    ax3 = axes[2]
    if 'H_pos_series' in health_results:
        ax3.plot(t, health_results['H_pos_series'] * 100, 'r-', alpha=0.7, label='H_pos (偏离度)')
    if 'H_stab_series' in health_results:
        ax3.plot(t, health_results['H_stab_series'] * 100, 'g-', alpha=0.7, label='H_stab (稳定性)')
    if 'H_trend_series' in health_results:
        ax3.plot(t, health_results['H_trend_series'] * 100, 'b-', alpha=0.7, label='H_trend (趋势)')
    ax3.set_ylabel('分量评分')
    ax3.set_xlabel('样本索引')
    ax3.set_ylim(0, 105)
    ax3.legend(loc='upper right')
    ax3.set_title('健康度三分量分解')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'health_score_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_results(output_dir, health_results, metrics, y_true, y_pred):
    """保存结果数据"""
    # 保存健康度数据
    health_data = {
        'health_scores': health_results['health_scores'],
        'pred_states': health_results['pred_states']
    }
    if 'true_health_scores' in health_results:
        health_data['true_health_scores'] = health_results['true_states']
    if 'true_states' in health_results:
        health_data['true_states'] = health_results['true_states']
    
    df_health = pd.DataFrame(health_data)
    df_health.to_csv(os.path.join(output_dir, 'health_scores.csv'), index=False)
    
    # 保存预测数据
    pred_data = {
        'true_btp': y_true[:, 2],
        'pred_btp': y_pred[:, 2],
        'pred_q10': y_pred[:, 0],
        'pred_q25': y_pred[:, 1],
        'pred_q50': y_pred[:, 2],
        'pred_q75': y_pred[:, 3],
        'pred_q90': y_pred[:, 4]
    }
    df_pred = pd.DataFrame(pred_data)
    df_pred.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # 保存指标数据
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)


def generate_summary_report(output_dir, health_results, metrics):
    """生成摘要报告"""
    report_path = os.path.join(output_dir, 'experiment_summary.md')
    
    # 计算混淆矩阵
    if 'true_states' in health_results and 'pred_states' in health_results:
        from sklearn.metrics import confusion_matrix, classification_report
        y_true_states = health_results['true_states']
        y_pred_states = health_results['pred_states']
        
        state_names = ['过烧', '疑似过烧', '正常', '疑似欠烧', '欠烧']
        
        # 计算分类准确率
        accuracy = np.mean(y_true_states == y_pred_states)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true_states, y_pred_states)
        
        # 分类报告
        try:
            clf_report = classification_report(
                y_true_states, y_pred_states, 
                target_names=state_names,
                output_dict=True,
                zero_division=0
            )
        except:
            clf_report = {}
    else:
        accuracy = 0
        cm = None
        clf_report = {}
    
    # 健康度统计
    health_mean = np.mean(health_results['health_scores'])
    health_std = np.std(health_results['health_scores'])
    health_min = np.min(health_results['health_scores'])
    
    # 相关性
    if 'true_health_scores' in health_results:
        corr = np.corrcoef(health_results['true_health_scores'], health_results['health_scores'])[0, 1]
    else:
        corr = 0
    
    # 生成报告
    report = f"""# 健康度模型实验报告

## 实验概述

本实验对烧结BTP预测模型进行健康度评估和工况分类分析。

## 1. 健康度评分统计

| 指标 | 数值 |
|------|------|
| 平均健康度 | {health_mean:.2f} |
| 健康度标准差 | {health_std:.2f} |
| 最低健康度 | {health_min:.2f} |

## 2. 工况分类准确率

| 指标 | 数值 |
|------|------|
| 总体准确率 | {accuracy:.2%} |
| 健康度相关性 (R) | {corr:.4f} |

### 混淆矩阵

|  | 过烧 | 疑似过烧 | 正常 | 疑似欠烧 | 欠烧 |
|--|------|----------|------|----------|------|
"""
    
    if cm is not None:
        n_classes = cm.shape[0]
        for i in range(n_classes):
            if i < len(state_names):
                row_name = state_names[i]
            else:
                row_name = f"类别{i}"
            row_vals = ' | '.join([f'{cm[i, j]:5d}' for j in range(n_classes)])
            report += f"| {row_name} | {row_vals} |\n"
    
    report += f"""
### 各工况分类指标

| 工况 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
"""
    
    for name in state_names:
        if name in clf_report:
            r = clf_report[name]
            report += f"| {name} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |\n"
    
    report += f"""
## 3. 预测性能指标

| 指标 | 数值 |
|------|------|
| MAE (Q50) | {metrics.get('mae_q50', 0):.4f} |
| RMSE | {metrics.get('rmse', 0):.4f} |
| Coverage 80% | {metrics.get('coverage_80', 0):.2%} |
| Coverage 50% | {metrics.get('coverage_50', 0):.2%} |
| 区间宽度均值 | {metrics.get('interval_width_mean', 0):.4f} |

## 4. 生成的图像文件

- `figures/health_confusion_matrix.png` - 工况分类混淆矩阵
- `figures/health_score_curves.png` - 健康度评分曲线
- `figures/btp_health_panorama.png` - BTP与健康度全景图
- `figures/health_correlation.png` - 健康度相关性散点图

## 5. 数据文件

- `health_scores.csv` - 健康度评分数据
- `predictions.csv` - 预测结果数据
- `metrics.csv` - 性能指标数据
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"摘要报告已保存至: {report_path}")


def main():
    """主函数"""
    # 配置 - 使用全量数据进行训练
    config = TrainConfig(
        excel_path="data/raw/20130503ycz.xlsx",
        target_column="北侧_计算BTP位置",
        model_type="enhanced_transformer",
        test_split=0.15,
        forecast_steps=5,
        mc_samples=20,
        # 健康度参数
        health_mu=22.5,
        health_sigma_left=0.3,
        health_sigma_right=0.5,
        health_beta_ewma=0.7,
    )
    
    # 输出目录
    output_dir = "实验记录/健康度模型/experiment_results"
    
    # 运行实验
    health_results, metrics = run_health_model_experiment(config, output_dir)
    
    print("\n" + "=" * 70)
    print("实验完成！主要结果：")
    print(f"  - 平均健康度: {np.mean(health_results['health_scores']):.2f}")
    if 'true_states' in health_results:
        accuracy = np.mean(health_results['true_states'] == health_results['pred_states'])
        print(f"  - 工况分类准确率: {accuracy:.2%}")
    print(f"  - 结果目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()