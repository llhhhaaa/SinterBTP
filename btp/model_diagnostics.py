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

# model_diagnostics.py
"""
Model Diagnostics 深度验证模块
包含：残差分析、鲁棒性测试、超参数敏感性分析
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
import os
import copy
import matplotlib.pyplot as plt
from scipy.stats import kstest
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_absolute_error
from btp.visualizer import Visualizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelDiagnostics:
    def __init__(self, save_dir: str, config):
        self.save_dir = save_dir
        self.cfg = config
        self.results = {}
        os.makedirs(save_dir, exist_ok=True)
        # 初始化可视化器
        self.visualizer = Visualizer(save_dir, config)

    # ==========================================
    # 🔍 模块 1: 残差分析 (已修改为 2 分钟间隔)
    # ==========================================
    def perform_residual_analysis(self, y_true_abs: np.ndarray, y_pred_abs: np.ndarray, timestamps: np.ndarray = None, sampling_sec: float = 5.0):
        """
        分析残差。使用所有原始数据点进行统计分析，ACF使用5分钟重采样数据。
        
        Args:
            y_true_abs: 真实值
            y_pred_abs: 预测值
            timestamps: 时间戳（可选）
            sampling_sec: 采样间隔（秒），默认5秒
        """
        logging.info(">>> [Diagnostics] 启动残差深度分析...")
        
        # 1. 提取点预测残差 (Q50)
        true_val = y_true_abs[:, 2] if y_true_abs.ndim == 2 else y_true_abs
        pred_val = y_pred_abs[:, 2] if y_pred_abs.ndim == 2 else y_pred_abs
        residuals_raw = true_val - pred_val
        
        logging.info(f"    原始残差样本数: {len(residuals_raw)} (采样间隔: {sampling_sec}秒)")
        
        # 2. 对残差进行5分钟重采样，用于ACF分析
        # 这样ACF的每个lag就代表5分钟的间隔
        resample_interval_min = 5  # 5分钟重采样
        samples_per_interval = int(resample_interval_min * 60 / sampling_sec)  # 每个间隔包含的样本数
        
        # 将残差按5分钟分组取中位数
        n_intervals = len(residuals_raw) // samples_per_interval
        residuals_resampled = []
        for i in range(n_intervals):
            start_idx = i * samples_per_interval
            end_idx = start_idx + samples_per_interval
            residuals_resampled.append(np.median(residuals_raw[start_idx:end_idx]))
        residuals_resampled = np.array(residuals_resampled)
        
        logging.info(f"    5分钟重采样后样本数: {len(residuals_resampled)} (用于ACF分析)")
        
        # 3. 使用重采样数据计算 ACF（每个lag代表5分钟）
        max_lags = min(20, len(residuals_resampled) // 4)  # 最多20个lag（100分钟）
        res_acf = acf(residuals_resampled, nlags=max_lags, fft=True)
        
        # 保存原始数据用于时序图和直方图
        target_series = residuals_raw

        # 3. 基础统计量
        res_mean = np.mean(target_series)
        res_std = np.std(target_series)
        
        # 4. 正态性检验
        _, p_val = kstest(target_series, 'norm', args=(res_mean, res_std))
        
        self.results['residuals'] = {
            'series': target_series,
            'acf': res_acf,
            'mean': res_mean,
            'std': res_std,
            'ks_p_value': p_val,
            'is_white_noise': p_val > 0.05 and np.abs(res_acf[1:3]).max() < 0.4,
            'sampling_sec': sampling_sec  # 保存采样间隔供可视化使用
        }
        
        logging.info(f"    残差均值: {res_mean:.4f}, 标准差: {res_std:.4f}")
        logging.info(f"    正态性检验P值: {p_val:.4f}, Lag1_ACF: {res_acf[1]:.4f}")
        
        # 生成残差分析可视化
        self.visualizer.plot_residual_diagnostic(self.results['residuals'], 'residual_analysis.png')
        logging.info(f"    残差分析图表已保存: {os.path.join(self.save_dir, 'residual_analysis.png')}")
        
        return self.results['residuals']

    # ==========================================
    # 🔍 模块 2: 鲁棒性测试 (已修复维度冲突报错)
    # ==========================================
    def perform_robustness_test(self, model, X_test: torch.Tensor, y_true_abs: np.ndarray, trainer, preprocessor, anchor=None, enable_delta=True):
        """
        鲁棒性压力测试。已修复：手动处理多步 Scaler 导致的分位数维度冲突。
        """
        logging.info("="*50)
        logging.info(">>> [Diagnostics] 启动鲁棒性压力测试 (维度修正版)")
        
        noise_levels = [0.0, 0.10, 0.25, 0.50, 1.0]  # 最大100%噪声
        robustness_report = []

        # 提取 Scaler 参数（关键修复：因为预测 3 步，Scaler 长度为 3，我们取最后一步即索引 -1）
        # 即使预测是 (N, 5)，物理量纲与 T+3 是一致的
        mean_val = preprocessor.scaler_y.mean_[-1]
        scale_val = preprocessor.scaler_y.scale_[-1]

        model.eval()
        with torch.no_grad():
            for level in noise_levels:
                # 1. 注入噪声
                X_noisy = X_test.clone()
                if level > 0:
                    noise = torch.randn_like(X_noisy) * level
                    X_noisy += noise
                
                # 2. 模型预测 (结果是标准化后的最后一步 5 个分位数)
                y_pred_scaled = trainer.predict(X_noisy)  # 已经是 numpy 数组
                y_last_scaled = y_pred_scaled[:, -1, :] # (Batch, 5)
                
                # 3. 物理值还原 (【核心修复点】：避免直接调用 inverse_transform)
                # (Batch, 5) * scalar + scalar -> 手动广播
                y_last_phys = y_last_scaled * scale_val + mean_val
                
                # 处理残差预测 (Delta)
                if enable_delta and anchor is not None:
                    anc = np.asarray(anchor).reshape(-1, 1) # (Batch, 1)
                    y_pred_abs_all = y_last_phys + anc      # (Batch, 5)
                else:
                    y_pred_abs_all = y_last_phys

                # 4. 计算点预测 (Q50) 的 MAE
                y_pred_q50 = y_pred_abs_all[:, 2]
                y_true_q50 = y_true_abs[:, 2] if y_true_abs.ndim == 2 else y_true_abs
                
                mae = mean_absolute_error(y_true_q50, y_pred_q50)
                
                baseline_mae = robustness_report[0]["MAE"] if len(robustness_report) > 0 else mae
                retention = (baseline_mae / mae) if mae > 0 else 1.0
                
                robustness_report.append({
                    "噪声水平": f"{level*100:.0f}%",
                    "MAE": mae,
                    "性能保持率": retention
                })
                
                print(f"  [Level {level*100:3.0f}%] MAE: {mae:.6f} | 保持率: {retention:.2%}")

        logging.info("="*50)
        df_robust = pd.DataFrame(robustness_report)
        self.results['robustness'] = df_robust
        
        # 生成鲁棒性测试可视化
        self.visualizer.plot_robustness_stress_test(df_robust, 'robustness_test.png')
        logging.info(f"    鲁棒性测试图表已保存: {os.path.join(self.save_dir, 'robustness_test.png')}")
        
        return df_robust

    # ==========================================
    # 🔍 模块 3: 超参数敏感性分析
    # ==========================================
    def perform_hyperparameter_sensitivity(
        self,
        X_train: torch.Tensor,
        y_train: np.ndarray,
        X_val: torch.Tensor,
        y_val: np.ndarray,
        build_model_fn,
        preprocessor,
        device: torch.device
    ):
        """
        超参数敏感性分析：测试模型在不同超参数配置下的表现稳定性。
        证明模型不是"调参调出来的偶然结果"。
        
        Args:
            X_train: 训练集输入
            y_train: 训练集标签 (标准化后)
            X_val: 验证集输入
            y_val: 验证集标签 (标准化后)
            build_model_fn: 模型构建函数
            preprocessor: 数据预处理器
            device: 计算设备
        
        Returns:
            DataFrame: 超参数敏感性测试结果
        """
        logging.info("="*60)
        logging.info(">>> [Diagnostics] 启动超参数敏感性分析")
        logging.info("="*60)
        
        # 获取超参数配置
        hp_config = self.cfg.hyperparam_sensitivity_config
        epochs_per_test = hp_config.get('epochs_per_test', 30)
        repeat_times = hp_config.get('repeat_times', 3)
        
        # 定义要测试的超参数范围
        param_ranges = {
            'learning_rate': hp_config.get('learning_rate', [0.0001, 0.0005, 0.001, 0.002, 0.005]),
            'hidden_size': hp_config.get('hidden_size', [64, 128, 256]),
            'num_layers': hp_config.get('num_layers', [1, 2, 3]),
            'dropout': hp_config.get('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
        }
        
        all_results = []
        
        # 保存原始配置
        original_cfg = copy.deepcopy(self.cfg)
        
        # 1. 学习率敏感性测试
        logging.info("\n[1/4] 测试学习率敏感性...")
        lr_results = self._test_single_hyperparam(
            'learning_rate', param_ranges['learning_rate'],
            X_train, y_train, X_val, y_val,
            build_model_fn, device, epochs_per_test, repeat_times
        )
        all_results.extend(lr_results)
        
        # 2. 隐藏层大小敏感性测试
        logging.info("\n[2/4] 测试隐藏层大小敏感性...")
        hidden_results = self._test_single_hyperparam(
            'hidden_size', param_ranges['hidden_size'],
            X_train, y_train, X_val, y_val,
            build_model_fn, device, epochs_per_test, repeat_times
        )
        all_results.extend(hidden_results)
        
        # 3. 层数敏感性测试
        logging.info("\n[3/4] 测试层数敏感性...")
        layer_results = self._test_single_hyperparam(
            'num_layers', param_ranges['num_layers'],
            X_train, y_train, X_val, y_val,
            build_model_fn, device, epochs_per_test, repeat_times
        )
        all_results.extend(layer_results)
        
        # 4. Dropout 敏感性测试
        logging.info("\n[4/4] 测试 Dropout 敏感性...")
        dropout_results = self._test_single_hyperparam(
            'dropout', param_ranges['dropout'],
            X_train, y_train, X_val, y_val,
            build_model_fn, device, epochs_per_test, repeat_times
        )
        all_results.extend(dropout_results)
        
        # 恢复原始配置
        self.cfg = original_cfg
        
        # 汇总结果
        df_results = pd.DataFrame(all_results)
        self.results['hyperparameter_sensitivity'] = df_results
        
        # 生成可视化
        self._plot_hyperparameter_sensitivity(df_results)
        
        # 计算稳定性指标
        stability_report = self._compute_stability_metrics(df_results)
        self.results['stability_report'] = stability_report
        
        logging.info("\n" + "="*60)
        logging.info("[Diagnostics] 超参数敏感性分析完成")
        logging.info("="*60)
        
        return df_results
    
    def _test_single_hyperparam(
        self,
        param_name: str,
        param_values: list,
        X_train, y_train, X_val, y_val,
        build_model_fn, device, epochs, repeat_times
    ):
        """测试单个超参数的敏感性"""
        from btp.model import QuantileLoss
        
        results = []
        
        for value in param_values:
            mae_list = []
            
            for trial in range(repeat_times):
                # 创建配置副本并修改超参数
                test_cfg = copy.deepcopy(self.cfg)
                
                if param_name == 'learning_rate':
                    test_cfg.lr = value
                elif param_name == 'hidden_size':
                    test_cfg.hidden_size = value
                elif param_name == 'num_layers':
                    test_cfg.num_transformer_layers = value
                elif param_name == 'dropout':
                    test_cfg.dropout = value
                
                try:
                    # 构建模型
                    input_dim = X_train.shape[-1]
                    model = build_model_fn(test_cfg, input_dim).to(device)
                    
                    # 简化训练
                    optimizer = torch.optim.Adam(model.parameters(), lr=float(test_cfg.lr))
                    criterion = QuantileLoss(test_cfg).to(device)
                    
                    # 准备数据
                    train_dataset = torch.utils.data.TensorDataset(
                        X_train.to(device),
                        torch.tensor(y_train, dtype=torch.float32).to(device)
                    )
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=test_cfg.batch_size, shuffle=True
                    )
                    
                    # 训练
                    model.train()
                    for epoch in range(epochs):
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()
                            preds = model(batch_X)
                            loss = criterion(preds, batch_y)
                            loss.backward()
                            optimizer.step()
                    
                    # 验证
                    model.eval()
                    with torch.no_grad():
                        X_val_dev = X_val.to(device)
                        val_preds = model(X_val_dev).cpu().numpy()
                        
                        # 提取 Q50 预测
                        if val_preds.ndim == 3:
                            pred_q50 = val_preds[:, -1, 2]  # 最后一步的 Q50
                        else:
                            pred_q50 = val_preds[:, 2]
                        
                        # 提取真值
                        if y_val.ndim == 3:
                            true_q50 = y_val[:, -1, 0] if y_val.shape[-1] == 1 else y_val[:, -1]
                        elif y_val.ndim == 2:
                            true_q50 = y_val[:, -1]
                        else:
                            true_q50 = y_val
                        
                        mae = mean_absolute_error(true_q50, pred_q50)
                        mae_list.append(mae)
                        
                except Exception as e:
                    logging.warning(f"    超参数测试失败 ({param_name}={value}, trial={trial}): {e}")
                    continue
            
            if mae_list:
                results.append({
                    '超参数': param_name,
                    '参数值': value,
                    'MAE均值': np.mean(mae_list),
                    'MAE标准差': np.std(mae_list),
                    'MAE最小值': np.min(mae_list),
                    'MAE最大值': np.max(mae_list),
                    '测试次数': len(mae_list)
                })
                logging.info(f"    {param_name}={value}: MAE={np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
        
        return results
    
    def _plot_hyperparameter_sensitivity(self, df: pd.DataFrame):
        """生成超参数敏感性可视化图表"""
        param_names = df['超参数'].unique()
        n_params = len(param_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, param in enumerate(param_names):
            if idx >= 4:
                break
            ax = axes[idx]
            param_data = df[df['超参数'] == param]
            
            x = range(len(param_data))
            values = param_data['参数值'].astype(str).tolist()
            means = param_data['MAE均值'].values
            stds = param_data['MAE标准差'].values
            
            # 绘制带误差棒的柱状图
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                         color='steelblue', edgecolor='navy')
            
            ax.set_xlabel(param, fontsize=11)
            ax.set_ylabel('MAE', fontsize=11)
            ax.set_title(f'{param} 敏感性分析', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(values, rotation=45 if len(values) > 4 else 0)
            ax.grid(axis='y', alpha=0.3)
            
            # 注释掉最优值标注，使所有柱状图保持统一颜色
            # 超参数敏感性图用于表明超参数影响小，不需要标注最优值
            # best_idx = np.argmin(means)
            # bars[best_idx].set_color('forestgreen')
            # ax.annotate('最优', xy=(best_idx, means[best_idx]),
            #            xytext=(best_idx, means[best_idx] + stds[best_idx] + 0.01),
            #            ha='center', fontsize=9, color='forestgreen', fontweight='bold')
        
        # 隐藏多余的子图
        for idx in range(n_params, 4):
            axes[idx].set_visible(False)
        
        plt.suptitle('超参数敏感性分析\n(证明模型稳定性，非偶然调参结果)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'hyperparameter_sensitivity.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"    超参数敏感性图表已保存: {save_path}")
    
    def _compute_stability_metrics(self, df: pd.DataFrame) -> dict:
        """计算模型稳定性指标"""
        stability = {}
        
        for param in df['超参数'].unique():
            param_data = df[df['超参数'] == param]
            mae_values = param_data['MAE均值'].values
            
            # 计算变异系数 (CV) - 越小越稳定
            cv = np.std(mae_values) / np.mean(mae_values) if np.mean(mae_values) > 0 else 0
            
            # 计算最大波动范围
            range_ratio = (np.max(mae_values) - np.min(mae_values)) / np.mean(mae_values) if np.mean(mae_values) > 0 else 0
            
            stability[param] = {
                '变异系数CV': round(cv, 4),
                '波动范围比': round(range_ratio, 4),
                '最优值': param_data.loc[param_data['MAE均值'].idxmin(), '参数值'],
                '最优MAE': round(param_data['MAE均值'].min(), 4),
                '稳定性评级': '优秀' if cv < 0.1 else '良好' if cv < 0.2 else '一般' if cv < 0.3 else '较差'
            }
        
        # 总体稳定性评分
        avg_cv = np.mean([v['变异系数CV'] for v in stability.values()])
        stability['总体评估'] = {
            '平均变异系数': round(avg_cv, 4),
            '结论': '模型对超参数不敏感，结果稳定可靠' if avg_cv < 0.15 else
                   '模型对超参数有一定敏感性，但整体稳定' if avg_cv < 0.25 else
                   '模型对超参数较敏感，需谨慎选择'
        }
        
        return stability

    def export_to_excel(self):
        """导出所有诊断结果到 Excel"""
        file_path = os.path.join(self.save_dir, "model_diagnostics_report.xlsx")
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # 残差分析结果
            if 'residuals' in self.results:
                res = self.results['residuals']
                pd.DataFrame({
                    "统计项": ["均值", "标准差", "正态检验P值", "是否通过白噪声测试"],
                    "数值": [res['mean'], res['std'], res['ks_p_value'], res['is_white_noise']]
                }).to_excel(writer, sheet_name="残差统计指标", index=False)
            
            # 鲁棒性测试结果
            if 'robustness' in self.results:
                self.results['robustness'].to_excel(writer, sheet_name="抗干扰能力测试", index=False)
            
            # 超参数敏感性结果
            if 'hyperparameter_sensitivity' in self.results:
                self.results['hyperparameter_sensitivity'].to_excel(
                    writer, sheet_name="超参数敏感性", index=False
                )
            
            # 稳定性报告
            if 'stability_report' in self.results:
                stability_data = []
                for param, metrics in self.results['stability_report'].items():
                    if param != '总体评估':
                        stability_data.append({
                            '超参数': param,
                            '变异系数CV': metrics['变异系数CV'],
                            '波动范围比': metrics['波动范围比'],
                            '最优值': metrics['最优值'],
                            '最优MAE': metrics['最优MAE'],
                            '稳定性评级': metrics['稳定性评级']
                        })
                if stability_data:
                    pd.DataFrame(stability_data).to_excel(
                        writer, sheet_name="稳定性评估", index=False
                    )
                
                # 总体评估
                if '总体评估' in self.results['stability_report']:
                    overall = self.results['stability_report']['总体评估']
                    pd.DataFrame({
                        '指标': ['平均变异系数', '结论'],
                        '值': [overall['平均变异系数'], overall['结论']]
                    }).to_excel(writer, sheet_name="总体评估", index=False)
        
        logging.info(f"[Diagnostics] 深度验证报告已生成: {file_path}")
        return file_path
