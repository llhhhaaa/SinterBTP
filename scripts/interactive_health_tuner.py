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
交互式健康度参数调优工具 v3.0
================================

基于已有实验数据 (run_20260303_153938) 的交互式健康度参数调优工具。

功能：
1. 加载已有预测数据和真实值
2. 交互式调整健康度各项参数
3. 实时显示三种关键图表：
   - BTP健康度全景图 (final_test_btp_health_panorama)
   - 工况分类混淆矩阵 (final_test_diagnosis_confusion_matrix)
   - 健康度相关性散点图 (final_test_health_correlation)

使用方法：
    python scripts/interactive_health_tuner.py
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
import json

# 导入健康度模型
try:
    from btp.health_model import HealthModel
except ImportError:
    messagebox.showerror("错误", "无法导入 health_model.py，请确保项目结构正确。")
    sys.exit(1)

# 设置 Matplotlib 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 配置类
# ==========================================

class TunerConfig:
    """动态配置类 - 与 HealthModel 参数对应"""
    def __init__(self, **kwargs):
        # 目标设定
        self.health_mu = kwargs.get('health_mu', 22.56)
        self.health_sigma_left = kwargs.get('health_sigma_left', 0.2)
        self.health_sigma_right = kwargs.get('health_sigma_right', 0.3)
        self.health_sigma_limit = kwargs.get('health_sigma_limit', 0.15)
        self.health_width_tol = kwargs.get('health_width_tol', 1.5)
        
        # 权重参数
        self.health_W_pos = kwargs.get('health_W_pos', 1.2)
        self.health_W_stab = kwargs.get('health_W_stab', 1.0)
        self.health_W_trend = kwargs.get('health_W_trend', 1.0)
        
        # 动态特性
        self.health_k_stab = kwargs.get('health_k_stab', 2.0)
        self.health_alpha_trend = kwargs.get('health_alpha_trend', 0.8)
        self.health_beta_ewma = kwargs.get('health_beta_ewma', 0.7)
        
        # 状态判定
        self.health_thresh_normal = kwargs.get('health_thresh_normal', 75.0)
        self.health_thresh_fault = kwargs.get('health_thresh_fault', 50.0)
        self.health_hysteresis_band = kwargs.get('health_hysteresis_band', 3.0)
        self.health_max_penalty = kwargs.get('health_max_penalty', 0.5)
        self.health_initial_filter_state = kwargs.get('health_initial_filter_state', 0.8)
        
        # 其他必要参数
        self.volatility_window_size = kwargs.get('volatility_window_size', 10)
        self.forecast_steps = kwargs.get('forecast_steps', 3)


# ==========================================
# 主应用程序
# ==========================================

class InteractiveHealthTuner:
    """交互式健康度参数调优器"""
    
    # 默认参数值
    DEFAULTS = {
        'mu': 22.56,
        'sigma_left': 0.2,
        'sigma_right': 0.3,
        'sigma_limit': 0.15,
        'width_tol': 1.5,
        'w_pos': 1.2,
        'w_stab': 1.0,
        'w_trend': 1.0,
        'k_stab': 2.0,
        'alpha_trend': 0.8,
        'beta_ewma': 0.7,
        'thresh_normal': 75.0,
        'hysteresis_band': 3.0,
        'max_penalty': 0.5,
        'initial_filter_state': 0.8,
    }
    
    # 状态标签
    STATE_LABELS = ['过烧', '正常', '欠烧']
    
    def __init__(self, root):
        self.root = root
        self.root.title("交互式健康度参数调优器 v3.0")
        self.root.geometry("1600x950")
        
        # 数据容器
        self.y_pred = None  # 预测分位数 (N, 5)
        self.y_true = None  # 真实值 (N, 1) 或 (N, 5)
        self.data_loaded = False
        self.current_results = None
        
        # 参数变量
        self.vars = {}
        self._init_variables()
        
        # 构建 UI
        self._build_ui()
        
        # 自动加载默认数据
        self._auto_load_data()
    
    def _init_variables(self):
        """初始化参数变量"""
        # 目标设定参数
        self.vars['mu'] = tk.DoubleVar(value=self.DEFAULTS['mu'])
        self.vars['sigma_left'] = tk.DoubleVar(value=self.DEFAULTS['sigma_left'])
        self.vars['sigma_right'] = tk.DoubleVar(value=self.DEFAULTS['sigma_right'])
        self.vars['sigma_limit'] = tk.DoubleVar(value=self.DEFAULTS['sigma_limit'])
        self.vars['width_tol'] = tk.DoubleVar(value=self.DEFAULTS['width_tol'])
        
        # 权重参数
        self.vars['w_pos'] = tk.DoubleVar(value=self.DEFAULTS['w_pos'])
        self.vars['w_stab'] = tk.DoubleVar(value=self.DEFAULTS['w_stab'])
        self.vars['w_trend'] = tk.DoubleVar(value=self.DEFAULTS['w_trend'])
        
        # 动态特性
        self.vars['k_stab'] = tk.DoubleVar(value=self.DEFAULTS['k_stab'])
        self.vars['alpha_trend'] = tk.DoubleVar(value=self.DEFAULTS['alpha_trend'])
        self.vars['beta_ewma'] = tk.DoubleVar(value=self.DEFAULTS['beta_ewma'])
        
        # 状态判定
        self.vars['thresh_normal'] = tk.DoubleVar(value=self.DEFAULTS['thresh_normal'])
        self.vars['hysteresis_band'] = tk.DoubleVar(value=self.DEFAULTS['hysteresis_band'])
        self.vars['max_penalty'] = tk.DoubleVar(value=self.DEFAULTS['max_penalty'])
        self.vars['initial_filter_state'] = tk.DoubleVar(value=self.DEFAULTS['initial_filter_state'])
        
        # 指标显示
        self.vars['status'] = tk.StringVar(value="等待加载数据...")
        self.vars['health_mean'] = tk.StringVar(value="--")
        self.vars['health_std'] = tk.StringVar(value="--")
        self.vars['correlation'] = tk.StringVar(value="--")
        self.vars['accuracy'] = tk.StringVar(value="--")
        self.vars['overburn_rate'] = tk.StringVar(value="--")
        self.vars['underburn_rate'] = tk.StringVar(value="--")
    
    def _build_ui(self):
        """构建用户界面"""
        # 顶部控制栏
        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.pack(fill="x")
        
        ttk.Label(top_frame, text="数据目录:").pack(side="left", padx=5)
        self.data_path_var = tk.StringVar(value="data/run_20260303_153938")
        ttk.Entry(top_frame, textvariable=self.data_path_var, width=40).pack(side="left", padx=5)
        ttk.Button(top_frame, text="加载数据", command=self._load_data).pack(side="left", padx=5)
        ttk.Button(top_frame, text="重置参数", command=self._reset_params).pack(side="left", padx=5)
        ttk.Button(top_frame, text="导出参数", command=self._export_params).pack(side="left", padx=5)
        
        # 状态显示
        ttk.Label(top_frame, textvariable=self.vars['status'], foreground="blue").pack(side="left", padx=20)
        
        # 主体区域 - 左右分栏
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 左侧：参数控制面板
        control_frame = ttk.LabelFrame(main_paned, text="参数控制", padding=10)
        main_paned.add(control_frame, weight=0)
        
        # 使用 Canvas + ScrollFrame 支持滚动
        canvas = tk.Canvas(control_frame, width=320)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 目标设定组
        grp_target = ttk.LabelFrame(scroll_frame, text="目标设定 (H_pos)", padding=5)
        grp_target.pack(fill="x", pady=5, padx=5)
        self._add_slider(grp_target, "目标值 (μ)", 'mu', 20.0, 25.0, 0.01)
        self._add_slider(grp_target, "左侧敏感度 (σ_L)", 'sigma_left', 0.05, 1.0, 0.01)
        self._add_slider(grp_target, "右侧敏感度 (σ_R)", 'sigma_right', 0.05, 1.0, 0.01)
        self._add_slider(grp_target, "极限敏感度 (σ_lim)", 'sigma_limit', 0.05, 0.5, 0.01)
        self._add_slider(grp_target, "宽度容忍", 'width_tol', 0.5, 5.0, 0.1)
        
        # 权重参数组
        grp_weight = ttk.LabelFrame(scroll_frame, text="权重参数", padding=5)
        grp_weight.pack(fill="x", pady=5, padx=5)
        self._add_slider(grp_weight, "位置权重 (W_pos)", 'w_pos', 0.1, 3.0, 0.1)
        self._add_slider(grp_weight, "稳定权重 (W_stab)", 'w_stab', 0.1, 3.0, 0.1)
        self._add_slider(grp_weight, "趋势权重 (W_trend)", 'w_trend', 0.1, 3.0, 0.1)
        
        # 动态特性组
        grp_dynamic = ttk.LabelFrame(scroll_frame, text="动态特性", padding=5)
        grp_dynamic.pack(fill="x", pady=5, padx=5)
        self._add_slider(grp_dynamic, "稳定敏感度 (K_stab)", 'k_stab', 0.5, 5.0, 0.1)
        self._add_slider(grp_dynamic, "趋势惩罚 (α_trend)", 'alpha_trend', 0.1, 3.0, 0.1)
        self._add_slider(grp_dynamic, "平滑系数 (β)", 'beta_ewma', 0.3, 0.99, 0.01)
        
        # 状态判定组
        grp_state = ttk.LabelFrame(scroll_frame, text="状态判定", padding=5)
        grp_state.pack(fill="x", pady=5, padx=5)
        self._add_slider(grp_state, "正常阈值", 'thresh_normal', 50.0, 90.0, 1.0)
        self._add_slider(grp_state, "迟滞带宽", 'hysteresis_band', 1.0, 10.0, 0.5)
        self._add_slider(grp_state, "最大罚分", 'max_penalty', 0.1, 1.0, 0.05)
        self._add_slider(grp_state, "初始滤波状态", 'initial_filter_state', 0.5, 1.0, 0.05)
        
        # 性能指标组
        grp_metrics = ttk.LabelFrame(scroll_frame, text="性能指标", padding=10)
        grp_metrics.pack(fill="x", pady=10, padx=5)
        
        metrics_grid = [
            ("平均健康度:", 'health_mean'),
            ("健康度标准差:", 'health_std'),
            ("相关性 (R):", 'correlation'),
            ("分类准确率:", 'accuracy'),
            ("过烧率:", 'overburn_rate'),
            ("欠烧率:", 'underburn_rate'),
        ]
        
        for label_text, var_name in metrics_grid:
            frame = ttk.Frame(grp_metrics)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label_text, width=12).pack(side="left")
            ttk.Label(frame, textvariable=self.vars[var_name], width=10, 
                     font=("Arial", 10, "bold")).pack(side="left")
        
        # 右侧：图表区域
        plot_frame = ttk.Frame(main_paned)
        main_paned.add(plot_frame, weight=1)
        
        # 创建三个子图的布局
        self.fig = Figure(figsize=(12, 9), dpi=100)
        gs = GridSpec(2, 2, figure=self.fig, height_ratios=[1.2, 1])
        
        # 子图1: 健康度全景图 (大图)
        self.ax_panorama = self.fig.add_subplot(gs[0, :])
        
        # 子图2: 混淆矩阵
        self.ax_confusion = self.fig.add_subplot(gs[1, 0])
        
        # 子图3: 相关性散点图
        self.ax_correlation = self.fig.add_subplot(gs[1, 1])
        
        self.fig.tight_layout(pad=3.0)
        
        # 嵌入 Tkinter
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_plot.draw()
        
        toolbar = NavigationToolbar2Tk(self.canvas_plot, plot_frame)
        toolbar.update()
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)
    
    def _add_slider(self, parent, label_text, var_name, min_val, max_val, resolution):
        """添加滑块控件"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=3)
        
        # 标签和数值显示
        lbl_frame = ttk.Frame(frame)
        lbl_frame.pack(fill="x")
        ttk.Label(lbl_frame, text=label_text, font=("Arial", 9)).pack(side="left")
        
        val_label = ttk.Label(lbl_frame, text=f"{self.vars[var_name].get():.2f}", 
                              width=6, anchor="e", font=("Arial", 9, "bold"))
        val_label.pack(side="right")
        
        # 滑块
        def on_change(val):
            val_label.configure(text=f"{float(val):.2f}")
            self._schedule_update()
        
        scale = ttk.Scale(frame, from_=min_val, to=max_val, 
                         variable=self.vars[var_name], command=on_change)
        scale.pack(fill="x")
        
        # 保存引用
        if not hasattr(self, 'slider_labels'):
            self.slider_labels = {}
        self.slider_labels[var_name] = val_label
    
    def _schedule_update(self):
        """延迟更新（防抖）"""
        if hasattr(self, '_update_after_id'):
            self.root.after_cancel(self._update_after_id)
        self._update_after_id = self.root.after(150, self._update_plots)
    
    def _auto_load_data(self):
        """自动加载默认数据"""
        self._load_data()
    
    def _load_data(self):
        """加载预测数据"""
        data_dir = self.data_path_var.get()
        health_file = os.path.join(data_dir, "health_data", "health_input_final.csv")
        
        if not os.path.exists(health_file):
            # 尝试其他可能的位置
            alt_file = os.path.join(root_dir, data_dir, "health_data", "health_input_final.csv")
            if os.path.exists(alt_file):
                health_file = alt_file
            else:
                messagebox.showerror("错误", f"找不到数据文件:\n{health_file}")
                return
        
        try:
            # 读取 CSV
            df = pd.read_csv(health_file)
            logging.info(f"加载数据: {len(df)} 条记录")
            
            # 提取预测分位数
            q_cols = ['Q10', 'Q25', 'Q50', 'Q75', 'Q90']
            if all(col in df.columns for col in q_cols):
                self.y_pred = df[q_cols].values
            else:
                # 如果只有 Q50，构造虚拟分位数
                if 'Q50' in df.columns:
                    q50 = df['Q50'].values
                    self.y_pred = np.column_stack([
                        q50 - 0.3, q50 - 0.15, q50, q50 + 0.15, q50 + 0.3
                    ])
                else:
                    messagebox.showerror("错误", "CSV 文件缺少分位数列")
                    return
            
            # 提取真实值
            if 'y_true' in df.columns:
                self.y_true = df['y_true'].values.reshape(-1, 1)
            elif 'True_Value' in df.columns:
                self.y_true = df['True_Value'].values.reshape(-1, 1)
            else:
                messagebox.showerror("错误", "CSV 文件缺少真实值列")
                return
            
            self.data_loaded = True
            self.vars['status'].set(f"已加载 {len(df)} 条记录")
            
            # 更新图表
            self._update_plots()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("加载错误", f"读取数据失败:\n{str(e)}")
    
    def _reset_params(self):
        """重置所有参数为默认值"""
        for key, val in self.DEFAULTS.items():
            if key in self.vars:
                self.vars[key].set(val)
                if key in self.slider_labels:
                    self.slider_labels[key].configure(text=f"{val:.2f}")
        self._update_plots()
    
    def _export_params(self):
        """导出当前参数到 JSON 文件"""
        if not self.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return
        
        params = {
            'health_mu': self.vars['mu'].get(),
            'health_sigma_left': self.vars['sigma_left'].get(),
            'health_sigma_right': self.vars['sigma_right'].get(),
            'health_sigma_limit': self.vars['sigma_limit'].get(),
            'health_width_tol': self.vars['width_tol'].get(),
            'health_W_pos': self.vars['w_pos'].get(),
            'health_W_stab': self.vars['w_stab'].get(),
            'health_W_trend': self.vars['w_trend'].get(),
            'health_k_stab': self.vars['k_stab'].get(),
            'health_alpha_trend': self.vars['alpha_trend'].get(),
            'health_beta_ewma': self.vars['beta_ewma'].get(),
            'health_thresh_normal': self.vars['thresh_normal'].get(),
            'health_hysteresis_band': self.vars['hysteresis_band'].get(),
            'health_max_penalty': self.vars['max_penalty'].get(),
            'health_initial_filter_state': self.vars['initial_filter_state'].get(),
        }
        
        output_path = "output/health_params_tuned.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        
        messagebox.showinfo("导出成功", f"参数已保存到:\n{output_path}")
    
    def _build_config(self):
        """构建配置对象"""
        return TunerConfig(
            health_mu=self.vars['mu'].get(),
            health_sigma_left=self.vars['sigma_left'].get(),
            health_sigma_right=self.vars['sigma_right'].get(),
            health_sigma_limit=self.vars['sigma_limit'].get(),
            health_width_tol=self.vars['width_tol'].get(),
            health_W_pos=self.vars['w_pos'].get(),
            health_W_stab=self.vars['w_stab'].get(),
            health_W_trend=self.vars['w_trend'].get(),
            health_k_stab=self.vars['k_stab'].get(),
            health_alpha_trend=self.vars['alpha_trend'].get(),
            health_beta_ewma=self.vars['beta_ewma'].get(),
            health_thresh_normal=self.vars['thresh_normal'].get(),
            health_hysteresis_band=self.vars['hysteresis_band'].get(),
            health_max_penalty=self.vars['max_penalty'].get(),
            health_initial_filter_state=self.vars['initial_filter_state'].get(),
        )
    
    def _update_plots(self):
        """更新所有图表"""
        if not self.data_loaded:
            return
        
        # 构建配置
        cfg = self._build_config()
        
        # 运行健康度模型
        model = HealthModel(cfg)
        self.current_results = model.analyze(self.y_pred, y_true=self.y_true)
        
        # 更新指标显示
        self._update_metrics()
        
        # 清空并重绘
        self.ax_panorama.clear()
        self.ax_confusion.clear()
        self.ax_correlation.clear()
        
        # 绘制三个子图
        self._draw_panorama(cfg)
        self._draw_confusion_matrix()
        self._draw_correlation()
        
        self.fig.tight_layout(pad=2.0)
        self.canvas_plot.draw()
    
    def _update_metrics(self):
        """更新性能指标显示"""
        r = self.current_results
        
        # 基本统计
        health_mean = np.mean(r['health_scores'])
        health_std = np.std(r['health_scores'])
        
        self.vars['health_mean'].set(f"{health_mean:.2f}")
        self.vars['health_std'].set(f"{health_std:.2f}")
        
        # 相关性
        if 'true_health_scores' in r:
            corr = np.corrcoef(r['health_scores'], r['true_health_scores'])[0, 1]
            self.vars['correlation'].set(f"{corr:.4f}")
        else:
            self.vars['correlation'].set("--")
        
        # 分类准确率
        if 'true_states' in r and 'pred_states' in r:
            accuracy = np.mean(r['true_states'] == r['pred_states'])
            self.vars['accuracy'].set(f"{accuracy:.2%}")
            
            # 过烧/欠烧率
            total = len(r['pred_states'])
            overburn_rate = np.sum(r['pred_states'] == 0) / total
            underburn_rate = np.sum(r['pred_states'] == 2) / total
            self.vars['overburn_rate'].set(f"{overburn_rate:.2%}")
            self.vars['underburn_rate'].set(f"{underburn_rate:.2%}")
        else:
            self.vars['accuracy'].set("--")
            self.vars['overburn_rate'].set("--")
            self.vars['underburn_rate'].set("--")
    
    def _draw_panorama(self, cfg):
        """绘制健康度全景图"""
        ax = self.ax_panorama
        r = self.current_results
        
        n = len(r['health_scores'])
        t = np.arange(n)
        
        # 创建双Y轴
        ax2 = ax.twinx()
        
        # 上半部分：BTP曲线
        btp_true = self.y_true.flatten() if self.y_true.ndim > 1 else self.y_true
        btp_pred = self.y_pred[:, 2]  # Q50
        
        ax2.plot(t, btp_true, 'b-', alpha=0.5, linewidth=1, label='BTP真实值')
        ax2.plot(t, btp_pred, 'g--', alpha=0.5, linewidth=1, label='BTP预测值')
        ax2.axhline(cfg.health_mu, color='gray', linestyle=':', alpha=0.5)
        ax2.set_ylabel('BTP位置 (m)', fontsize=10)
        ax2.legend(loc='upper left', fontsize=8)
        
        # 下半部分：健康度评分
        ax.fill_between(t, 0, r['health_scores'], alpha=0.3, color='blue')
        ax.plot(t, r['health_scores'], 'b-', linewidth=1.5, label='预测健康度')
        
        if 'true_health_scores' in r:
            ax.plot(t, r['true_health_scores'], 'r--', alpha=0.7, linewidth=1, label='真实健康度')
        
        # 阈值线
        thresh_normal = cfg.health_thresh_normal
        ax.axhline(thresh_normal, color='green', linestyle='--', alpha=0.7, 
                   label=f'正常阈值 ({thresh_normal:.0f})')
        
        # 状态背景色
        state_colors = {0: ('red', '过烧'), 2: ('orange', '欠烧')}
        for state, (color, label) in state_colors.items():
            mask = r['pred_states'] == state
            if np.any(mask):
                ax.fill_between(t, 0, 100, where=mask, alpha=0.1, color=color)
        
        ax.set_xlim(0, n)
        ax.set_ylim(0, 105)
        ax.set_xlabel('样本索引', fontsize=10)
        ax.set_ylabel('健康度评分', fontsize=10)
        ax.set_title('BTP健康度全景图 (final_test_btp_health_panorama)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _draw_confusion_matrix(self):
        """绘制混淆矩阵"""
        ax = self.ax_confusion
        r = self.current_results
        
        if 'true_states' not in r or 'pred_states' not in r:
            ax.text(0.5, 0.5, '缺少真值数据', ha='center', va='center', fontsize=12)
            ax.set_title('工况分类混淆矩阵', fontsize=11)
            return
        
        # 计算 3x3 混淆矩阵
        cm = confusion_matrix(r['true_states'], r['pred_states'], labels=[0, 1, 2])
        
        # 归一化（按行）
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_percent = cm.astype('float') / row_sums * 100
        
        # 绘制热力图
        im = ax.imshow(cm_percent, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        
        # 设置标签
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(self.STATE_LABELS, fontsize=9)
        ax.set_yticklabels(self.STATE_LABELS, fontsize=9)
        
        # 添加文本
        for i in range(3):
            for j in range(3):
                pct = cm_percent[i, j]
                cnt = cm[i, j]
                color = 'white' if pct > 40 else 'black'
                ax.text(j, i, f'{pct:.1f}%\n({cnt})', ha='center', va='center', 
                       fontsize=8, color=color)
        
        # 对角线高亮
        for i in range(3):
            rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                 edgecolor='gold', linewidth=2, alpha=0.8)
            ax.add_patch(rect)
        
        ax.set_xlabel('预测工况', fontsize=10)
        ax.set_ylabel('真实工况', fontsize=10)
        ax.set_title('工况分类混淆矩阵', fontsize=11, fontweight='bold')
    
    def _draw_correlation(self):
        """绘制相关性散点图"""
        ax = self.ax_correlation
        r = self.current_results
        
        if 'true_health_scores' not in r:
            ax.text(0.5, 0.5, '缺少真值数据', ha='center', va='center', fontsize=12)
            ax.set_title('健康度相关性', fontsize=11)
            return
        
        h_pred = r['health_scores']
        h_true = r['true_health_scores']
        
        # 散点图
        ax.scatter(h_true, h_pred, alpha=0.3, s=5, c='blue')
        
        # 对角线
        ax.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='理想情况')
        
        # 计算相关性
        corr = np.corrcoef(h_true, h_pred)[0, 1]
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('真实健康度', fontsize=10)
        ax.set_ylabel('预测健康度', fontsize=10)
        ax.set_title(f'健康度相关性 (R = {corr:.4f})', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(h_true, h_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 100, 100)
        ax.plot(x_line, p(x_line), 'g-', alpha=0.5, linewidth=2, label='趋势线')


# ==========================================
# 主程序入口
# ==========================================

def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    root = tk.Tk()
    app = InteractiveHealthTuner(root)
    
    # 设置窗口最小尺寸
    root.minsize(1200, 700)
    
    root.mainloop()


if __name__ == "__main__":
    main()