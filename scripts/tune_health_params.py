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
import sys as _sys
import logging
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import random

# 设置 Matplotlib 风格
plt.style.use('ggplot')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial'] # 适配中文显示
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 0. 尝试导入模型
# ==========================================
try:
    from btp.health_model import HealthModel
except ImportError:
    messagebox.showwarning("警告", "找不到 health_model.py。自动调优和计算功能将不可用。\n请确保该脚本在项目根目录下。")
    class HealthModel:
        def __init__(self, cfg): pass
        def analyze(self, p, y_true=None): return {'health_scores': np.zeros(len(p)), 'true_health_scores': np.zeros(len(p))}

# ==========================================
# 1. 动态配置类
# ==========================================

class InteractiveConfig:
    """动态配置类"""
    def __init__(self, **kwargs):
        # --- 默认参数 (保留你提供的特定数值) ---
        self.health_mu = 22.56
        self.health_sigma_left = 0.25
        self.health_sigma_right = 0.2
        self.health_sigma_limit = 0.15
        self.health_width_tol = 2.65
        
        self.health_k_stab = 2.0
        self.health_alpha_trend = 1.97
        
        self.health_W_pos = 2.79
        self.health_W_stab = 1.46
        self.health_W_trend = 1.99
        
        self.health_beta_ewma = 0.91
        self.volatility_window_size = 10
        
        for k, v in kwargs.items():
            setattr(self, k, v)

# ==========================================
# 2. 交互式调优应用程序
# ==========================================

class HealthVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BTP 健康度模型交互式调优台 (Auto-Tuner v2.2 - MAE Weighted)")
        self.root.geometry("1400x950")
        
        # --- 数据容器 ---
        self.df = None
        self.y_pred_full = None 
        self.y_true_full = None
        self.loaded_file_path = None
        
        # --- 默认值定义 (用于重置) ---
        self.DEFAULTS = {
            'mu': 22.56,
            'sigma_left': 0.25,
            'sigma_right': 0.2,
            'width_tol': 2.65,
            'w_pos': 2.79,
            'w_stab': 1.46,
            'w_trend': 2.0,       # 注意：你提供的代码里这里是 1.99 或 2，我取整便于滑块
            'alpha_trend': 1.97,
            'beta_ewma': 0.91,
            'k_stab': 2.0
        }
        
        # --- GUI 变量绑定 ---
        self.vars = {
            # 基础参数
            'mu': tk.DoubleVar(value=self.DEFAULTS['mu']),
            'sigma_left': tk.DoubleVar(value=self.DEFAULTS['sigma_left']),
            'sigma_right': tk.DoubleVar(value=self.DEFAULTS['sigma_right']),
            'width_tol': tk.DoubleVar(value=self.DEFAULTS['width_tol']),
            
            # 权重参数
            'w_pos': tk.DoubleVar(value=self.DEFAULTS['w_pos']),
            'w_stab': tk.DoubleVar(value=self.DEFAULTS['w_stab']),
            'w_trend': tk.DoubleVar(value=self.DEFAULTS['w_trend']),
            
            # 高级参数
            'alpha_trend': tk.DoubleVar(value=self.DEFAULTS['alpha_trend']),
            'beta_ewma': tk.DoubleVar(value=self.DEFAULTS['beta_ewma']),
            'k_stab': tk.DoubleVar(value=self.DEFAULTS['k_stab']),
            
            # 性能指标
            'mae': tk.StringVar(value="MAE: -"),
            'corr': tk.StringVar(value="Corr: -"),
            'bias': tk.StringVar(value="Bias: -")
        }
        
        self._init_ui()

    def _init_ui(self):
        # --- 顶部：加载数据栏 ---
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")
        
        ttk.Label(top_frame, text="分析报告文件 (.xlsx):").pack(side="left")
        self.path_entry = ttk.Entry(top_frame, width=60)
        self.path_entry.pack(side="left", padx=5)
        
        ttk.Button(top_frame, text="📂 选择 Excel 文件", command=self._load_file).pack(side="left")
        
        # [修改] 按钮文案，体现新的优化目标
        self.btn_optimize = ttk.Button(top_frame, text="✨ 自动调优 (Corr & MAE)", command=self._run_auto_tune, state="disabled")
        self.btn_optimize.pack(side="left", padx=10)

        ttk.Button(top_frame, text="↺ 重置默认", command=self._reset_to_defaults).pack(side="left", padx=5)

        # --- 主体：左右分栏 ---
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # === 左侧：参数控制面板 ===
        control_frame = ttk.LabelFrame(main_paned, text="参数控制 (实时响应)", padding=10, width=320)
        main_paned.add(control_frame, weight=0)
        
        # 1. 目标设定
        grp_target = ttk.LabelFrame(control_frame, text="目标设定 (H_pos)", padding=5)
        grp_target.pack(fill="x", pady=5)
        self._add_slider(grp_target, "目标值 (Mu)", 'mu', 20.0, 25.0, 0.1)
        self._add_slider(grp_target, "左侧容忍 (Sigma L)", 'sigma_left', 0.1, 2.0, 0.05)
        self._add_slider(grp_target, "右侧容忍 (Sigma R)", 'sigma_right', 0.1, 2.0, 0.05)
        
        # 2. 权重分配
        grp_weights = ttk.LabelFrame(control_frame, text="分量权重 (Weights)", padding=5)
        grp_weights.pack(fill="x", pady=5)
        self._add_slider(grp_weights, "位置权重 (W_pos)", 'w_pos', 0.0, 5.0, 0.1)
        self._add_slider(grp_weights, "稳定权重 (W_stab)", 'w_stab', 0.0, 5.0, 0.1)
        self._add_slider(grp_weights, "趋势权重 (W_trend)", 'w_trend', 0.0, 5.0, 0.1)
        
        # 3. 动态特性
        grp_dynamic = ttk.LabelFrame(control_frame, text="动态特性", padding=5)
        grp_dynamic.pack(fill="x", pady=5)
        self._add_slider(grp_dynamic, "宽度罚分 (Width Tol)", 'width_tol', 0.5, 5.0, 0.1)
        self._add_slider(grp_dynamic, "趋势惩罚 (Alpha Trend)", 'alpha_trend', 0.0, 3.0, 0.1)
        self._add_slider(grp_dynamic, "平滑系数 (Beta)", 'beta_ewma', 0.0, 0.99, 0.01)
        self._add_slider(grp_dynamic, "稳定敏感 (K Stab)", 'k_stab', 0.1, 5.0, 0.1)

        # 4. 指标显示区
        metrics_frame = ttk.LabelFrame(control_frame, text="性能指标 (Pred vs True)", padding=10)
        metrics_frame.pack(fill="x", pady=15)
        
        lbl_style = {"font": ("Arial", 11, "bold")}
        ttk.Label(metrics_frame, textvariable=self.vars['mae'], foreground="#d9534f", **lbl_style).pack(anchor="w")
        ttk.Label(metrics_frame, textvariable=self.vars['corr'], foreground="#5cb85c", **lbl_style).pack(anchor="w")
        ttk.Label(metrics_frame, textvariable=self.vars['bias'], foreground="#0275d8", **lbl_style).pack(anchor="w")

        # === 右侧：绘图区 ===
        plot_frame = ttk.Frame(main_paned)
        main_paned.add(plot_frame, weight=4)
        
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _add_slider(self, parent, label, var_name, min_val, max_val, resolution):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        
        lbl_frame = ttk.Frame(frame)
        lbl_frame.pack(fill="x")
        ttk.Label(lbl_frame, text=label, font=("Arial", 9)).pack(side="left")
        val_lbl = ttk.Label(lbl_frame, text=f"{self.vars[var_name].get():.2f}", width=5, anchor="e")
        val_lbl.pack(side="right")
        
        def on_slide(v):
            val_lbl.configure(text=f"{float(v):.2f}")
            self._update_plot_delayed()

        scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=self.vars[var_name], command=on_slide)
        scale.pack(fill="x")
        
        if not hasattr(self, 'scale_widgets'):
            self.scale_widgets = {}
        self.scale_widgets[var_name] = (scale, val_lbl)

    def _update_plot_delayed(self):
        if hasattr(self, '_after_id'):
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(100, self._calculate_and_draw)

    def _reset_to_defaults(self):
        """将所有参数重置为默认值"""
        for key, val in self.DEFAULTS.items():
            if key in self.vars:
                self.vars[key].set(val)
                if key in self.scale_widgets:
                    _, lbl = self.scale_widgets[key]
                    lbl.configure(text=f"{val:.2f}")
        self._calculate_and_draw()
        print("已重置所有参数为默认值。")

    def _load_file(self):
        file_path = filedialog.askopenfilename(
            title="选择分析报告",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, file_path)
        self.loaded_file_path = file_path
        
        try:
            # 尝试读取“预测详情”页签
            try:
                df = pd.read_excel(file_path, sheet_name="预测详情")
            except:
                df = pd.read_excel(file_path, sheet_name=0)
            
            # --- 💡 兼容性修复逻辑 ---
            # 1. 确定真实值列名 (新版叫 "真实值", 旧版叫 "真实_Q50")
            true_col = None
            if "真实值" in df.columns:
                true_col = "真实值"
            elif "真实_Q50" in df.columns:
                true_col = "真实_Q50"
            
            # 2. 检查必要列
            if "预测_Q50" not in df.columns or true_col is None:
                messagebox.showerror("错误", f"Excel 缺少必要列!\n需要: '预测_Q50' 和 ('真实值' 或 '真实_Q50')\n当前列名: {list(df.columns[:8])}")
                return

            # 3. 构建预测分布矩阵 (N, 5)
            q_names = ["预测_Q10", "预测_Q25", "预测_Q50", "预测_Q75", "预测_Q90"]
            pred_cols = [df[qn].values if qn in df.columns else df["预测_Q50"].values for qn in q_names]
            self.y_pred_full = np.column_stack(pred_cols)
            
            # 4. 构建真值分布矩阵 (N, 5) 
            # 现在的报告里真值是标量，我们需要将其广播成 (N, 5) 以适配 HealthModel 的接口
            y_true_scalar = df[true_col].values
            # 将标量平铺成 5 列，保持与预测端 Shape 对齐
            self.y_true_full = np.tile(y_true_scalar.reshape(-1, 1), (1, 5))
            
            self.df = df
            self.btn_optimize.config(state="normal")
            
            messagebox.showinfo("加载成功", f"数据已加载: {len(df)} 行\n识别到真值列: [{true_col}]")
            self._calculate_and_draw()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("加载异常", f"读取错误:\n{str(e)}")

    # ==========================================
    # 自动调优逻辑 (含 MAE 惩罚)
    # ==========================================
    def _run_auto_tune(self):
        if self.y_pred_full is None:
            return

        current_mu = self.vars['mu'].get()
        
        param_bounds = {
            'sigma_left': (0.1, 1.5),
            'sigma_right': (0.1, 1.5),
            'w_pos': (0.5, 3.5), # 稍微放宽一点范围
            'w_stab': (0.1, 2.5),
            'w_trend': (0.1, 2.5),
            'beta_ewma': (0.1, 0.98),
            'alpha_trend': (0.1, 2.5),
            'width_tol': (0.8, 3.0)
        }

        best_score = -9999
        best_params = {}
        
        # 计算当前基准分数
        current_score, c_corr, c_mae = self._evaluate_current_params()
        best_score = current_score
        
        n_iter = 200 # 增加迭代次数以找到更好的平衡点
        prog_win = tk.Toplevel(self.root)
        prog_win.title("优化中...")
        prog_bar = ttk.Progressbar(prog_win, length=300, mode='determinate', maximum=n_iter)
        prog_bar.pack(padx=20, pady=20)
        status_lbl = ttk.Label(prog_win, text="正在寻找最佳参数...")
        status_lbl.pack(pady=5)
        
        try:
            for i in range(n_iter):
                trial_params = {}
                for k, (low, high) in param_bounds.items():
                    trial_params[k] = random.uniform(low, high)
                
                trial_params['health_mu'] = current_mu
                
                # 获取分数
                score, t_corr, t_mae = self._evaluate_params(trial_params)
                
                if score > best_score:
                    best_score = score
                    best_params = trial_params.copy()
                    # 记录最佳的具体指标
                    best_corr = t_corr
                    best_mae = t_mae
                
                if i % 5 == 0:
                    prog_bar['value'] = i
                    prog_win.update()
            
            if best_params:
                msg = (
                    f"调优完成!\n"
                    f"综合得分: {best_score:.3f} (基准: {current_score:.3f})\n"
                    f"--------------------\n"
                    f"相关性 (Corr): {best_corr:.3f}\n"
                    f"绝对误差 (MAE): {best_mae:.2f}"
                )
                
                for k, v in best_params.items():
                    if k in self.vars:
                        self.vars[k].set(v)
                        if k in self.scale_widgets:
                            _, lbl = self.scale_widgets[k]
                            lbl.configure(text=f"{v:.2f}")

                self._calculate_and_draw()
                messagebox.showinfo("优化成功", msg)
            else:
                messagebox.showinfo("优化结果", "未找到综合得分更高的参数组合。")
                
        except Exception as e:
            messagebox.showerror("优化出错", str(e))
        finally:
            prog_win.destroy()

    def _evaluate_current_params(self):
        current_p = {k: v.get() for k, v in self.vars.items()}
        return self._evaluate_params(current_p)

    def _evaluate_params(self, params_dict):
        """
        核心评估函数：返回 (综合分数, Corr, MAE)
        综合分数 = Corr - (MAE * 0.04)
        """
        cfg = InteractiveConfig(
            health_mu = self.vars['mu'].get(),
            health_sigma_left = params_dict.get('sigma_left', 0.3),
            health_sigma_right = params_dict.get('sigma_right', 0.5),
            health_width_tol = params_dict.get('width_tol', 1.5),
            health_W_pos = params_dict.get('w_pos', 1.2),
            health_W_stab = params_dict.get('w_stab', 0.8),
            health_W_trend = params_dict.get('w_trend', 1.0),
            health_alpha_trend = params_dict.get('alpha_trend', 0.8),
            health_beta_ewma = params_dict.get('beta_ewma', 0.7),
            health_k_stab = self.vars['k_stab'].get()
        )
        
        model = HealthModel(cfg)
        try:
            res = model.analyze(self.y_pred_full, y_true=self.y_true_full)
            h_pred = res['health_scores']
            h_true = res.get('true_health_scores', np.zeros_like(h_pred))
            
            valid_mask = ~np.isnan(h_pred) & ~np.isnan(h_true)
            if np.sum(valid_mask) > 10:
                p_clean = h_pred[valid_mask]
                t_clean = h_true[valid_mask]
                
                # 1. 计算相关性
                corr = np.corrcoef(p_clean, t_clean)[0, 1]
                if np.isnan(corr): corr = 0
                
                # 2. 计算 MAE
                mae = np.mean(np.abs(p_clean - t_clean))
                
                # 3. [核心修改] 混合评分公式
                # 权重 0.04 意味着：MAE每增加 25，扣除 1.0 的相关性分数
                # 这迫使优化器必须降低 MAE 才能获得高分
                mae_penalty_weight = 0.04 
                score = corr - (mae * mae_penalty_weight)
                
                return score, corr, mae
            
            return -999.0, 0, 999
        except:
            return -999.0, 0, 999

    # ==========================================
    # 核心绘图
    # ==========================================
    def _calculate_and_draw(self):
        if self.y_pred_full is None:
            return
            
        cfg = InteractiveConfig(
            health_mu = self.vars['mu'].get(),
            health_sigma_left = self.vars['sigma_left'].get(),
            health_sigma_right = self.vars['sigma_right'].get(),
            health_width_tol = self.vars['width_tol'].get(),
            health_W_pos = self.vars['w_pos'].get(),
            health_W_stab = self.vars['w_stab'].get(),
            health_W_trend = self.vars['w_trend'].get(),
            health_alpha_trend = self.vars['alpha_trend'].get(),
            health_beta_ewma = self.vars['beta_ewma'].get(),
            health_k_stab = self.vars['k_stab'].get()
        )
        
        model = HealthModel(cfg)
        try:
            results = model.analyze(self.y_pred_full, y_true=self.y_true_full)
            h_pred = results['health_scores']
            h_true = results.get('true_health_scores', np.zeros_like(h_pred))
            
            valid_mask = ~np.isnan(h_pred) & ~np.isnan(h_true)
            if np.sum(valid_mask) > 0:
                mae = np.mean(np.abs(h_pred[valid_mask] - h_true[valid_mask]))
                bias = np.mean(h_pred[valid_mask]) - np.mean(h_true[valid_mask])
                if np.std(h_pred[valid_mask]) > 1e-4 and np.std(h_true[valid_mask]) > 1e-4:
                    corr = np.corrcoef(h_pred[valid_mask], h_true[valid_mask])[0, 1]
                else:
                    corr = 0.0
            else:
                mae, bias, corr = 0, 0, 0

            self.vars['mae'].set(f"MAE: {mae:.2f}")
            self.vars['corr'].set(f"Corr: {corr:.3f}")
            self.vars['bias'].set(f"Bias: {bias:.2f}")
            
            self.axs[0].clear()
            self.axs[1].clear()
            
            x_axis = np.arange(len(h_pred))
            
            self.axs[0].plot(x_axis, h_true, color='#2ca02c', alpha=0.6, label='True Health', linewidth=1.5)
            self.axs[0].plot(x_axis, h_pred, color='#d62728', alpha=0.9, linestyle='-', label='Pred Health', linewidth=1.5)
            self.axs[0].set_title("Health Score Comparison", fontsize=10)
            self.axs[0].legend(loc='upper right')
            self.axs[0].grid(True, alpha=0.3)
            self.axs[0].set_ylim(-5, 105)
            
            btp_pred = self.y_pred_full[:, 2] 
            btp_true = self.y_true_full[:, 2]
            
            self.axs[1].plot(x_axis, btp_true, color='gray', alpha=0.4, label='BTP True (Q50)')
            self.axs[1].plot(x_axis, btp_pred, color='#1f77b4', alpha=0.6, linestyle=':', label='BTP Pred (Q50)')
            self.axs[1].axhline(cfg.health_mu, color='black', linestyle='--', alpha=0.4)
            self.axs[1].axhline(cfg.health_mu - cfg.health_sigma_left, color='orange', linestyle=':', alpha=0.3)
            self.axs[1].axhline(cfg.health_mu + cfg.health_sigma_right, color='orange', linestyle=':', alpha=0.3)
            
            self.axs[1].set_title("BTP Signal Context", fontsize=10)
            self.axs[1].legend(loc='upper right', fontsize='small')
            self.axs[1].grid(True, alpha=0.3)
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Plotting Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HealthVisualizerApp(root)
    root.mainloop()
