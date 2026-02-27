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
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import logging
import queue
from btp.config import TrainConfig

# from main import run_full_pipeline # 假设 main 存在

class GUILogHandler(logging.Handler):
    """✅ 线程安全的GUI日志处理器"""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        ))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)

class BTPPredictionGUI:
    """
    🔧 分层容错GUI (DLinear Enhanced Version)
    """
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ 分层容错 BTP 预测系统 (DLinear v2.0)")
        
        # ✅ 日志消息队列（线程安全）
        self.log_queue = queue.Queue()
        
        # ✅ 创建默认配置实例以获取默认值
        self.default_cfg = TrainConfig()
        
        # ✅ 调整窗口大小
        self.root.geometry("1400x900")
        self.root.minsize(1100, 700)
        
        # ========== 顶部：文件选择区 ==========
        file_frame = tk.LabelFrame(root, text="1. 数据文件选择", padx=8, pady=5)
        file_frame.pack(fill="x", padx=8, pady=5)
        self.excel_path_var = tk.StringVar()
        tk.Label(file_frame, text="Excel:").grid(row=0, column=0, sticky="w", padx=(0,5))
        tk.Entry(file_frame, textvariable=self.excel_path_var, width=90).grid(row=0, column=1, padx=5)
        tk.Button(file_frame, text="📁 浏览", command=self.browse_file, width=8).grid(row=0, column=2, padx=3)
        tk.Label(file_frame, text="目标列:").grid(row=1, column=0, sticky="w", pady=3, padx=(0,5))
        self.target_col_var = tk.StringVar(value=self.default_cfg.target_column)
        tk.Entry(file_frame, textvariable=self.target_col_var, width=90).grid(row=1, column=1, padx=5, pady=3)

        # ========== 右侧：控制按钮 ==========
        btn_container = tk.Frame(file_frame)
        btn_container.grid(row=0, column=3, rowspan=2, padx=10, pady=5, sticky="ne")
        self.run_btn = tk.Button(
            btn_container,
            text="▶ 开始训练",
            command=self.start_training,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            width=14
        )
        self.run_btn.pack(pady=3)
        tk.Button(
            btn_container,
            text="📋 重置参数",
            command=self.reset_params,
            width=14
        ).pack(pady=3)
        tk.Button(
            btn_container,
            text="🗑️ 清空日志",
            command=self.clear_log,
            width=14
        ).pack(pady=3)

        # ========== 中间：左右分栏 ==========
        main_paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6)
        main_paned.pack(fill="both", expand=True, padx=5, pady=5)

        # ---------- 左侧：参数配置 ----------
        left_container = tk.Frame(main_paned)
        main_paned.add(left_container, minsize=480)
        param_frame = tk.LabelFrame(left_container, text="2. 参数配置", padx=5, pady=5)
        param_frame.pack(fill="both", expand=True, padx=3, pady=3)
        notebook = ttk.Notebook(param_frame)
        notebook.pack(fill="both", expand=True)
        self.params = {}

        # --- Tab 1: 质量控制 ---
        tab_quality = tk.Frame(notebook)
        notebook.add(tab_quality, text="🔧 质量")
        q_frame = tk.Frame(tab_quality, padx=10, pady=10)
        q_frame.pack(fill="both", expand=True)
        
        row = 0
        tk.Label(q_frame, text="窗口检测模式:", font=("Arial", 9, "bold")).grid(row=row, column=0, sticky="w", pady=5)
        self.params["validity_check_mode"] = tk.StringVar(value=self.default_cfg.validity_check_mode)
        modes = [("分层容错（推荐）", "layered"), ("严格模式", "strict"), ("宽松模式", "relaxed")]
        row += 1
        for i, (text, mode) in enumerate(modes):
            tk.Radiobutton(q_frame, text=text, variable=self.params["validity_check_mode"], value=mode).grid(
                row=row, column=0, columnspan=2, sticky="w", padx=20
            )
            row += 1
        ttk.Separator(q_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        params_quality = [
            ("核心特征阈值:", "min_valid_ratio_core", self.default_cfg.min_valid_ratio_core),
            ("辅助特征阈值:", "min_valid_ratio_aux", self.default_cfg.min_valid_ratio_aux),
            ("整体有效率:", "min_valid_ratio_overall", self.default_cfg.min_valid_ratio_overall),
            ("最大填充倍数:", "max_gap_fill_multiplier", self.default_cfg.max_gap_fill_multiplier),
        ]
        for label, key, default in params_quality:
            tk.Label(q_frame, text=label, anchor='w').grid(row=row, column=0, sticky="w", pady=3)
            self.params[key] = tk.DoubleVar(value=default)
            tk.Entry(q_frame, textvariable=self.params[key], width=12).grid(row=row, column=1, sticky="w", pady=3)
            row += 1
        self.params["strict_mode"] = tk.BooleanVar(value=self.default_cfg.strict_mode)
        tk.Checkbutton(q_frame, text="严格模式（验证集保留NaN）", variable=self.params["strict_mode"]).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=10
        )

        # --- Tab 2: 数据参数 ---
        tab_data = tk.Frame(notebook)
        notebook.add(tab_data, text="📊 数据")
        d_frame = tk.Frame(tab_data, padx=10, pady=10)
        d_frame.pack(fill="both")
        # [已简化] 移除了 multi_window_minutes, seq_len, overlap_ratio 等已删除的参数
        data_params = [
            ("原始序列长度:", "raw_seq_len", self.default_cfg.raw_seq_len, int),
            ("验证集比例 (Val):", "val_split", self.default_cfg.val_split, float),
            ("测试集比例 (Test):", "test_split", self.default_cfg.test_split, float),
        ]
        for i, (label, key, default, dtype) in enumerate(data_params):
            tk.Label(d_frame, text=label, anchor='w').grid(row=i, column=0, sticky="w", pady=6)
            if dtype == str:
                self.params[key] = tk.StringVar(value=default)
            elif dtype == int:
                self.params[key] = tk.IntVar(value=default)
            else:
                self.params[key] = tk.DoubleVar(value=default)
            tk.Entry(d_frame, textvariable=self.params[key], width=20).grid(row=i, column=1, sticky="w", pady=6)

        # --- Tab 3: 模型参数 (DLinear Updated) ---
        tab_model = tk.Frame(notebook)
        notebook.add(tab_model, text="🤖 DLinear模型")
        m_frame = tk.Frame(tab_model, padx=10, pady=10)
        m_frame.pack(fill="both")
        model_params = [
            ("隐藏层大小:", "hidden_size", self.default_cfg.hidden_size),
            ("Dropout概率:", "dropout", self.default_cfg.dropout),
            ("学习率 (LR):", "lr", self.default_cfg.lr),
            ("训练轮数 (Epochs):", "epochs", self.default_cfg.epochs),
            ("批次大小 (Batch):", "batch_size", self.default_cfg.batch_size),
            ("随机种子:", "seed", self.default_cfg.seed),
        ]
        for i, (label, key, default) in enumerate(model_params):
            tk.Label(m_frame, text=label, anchor='w').grid(row=i, column=0, sticky="w", pady=4)
            if isinstance(default, int):
                self.params[key] = tk.IntVar(value=default)
            else:
                self.params[key] = tk.DoubleVar(value=default)
            tk.Entry(m_frame, textvariable=self.params[key], width=20).grid(row=i, column=1, sticky="w", pady=4)

        row = len(model_params)
        ttk.Separator(m_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)

        # 🔧 RevIN
        row += 1
        self.params["enable_revin"] = tk.BooleanVar(value=self.default_cfg.enable_revin)
        tk.Checkbutton(m_frame, text="启用 RevIN (可逆归一化)", variable=self.params["enable_revin"]).grid(
            row=row, column=0, sticky="w", padx=5, pady=5
        )

        # --- Tab 4: CQR ---
        # [Modified] Removed Loss Weights Section
        tab_loss = tk.Frame(notebook)
        notebook.add(tab_loss, text="⚖️ CQR")
        l_frame = tk.Frame(tab_loss, padx=10, pady=10)
        l_frame.pack(fill="both")

        row = 0
        # 分组：CQR Config
        tk.Label(l_frame, text="CQR Calibration Targets", font=("Arial", 9, "bold")).grid(row=row, column=0, sticky="w", pady=5)
        row += 1
        cqr_params = [
            ("Inner Coverage:", "cqr_target_coverage_inner", self.default_cfg.cqr_target_coverage_inner),
            ("Outer Coverage:", "cqr_target_coverage_outer", self.default_cfg.cqr_target_coverage_outer),
        ]
        for label, key, default in cqr_params:
            tk.Label(l_frame, text=label, anchor='w').grid(row=row, column=0, sticky="w", pady=2)
            self.params[key] = tk.DoubleVar(value=default)
            tk.Entry(l_frame, textvariable=self.params[key], width=10).grid(row=row, column=1, sticky="w", pady=2)
            row += 1

        # --- Tab 5: BTP 阈值 ---
        tab_btp = tk.Frame(notebook)
        notebook.add(tab_btp, text="📏 BTP")
        b_frame = tk.Frame(tab_btp, padx=10, pady=10)
        b_frame.pack(fill="both")
        btp_params = [
            ("BTP 下界 (L_low):", "btp_L_low", self.default_cfg.btp_L_low),
            ("BTP 中心 (L_r):", "btp_L_r", self.default_cfg.btp_L_r),
            ("BTP 上界 (L_up):", "btp_L_up", self.default_cfg.btp_L_up),
            ("正常阈值 (H_norm):", "h_normal_min", self.default_cfg.h_normal_min),
            ("疑似阈值 (H_susp):", "h_suspect_min", self.default_cfg.h_suspect_min),
        ]
        for i, (label, key, default) in enumerate(btp_params):
            tk.Label(b_frame, text=label, anchor='w').grid(row=i, column=0, sticky="w", pady=6)
            self.params[key] = tk.DoubleVar(value=default)
            tk.Entry(b_frame, textvariable=self.params[key], width=20).grid(row=i, column=1, sticky="w", pady=6)

        # ---------- 右侧：运行日志 ----------
        right_container = tk.Frame(main_paned)
        main_paned.add(right_container, minsize=500)
        log_frame = tk.LabelFrame(right_container, text="3. 运行日志", padx=3, pady=3)
        log_frame.pack(fill="both", expand=True, padx=3, pady=3)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, state="disabled", wrap="word", font=("Consolas", 9), bg="#f5f5f5"
        )
        self.log_text.pack(fill="both", expand=True)
        self.status_var = tk.StringVar(value="就绪")
        tk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w', bg="#e0e0e0").pack(fill="x", side="bottom")

        self._setup_logging()
        self._start_log_polling()
        self.log_message("✅ 系统初始化完成 (DLinear v2.0)")

    def _setup_logging(self):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        gui_handler = GUILogHandler(self.log_queue)
        gui_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(gui_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

    def _start_log_polling(self):
        def poll():
            try:
                while True:
                    msg = self.log_queue.get_nowait()
                    self.log_text.config(state="normal")
                    self.log_text.insert("end", msg + "\n")
                    self.log_text.see("end")
                    self.log_text.config(state="disabled")
            except queue.Empty:
                pass
            finally:
                self.root.after(100, poll)
        poll()

    def log_message(self, msg: str):
        self.log_queue.put(msg)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx *.xls"), ("All Files", "*.*")])
        if file_path:
            self.excel_path_var.set(file_path)
            self.log_message(f"✅ 已选择文件: {os.path.basename(file_path)}")

    def clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, "end")
        self.log_text.config(state="disabled")

    def reset_params(self):
        """重置参数为默认值"""
        cfg = TrainConfig()
        
        # 质量控制
        self.params["validity_check_mode"].set(cfg.validity_check_mode)
        self.params["min_valid_ratio_core"].set(cfg.min_valid_ratio_core)
        self.params["min_valid_ratio_aux"].set(cfg.min_valid_ratio_aux)
        self.params["min_valid_ratio_overall"].set(cfg.min_valid_ratio_overall)
        self.params["max_gap_fill_multiplier"].set(cfg.max_gap_fill_multiplier)
        self.params["strict_mode"].set(cfg.strict_mode)
        
        # 数据 (已简化，移除了已删除的参数)
        self.params["raw_seq_len"].set(cfg.raw_seq_len)
        self.params["val_split"].set(cfg.val_split)
        self.params["test_split"].set(cfg.test_split)
        
        # 模型 (Updated)
        self.params["hidden_size"].set(cfg.hidden_size)
        self.params["dropout"].set(cfg.dropout)
        self.params["lr"].set(cfg.lr)
        self.params["epochs"].set(cfg.epochs)
        self.params["batch_size"].set(cfg.batch_size)
        self.params["seed"].set(cfg.seed)
        
        self.params["enable_revin"].set(cfg.enable_revin)
        
        # CQR
        self.params["cqr_target_coverage_inner"].set(cfg.cqr_target_coverage_inner)
        self.params["cqr_target_coverage_outer"].set(cfg.cqr_target_coverage_outer)
        
        # BTP
        self.params["btp_L_low"].set(cfg.btp_L_low)
        self.params["btp_L_r"].set(cfg.btp_L_r)
        self.params["btp_L_up"].set(cfg.btp_L_up)
        self.params["h_normal_min"].set(cfg.h_normal_min)
        self.params["h_suspect_min"].set(cfg.h_suspect_min)
        
        self.target_col_var.set(cfg.target_column)
        self.log_message("✅ 参数已重置")

    def start_training(self):
        excel_path = self.excel_path_var.get().strip()
        target_col = self.target_col_var.get().strip()
        if not excel_path or not os.path.isfile(excel_path):
            messagebox.showerror("错误", "请选择有效的 Excel 文件！")
            return
        try:
            from btp.config import make_timestamp
            output_dir = os.path.join("outputs", f"run_{make_timestamp()}")
            
            config = TrainConfig(
                excel_path=excel_path,
                target_column=target_col,
                output_dir=output_dir,
                
                # Quality
                validity_check_mode=self.params["validity_check_mode"].get(),
                min_valid_ratio_core=self.params["min_valid_ratio_core"].get(),
                min_valid_ratio_aux=self.params["min_valid_ratio_aux"].get(),
                min_valid_ratio_overall=self.params["min_valid_ratio_overall"].get(),
                max_gap_fill_multiplier=self.params["max_gap_fill_multiplier"].get(),
                strict_mode=self.params["strict_mode"].get(),
                
                # Data (已简化，移除了已删除的参数)
                raw_seq_len=self.params["raw_seq_len"].get(),
                val_split=self.params["val_split"].get(),
                test_split=self.params["test_split"].get(),
                
                # Model (Updated for DLinear)
                hidden_size=self.params["hidden_size"].get(),
                dropout=self.params["dropout"].get(),
                lr=self.params["lr"].get(),
                epochs=self.params["epochs"].get(),
                batch_size=self.params["batch_size"].get(),
                seed=self.params["seed"].get(),
                
                # Architecture
                enable_revin=self.params["enable_revin"].get(),
                
                # CQR
                cqr_target_coverage_inner=self.params["cqr_target_coverage_inner"].get(),
                cqr_target_coverage_outer=self.params["cqr_target_coverage_outer"].get(),
                
                # BTP
                btp_L_low=self.params["btp_L_low"].get(),
                btp_L_r=self.params["btp_L_r"].get(),
                btp_L_up=self.params["btp_L_up"].get(),
                h_normal_min=self.params["h_normal_min"].get(),
                h_suspect_min=self.params["h_suspect_min"].get(),
            )
            
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        self.run_btn.config(state="disabled", text="⏳ 训练中...")
        self.status_var.set("正在训练...")
        
        def run_thread():
            try:
                self.log_message(f"🚀 开始训练... 结果将保存至: {output_dir}")
                from main import run_full_pipeline
                run_full_pipeline(config)
                self.log_message("✅ 训练完成！")
                self.root.after(0, lambda: messagebox.showinfo("完成", "训练成功完成！"))
            except Exception as e:
                self.log_message(f"❌ 失败: {str(e)}")
                logging.exception("Training failed")
                self.root.after(0, lambda: messagebox.showerror("失败", f"训练出错:\n{str(e)}"))
            finally:
                self.root.after(0, lambda: self.run_btn.config(state="normal", text="▶ 开始训练"))
                self.root.after(0, lambda: self.status_var.set("就绪"))
        threading.Thread(target=run_thread, daemon=True).start()

def main():
    root = tk.Tk()
    BTPPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
