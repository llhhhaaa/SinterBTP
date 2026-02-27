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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import os
from typing import Dict

# [Modified] Import the correctly named losses
from btp.model import QuantileLoss

class Trainer:
    def __init__(self, model: nn.Module, cfg, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        # [修复] 训练配置不一致：加入 weight_decay
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(getattr(cfg, "weight_decay", 0.0))
        )
        self.criterion = QuantileLoss(cfg).to(device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
        self.history = {"train_loss": [], "val_loss": []}
        # [新增] 用于存储该模型在当前 Fold 上的原始残差序列
        self.last_residuals = None
        # [诊断] 存储每个 epoch 的诊断数据
        self.diagnostics_history = []

    def _should_diagnose(self, epoch: int) -> bool:
        """判断是否在当前 epoch 输出诊断信息"""
        # 第 1, 2, 3, 5, 10 个 epoch，之后每 10 个 epoch
        return epoch in [0, 1, 2, 4, 9] or (epoch + 1) % 10 == 0

    @torch.no_grad()
    def _compute_diagnostics(self, X: torch.Tensor, y: torch.Tensor, 
                              preds: torch.Tensor, epoch: int, split: str) -> dict:
        """
        计算训练诊断指标，帮助定位分布漂移和训练问题。
        
        诊断维度:
        1. 输入分布: X 的均值/标准差 (检测输入端漂移)
        2. 输出分布: pred vs true 的统计量 (检测预测偏差)
        3. 残差分析: 均值/标准差/偏度 (检测系统性偏移)
        4. 区间宽度: Q90-Q10 (检测区间坍缩)
        5. 梯度范数: 各层梯度大小 (检测梯度消失/爆炸)
        """
        diag = {"epoch": epoch + 1, "split": split}
        
        # --- 1. 输入分布 ---
        x_mean = X.mean().item()
        x_std = X.std().item()
        # 目标特征 (第0维) 的统计量
        x_target_mean = X[:, -1, 0].mean().item()
        x_target_std = X[:, -1, 0].std().item()
        diag["x_mean"] = round(x_mean, 5)
        diag["x_std"] = round(x_std, 5)
        diag["x_target_last_mean"] = round(x_target_mean, 5)
        diag["x_target_last_std"] = round(x_target_std, 5)
        
        # --- 2. 输出分布: pred Q50 vs true ---
        pred_q50 = preds[..., 2]  # (B, Steps)
        true_vals = y if y.ndim == 2 else y.squeeze(-1)  # (B, Steps)
        
        # 最终步 (forecast_steps 的最后一步，最重要)
        pred_last = pred_q50[:, -1]
        true_last = true_vals[:, -1]
        
        diag["pred_q50_mean"] = round(pred_q50.mean().item(), 5)
        diag["pred_q50_std"] = round(pred_q50.std().item(), 5)
        diag["true_mean"] = round(true_vals.mean().item(), 5)
        diag["true_std"] = round(true_vals.std().item(), 5)
        diag["pred_last_mean"] = round(pred_last.mean().item(), 5)
        diag["true_last_mean"] = round(true_last.mean().item(), 5)
        
        # --- 3. 残差分析 ---
        residuals = true_vals - pred_q50  # (B, Steps)
        res_mean = residuals.mean().item()
        res_std = residuals.std().item()
        res_abs_mean = residuals.abs().mean().item()
        
        # 偏度 (skewness) — 检测系统性偏移方向
        if res_std > 1e-8:
            res_skew = ((residuals - res_mean) ** 3).mean().item() / (res_std ** 3)
        else:
            res_skew = 0.0
        
        # 最终步残差
        res_last = true_last - pred_last
        res_last_mean = res_last.mean().item()
        
        diag["residual_mean"] = round(res_mean, 5)
        diag["residual_std"] = round(res_std, 5)
        diag["residual_abs_mean"] = round(res_abs_mean, 5)
        diag["residual_skew"] = round(res_skew, 3)
        diag["residual_last_step_mean"] = round(res_last_mean, 5)
        
        # --- 4. 区间宽度 ---
        pred_q10 = preds[..., 0]
        pred_q90 = preds[..., 4]
        interval_width = pred_q90 - pred_q10  # (B, Steps)
        
        diag["interval_width_mean"] = round(interval_width.mean().item(), 5)
        diag["interval_width_std"] = round(interval_width.std().item(), 5)
        diag["interval_width_min"] = round(interval_width.min().item(), 5)
        
        # 覆盖率: 真值落在 [Q10, Q90] 内的比例
        covered = ((true_vals >= pred_q10) & (true_vals <= pred_q90)).float()
        diag["coverage_q10_q90"] = round(covered.mean().item(), 4)
        
        # --- 5. 预测动态范围 vs 真值动态范围 ---
        pred_range = pred_q50.max().item() - pred_q50.min().item()
        true_range = true_vals.max().item() - true_vals.min().item()
        diag["pred_dynamic_range"] = round(pred_range, 5)
        diag["true_dynamic_range"] = round(true_range, 5)
        diag["range_ratio"] = round(pred_range / max(true_range, 1e-8), 3)
        
        return diag

    @torch.no_grad()
    def _log_gradient_norms(self) -> dict:
        """计算各层梯度范数"""
        grad_norms = {}
        total_norm = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                total_norm += norm ** 2
                # 只记录关键层
                if any(k in name for k in ['input_proj', 'head', 'transformer', 'core', 'revin']):
                    grad_norms[name] = round(norm, 6)
        grad_norms["total"] = round(total_norm ** 0.5, 6)
        return grad_norms

    def _format_diagnostics(self, diag: dict, grad_norms: dict = None) -> str:
        """格式化诊断信息为可读日志"""
        lines = []
        ep = diag["epoch"]
        sp = diag["split"]
        
        lines.append(f"  🔬 [诊断 E{ep:03d}/{sp}]")
        lines.append(f"     输入: mean={diag['x_mean']}, std={diag['x_std']}, "
                     f"target_last: mean={diag['x_target_last_mean']}, std={diag['x_target_last_std']}")
        lines.append(f"     预测Q50: mean={diag['pred_q50_mean']}, std={diag['pred_q50_std']} | "
                     f"真值: mean={diag['true_mean']}, std={diag['true_std']}")
        lines.append(f"     残差: mean={diag['residual_mean']}, std={diag['residual_std']}, "
                     f"skew={diag['residual_skew']}, |mean|={diag['residual_abs_mean']}")
        lines.append(f"     最终步偏差: pred={diag['pred_last_mean']}, true={diag['true_last_mean']}, "
                     f"gap={diag['residual_last_step_mean']}")
        lines.append(f"     区间[Q10,Q90]: width={diag['interval_width_mean']}±{diag['interval_width_std']}, "
                     f"min={diag['interval_width_min']}, coverage={diag['coverage_q10_q90']}")
        lines.append(f"     动态范围: pred={diag['pred_dynamic_range']}, true={diag['true_dynamic_range']}, "
                     f"ratio={diag['range_ratio']}")
        
        if grad_norms:
            top_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]
            grad_str = ", ".join(f"{k}={v}" for k, v in top_grads)
            lines.append(f"     梯度: {grad_str}")
        
        return "\n".join(lines)

    def train(self, data_dict: Dict, verbose=False) -> float:
        # 数据保留在 CPU，用 DataLoader 流水线预加载
        X_tr = torch.as_tensor(data_dict["X_tr"], dtype=torch.float32)
        y_tr = torch.as_tensor(data_dict["y_tr"], dtype=torch.float32)
        X_val = torch.as_tensor(data_dict["X_val"], dtype=torch.float32)
        y_val = torch.as_tensor(data_dict["y_val"], dtype=torch.float32)
        
        # [诊断] 训练开始前输出数据集统计
        logging.info(f"  📊 [数据统计] Train: X={list(X_tr.shape)}, y={list(y_tr.shape)} | "
                     f"Val: X={list(X_val.shape)}, y={list(y_val.shape)}")
        logging.info(f"     Train y: mean={y_tr.mean().item():.5f}, std={y_tr.std().item():.5f}, "
                     f"min={y_tr.min().item():.5f}, max={y_tr.max().item():.5f}")
        logging.info(f"     Val   y: mean={y_val.mean().item():.5f}, std={y_val.std().item():.5f}, "
                     f"min={y_val.min().item():.5f}, max={y_val.max().item():.5f}")
        
        # 检测 train/val 分布偏移
        y_shift = abs(y_tr.mean().item() - y_val.mean().item())
        if y_shift > 0.5 * y_tr.std().item():
            logging.warning(f"  ⚠️ [分布偏移] Train/Val y均值差={y_shift:.4f} > 0.5×std={0.5*y_tr.std().item():.4f}")
        
        # DataLoader: pin_memory + 多 worker 实现 CPU/GPU 流水线
        num_workers = int(getattr(self.cfg, "num_workers", 4))
        use_pin = (self.device.type == "cuda")
        train_dataset = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin,
            persistent_workers=(num_workers > 0),
            drop_last=False
        )
        
        # 验证集一次性搬到 GPU（数据量小）
        X_val_gpu = X_val.to(self.device)
        y_val_gpu = y_val.to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        total_batches = len(train_loader)
        # [修复] 学习率 warmup
        warmup_epochs = int(getattr(self.cfg, "warmup_epochs", 0))
        min_lr = float(getattr(self.cfg, "min_lr", 0.0))
        base_lr = float(getattr(self.cfg, "lr", 1e-3))
        
        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            grad_norms = None

            # [修复] warmup 调度：前 warmup_epochs 线性升温到 base_lr
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warmup_lr = min_lr + (base_lr - min_lr) * (epoch + 1) / warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr
            
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                # 每个 batch 搬到 GPU（DataLoader 已预加载到 pinned memory）
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    preds = self.model(X_batch)
                    loss = self.criterion(preds, y_batch)
                self.scaler.scale(loss).backward()
                
                # [诊断] 在诊断 epoch 的最后一个 batch 记录梯度
                if self._should_diagnose(epoch) and batch_idx == total_batches - 1:
                    self.scaler.unscale_(self.optimizer)
                    grad_norms = self._log_gradient_norms()
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_train_loss = epoch_loss / max(1, batch_count)
            self.history["train_loss"].append(avg_train_loss)
            avg_val_loss = self.validate(X_val_gpu, y_val_gpu)
            self.history["val_loss"].append(avg_val_loss)
            
            if verbose:
                logging.info(f"  📝 Epoch {epoch+1:03d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
            
            # 诊断输出已关闭以加速训练
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.patience: break
        
        # 加载最佳模型
        if hasattr(self, 'best_state_dict'):
            self.model.load_state_dict(self.best_state_dict)
        
        return best_val_loss

    def validate(self, X_val, y_val, batch_size=256):
        """分批验证，避免显存溢出"""
        self.model.eval()
        total_loss = 0.0
        n_samples = len(X_val)
        n_batches = (n_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_samples)
                X_batch = X_val[start:end]
                y_batch = y_val[start:end]

                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    preds = self.model(X_batch)
                    loss = self.criterion(preds, y_batch)
                total_loss += loss.item() * (end - start)

        return total_loss / n_samples

    def predict(self, X, y_true_scaled=None, mc_samples=1):
        """
        预测函数：支持 MC Dropout 多次采样平均

        Args:
            X: 输入数据
            y_true_scaled: 真值（可选，用于计算残差）
            mc_samples: MC采样次数，1表示普通推理，>1表示多次采样平均

        Returns:
            preds: (B, Steps, Quantiles) 预测结果
        """
        self.model.eval()
        X = X.to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                # 直接调用 forward，mc_samples>1 时自动返回平均值
                if hasattr(self.model, 'forward') and 'mc_samples' in self.model.forward.__code__.co_varnames:
                    show_progress = mc_samples > 1  # 多次采样时显示进度条
                    preds = self.model(X, mc_samples=mc_samples, show_progress=show_progress)
                else:
                    preds = self.model(X)

        preds_np = preds.cpu().numpy()

        # 保存残差用于后续分析
        if y_true_scaled is not None:
            y_t = torch.as_tensor(y_true_scaled, dtype=torch.float32, device=self.device)
            if y_t.ndim == 3:
                y_t = y_t.squeeze(-1)
            pred_q50 = preds[..., 2]  # Q50
            self.last_residuals = (y_t - pred_q50).cpu().numpy()

        return preds_np

    def get_model_state(self):
        return self.model.state_dict()
