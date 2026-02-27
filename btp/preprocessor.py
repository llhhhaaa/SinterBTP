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


# preprocessor.py
# (精简版 - 原始高频序列，已移除粒化功能)

import re
import os
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from scipy.interpolate import CubicSpline  
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
from tqdm import tqdm
from numba import jit, prange

from btp.utils import pretty_title, normalize_col_name

# =========================================================
# 1. Numba 加速核心计算函数 (已移除粒化相关函数)
# =========================================================



@jit(nopython=True, cache=True)
def _check_time_continuity(timestamps_sec: np.ndarray, max_gap: float) -> bool:
    if timestamps_sec.size < 2:
        return False
    for i in range(1, timestamps_sec.size):
        if timestamps_sec[i] - timestamps_sec[i-1] > max_gap:
            return False
    return True


@jit(nopython=True, cache=True)
def _check_validity_relaxed_numba(data: np.ndarray, min_ratio: float) -> bool:
    valid_count = np.sum(np.isfinite(data))
    total = data.size
    return (valid_count / total) >= min_ratio

def _extract_spline_features(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    物理引擎：使用三次样条插值从风箱温度序列中提取形状特征
    """
    # 1. 定义物理坐标 (风箱编号作为 X 轴)
    x_nodes = np.array([15, 17, 18, 19, 20, 21, 22, 23, 24])
    temp_cols = [normalize_col_name(f"{i}#风箱温度({side})") for i in x_nodes]
    
    # 提取数据矩阵
    y_matrix = df[temp_cols].values
    
    # 准备存储容器
    calc_pos = []
    calc_temp = []
    calc_slope = []
    calc_auc = []

    # 遍历每一行
    for i in range(len(y_matrix)):
        y = y_matrix[i]
        mask = np.isfinite(y)
        
        # 鲁棒性检查：如果有效点太少，返回空值
        if np.sum(mask) < 4:
            calc_pos.append(np.nan); calc_temp.append(np.nan)
            calc_slope.append(np.nan); calc_auc.append(np.nan)
            continue
            
        try:
            # 执行三次样条插值
            cs = CubicSpline(x_nodes[mask], y[mask])
            
            # 在高密度空间寻找最大值 (BTP)
            x_dense = np.linspace(15, 24, 5000)
            y_dense = cs(x_dense)
            peak_idx = np.argmax(y_dense)
            
            p_pos = x_dense[peak_idx]
            p_temp = y_dense[peak_idx]
            
            # 计算 BTP 位置的一阶导数（斜率）和 15-24 区间的积分
            p_slope_pre = (cs(p_pos) - cs(p_pos - 1)) / 1.0 
            p_auc = cs.integrate(15, 24)
            
            calc_pos.append(p_pos); calc_temp.append(p_temp)
            calc_slope.append(p_slope_pre); calc_auc.append(p_auc)
        except:
            calc_pos.append(np.nan); calc_temp.append(np.nan)
            calc_slope.append(np.nan); calc_auc.append(np.nan)

    return pd.DataFrame({
        f"{side}侧_计算BTP位置": calc_pos,
        f"{side}侧_计算BTP温度": calc_temp,
        f"{side}侧_BTP斜率": calc_slope,
        f"{side}侧_BTP积分面积": calc_auc
    }, index=df.index)

# =========================================================
# 2. 批量序列构建函数 (单一尺度)
# =========================================================
def _build_sequences_batch(
    batch_indices: np.ndarray,
    X_combined_all: np.ndarray,
    timestamps_all: np.ndarray,
    y_combined_all: np.ndarray,
    scale_param: Dict,
    tgt_w: int,
    K: int,
    granulated_feat_dim: int,  # 保留参数以兼容调用签名，但不再使用
    max_gap_sec: float,
    enable_delta: bool,
    sampling_sec: float = 5.0,
    forecast_steps: int = 5,
    prediction_offset: int = 0,
    y_raw_all: np.ndarray = None,
    max_future_index: Optional[int] = None
) -> Tuple:
    """
    构建训练序列（已移除粒化功能，直接使用原始序列）
    
    返回:
        X_macro_batch: 占位数组 (B, 1, 1)，保持接口兼容
        X_raw_batch: 原始序列 (B, raw_seq_len, feat_dim)
        y_batch: 预测目标 (B, forecast_steps)
        anchor_batch: 锚点值 (B, 1)
        y_obs_batch: 观测窗 (B, tgt_w)
        valid_count: 有效样本数
    """
    max_samples = len(batch_indices)
    rows_per_step = int(round(scale_param["w_rows"]))

    raw_seq_len = scale_param["buf_needed"]
    raw_feat_dim = X_combined_all.shape[1]

    # X_macro_batch 作为占位数组，保持返回格式兼容
    X_macro_batch = np.zeros((max_samples, 1, 1), dtype=np.float32)
    X_raw_batch = np.zeros((max_samples, raw_seq_len, raw_feat_dim), dtype=np.float32)
    
    y_batch = np.zeros((max_samples, forecast_steps), dtype=np.float32)
    anchor_batch = np.zeros((max_samples, 1), dtype=np.float32)
    y_obs_batch = np.full((max_samples, tgt_w), np.nan, dtype=np.float32)
    
    valid_count = 0
    timestamps_sec = timestamps_all.astype('datetime64[s]').astype(np.float64)
    total_len = len(y_combined_all)
    # [防止标签泄露] 训练/验证/测试数据构建时限制未来索引上界
    # max_future_index 为允许的最大索引（含），默认不限制
    future_index_cap = max_future_index if max_future_index is not None else total_len - 1

    for i in batch_indices:
        input_anchor = i - tgt_w
        # 添加 prediction_offset：从输入序列末尾跳过 prediction_offset 行后开始预测
        future_indices = [i + prediction_offset + (s+1) * rows_per_step for s in range(forecast_steps)]
        
        # 1) 不能超过数据长度
        if future_indices[-1] >= total_len:
            continue
        # 2) 不能超过分段上界（用于避免训练标签泄露到验证/测试）
        if future_indices[-1] > future_index_cap:
            continue
        if i < raw_seq_len: continue
            
        future_vals = y_combined_all[future_indices]
        if np.any(np.isnan(future_vals)): continue
            
        # 抽取原始序列 (Raw Sequence)
        raw_slice = X_combined_all[i - raw_seq_len : i, :]
        raw_ts = timestamps_sec[i - raw_seq_len : i]
        if not _check_time_continuity(raw_ts, max_gap_sec): continue
        if not _check_validity_relaxed_numba(raw_slice, 0.8): continue
        
        X_raw_batch[valid_count] = raw_slice.astype(np.float32)
        
        # 标签与锚点
        anchor_scalar = np.nanmean(y_combined_all[max(0, i-1):i+1]) if enable_delta else 0.0
        y_batch[valid_count] = future_vals - anchor_scalar if enable_delta else future_vals
        anchor_batch[valid_count] = anchor_scalar
        
        # 观测窗使用原始未平滑的 y（用于可视化真实波动范围）
        obs_source = y_raw_all if y_raw_all is not None else y_combined_all
        y_obs_batch[valid_count] = obs_source[input_anchor : i].astype(np.float32)
        valid_count += 1
        
    return (
        X_macro_batch[:valid_count],
        X_raw_batch[:valid_count],
        y_batch[:valid_count],
        anchor_batch[:valid_count],
        y_obs_batch[:valid_count],
        valid_count
    )

# =========================================================
# 3. DataPreprocessor 主类
# =========================================================

class DataPreprocessor:
    """🚀 极致精简版预处理器 (Single Scale Only)"""

    def __init__(self, config):
        self.cfg = config
        self.scaler_core = StandardScaler()
        self.scaler_aux = StandardScaler()
        self.pca_aux = PCA(n_components=0.95, random_state=self.cfg.seed)
        self.scaler_y = StandardScaler()

        self.target_col = "BTP_pos_target"
        self.core_cols: List[str] = []
        self.aux_cols: List[str] = []
        self.input_cols: List[str] = []
        self.raw_input_cols: List[str] = []
        self.clip_limits: Dict[str, Tuple[float, float]] = {}

    def get_cache_path(self) -> str:
        """根据当前配置生成哈希值作为缓存文件名"""
        # 挑选影响数据预处理和切分的关键配置
        cache_params = {
            "raw_seq_len": self.cfg.raw_seq_len,
            "forecast_steps": self.cfg.forecast_steps,
            "prediction_offset": self.cfg.prediction_offset,
            "target_column": self.cfg.target_column,
            "cv_n_splits": self.cfg.cv_n_splits,
            "val_split": self.cfg.val_split,
            "test_split": self.cfg.test_split,
            "seed": self.cfg.seed,
            "target_smooth_span": getattr(self.cfg, "target_smooth_span", 0),
            "enable_delta_forecast": self.cfg.enable_delta_forecast,
            "optimize_gap_size": self.cfg.optimize_gap_size
        }
        param_str = str(sorted(cache_params.items()))
        hash_val = hashlib.md5(param_str.encode()).hexdigest()
        
        cache_dir = getattr(self.cfg, "CACHE_DIR", "data/cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"pregen_data_{hash_val}.pkl")

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pretty_title("Step 2  特征工程 (物理特征引擎版)")

        # 记录原始 Excel 输入列（仅用于异常值截断）
        original_cols = list(df.columns)
        
        # 1. 自动提取南北两侧物理特征
        logging.info("[Physics] 正在执行三次样条插值提取物理描述符...")
        df_south = _extract_spline_features(df, side='南')
        df_north = _extract_spline_features(df, side='北')
        df = pd.concat([df, df_south, df_north], axis=1)
        
        # 2. 设置目标值
        target_source_col = self.cfg.target_column
        df[self.target_col] = pd.to_numeric(df[target_source_col], errors="coerce")

        # 仅对原始 Excel 传感器列做截断（排除时间列与目标列）
        self.raw_input_cols = [
            c for c in original_cols if c not in ["时间", target_source_col]
        ]
        
        # 3. 物理导数特征
        dt = df["时间"].diff().dt.total_seconds()
        prefix = "南" if "南" in target_source_col else "北"
        df["BTP_vel"] = df[f"{prefix}侧_计算BTP位置"].diff() / dt
        df["BTP_acc"] = df["BTP_vel"].diff() / dt

        # 4. 定义 Core Columns (Enhanced Transformer 使用，包含衍生特征)
        self.core_cols = [
            "南侧_计算BTP位置", "南侧_计算BTP温度", "南侧_BTP斜率", "南侧_BTP积分面积",
            "北侧_计算BTP位置", "北侧_计算BTP温度", "北侧_BTP斜率", "北侧_BTP积分面积",
            "机速检测值", "料层厚度平均值", "南侧风箱负压", "BTP_vel", "BTP_acc"
        ]
        
        # 4.1 定义基线模型专用的 Core Columns (不含衍生特征，只用原始风箱温度)
        # 衍生特征列表 (三次样条插值计算得出，基线模型禁用)
        self.spline_derived_cols = [
            "南侧_计算BTP位置", "南侧_计算BTP温度", "南侧_BTP斜率", "南侧_BTP积分面积",
            "北侧_计算BTP位置", "北侧_计算BTP温度", "北侧_BTP斜率", "北侧_BTP积分面积",
            "BTP_vel", "BTP_acc"  # 这些也依赖于计算BTP位置
        ]
        
        # 基线模型使用的原始风箱温度列 (15#-24#)
        from btp.utils import normalize_col_name
        self.baseline_temp_cols = []
        for side in ['南', '北']:
            for box_id in [15, 17, 18, 19, 20, 21, 22, 23, 24]:
                col_name = normalize_col_name(f"{box_id}#风箱温度({side})")
                if col_name in df.columns:
                    self.baseline_temp_cols.append(col_name)
        
        # 基线模型核心特征 = 原始风箱温度 + BTP位置(保留用于物理路径) + 非衍生的工艺参数
        # 注意：BTP位置是三次样条计算的，但作为目标相关特征需要保留
        self.baseline_core_cols = (
            self.baseline_temp_cols +
            ["南侧_计算BTP位置", "北侧_计算BTP位置"] +  # 保留BTP位置用于物理路径
            ["机速检测值", "料层厚度平均值", "南侧风箱负压"]
        )
        
        # 5. 定义 Aux Columns
        ignore_keywords = ["风箱温度", "设定值", "上限", "下限", "报警", "位置", "温度"]
        exclude_set = set(self.core_cols + [self.target_col, "时间"])
        
        self.aux_cols = []
        for c in df.columns:
            if c in exclude_set: continue
            if any(k in c for k in ignore_keywords): continue
            if pd.api.types.is_numeric_dtype(df[c]):
                self.aux_cols.append(c)
        
        self.input_cols = self.core_cols + self.aux_cols
        
        # 基线模型输入列 (不含衍生特征)
        self.baseline_input_cols = self.baseline_core_cols + self.aux_cols
        
        # [消融] 如果关闭拟合模块，退化为基线特征集（不含三次样条衍生特征）
        if not getattr(self.cfg, "enable_fitting_module", True):
            logging.info("[Ablation] enable_fitting_module=False → 使用基线特征集 (不含三次样条衍生特征)")
            self.core_cols = self.baseline_core_cols
            self.input_cols = self.baseline_input_cols
        
        for c in self.input_cols + [self.target_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
        logging.info(f"[Features] 物理核心特征: {len(self.core_cols)}, 辅助特征: {len(self.aux_cols)}")
        return df

    def _fill_missing_values(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        if len(cols) == 0:
            return df
        df[cols] = df[cols].ffill().bfill()
        col_means = df[cols].mean()
        df[cols] = df[cols].fillna(col_means)
        return df

    def _compute_clip_limits(self, df: pd.DataFrame, cols: List[str]) -> None:
        for c in cols:
            q_low, q_high = df[c].quantile(0.01), df[c].quantile(0.99)
            if np.isfinite(q_low) and np.isfinite(q_high) and q_high > q_low:
                self.clip_limits[c] = (float(q_low), float(q_high))

    def _apply_clip_limits(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for c in cols:
            if c in self.clip_limits:
                lower, upper = self.clip_limits[c]
                df[c] = df[c].clip(lower=lower, upper=upper)
        return df

    def _get_raw_clip_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in self.raw_input_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    def _final_safety_net(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if len(cols) == 0:
            return df
        if df[cols].isnull().any().any():
            df[cols] = df[cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df


    def _smart_fill_small_gaps(self, df: pd.DataFrame, cols: List[str], time_col: str, max_gap_sec: float) -> pd.DataFrame:
        df = df.copy()
        time_diffs = df[time_col].diff().dt.total_seconds()
        for c in cols:
            mask_null = df[c].isnull()
            if not mask_null.any(): continue
            can_fill = mask_null & (time_diffs <= max_gap_sec)
            if can_fill.any():
                df.loc[can_fill, c] = df[c].fillna(method='ffill')[can_fill]
        return df

    def _internal_parallel_build(self, start_idx, end_idx, X_all, y_all, time_all, scale_param, tgt_w, K, granulated_feat_dim, max_gap_sec, sampling_sec, set_name, y_raw_all=None, max_future_index=None):
        num_candidates = end_idx - start_idx
        n_workers = min(os.cpu_count() or 4, 8)
        batch_size = max(100, num_candidates // (n_workers * 4))
        all_indices = np.arange(start_idx, end_idx)
        batches = [all_indices[i:i+batch_size] for i in range(0, len(all_indices), batch_size)]
        
        executor = ProcessPoolExecutor(max_workers=n_workers)
        prediction_offset = getattr(self.cfg, 'prediction_offset', 0)
        worker_func = partial(
            _build_sequences_batch, X_combined_all=X_all, timestamps_all=time_all, y_combined_all=y_all,
            scale_param=scale_param, tgt_w=tgt_w, K=K, granulated_feat_dim=granulated_feat_dim,
            max_gap_sec=max_gap_sec, enable_delta=self.cfg.enable_delta_forecast,
            forecast_steps=self.cfg.forecast_steps, prediction_offset=prediction_offset,
            sampling_sec=sampling_sec, y_raw_all=y_raw_all, max_future_index=max_future_index
        )
        
        futures = [executor.submit(worker_func, batch) for batch in batches]
        X_mac_list, X_raw_list, y_list, anchor_list, yobs_list, total_valid = [], [], [], [], [], 0

        pbar = tqdm(as_completed(futures), total=len(futures), desc=f"[{set_name}] 构建序列", ncols=100)
        for future in pbar:
            xb_mac, xb_raw, yb, anc_b, yobs_b, vc = future.result()
            if vc > 0:
                X_mac_list.append(xb_mac); X_raw_list.append(xb_raw)
                y_list.append(yb); anchor_list.append(anc_b); yobs_list.append(yobs_b)
                total_valid += vc
        
        executor.shutdown()
        if total_valid == 0: raise ValueError(f"{set_name}无有效样本")
        
        return np.vstack(X_mac_list), np.vstack(X_raw_list), np.vstack(y_list), np.vstack(anchor_list), np.vstack(yobs_list)

    def yield_rolling_folds(self, df: pd.DataFrame, sampling_sec: float):
        """
        时序版交叉验证，每折都有独立的 test 集
        
        时间轴: [0 ========================== 100%]
        
        Fold 1: [Train 50%    ][Val 10%][Test 10%]
        Fold 2: [Train 60%        ][Val 10%][Test 10%]
        Fold 3: [Train 70%            ][Val 10%][Test 10%]
        ...
        
        Args:
            df: 按时间排序的数据
            sampling_sec: 采样间隔（秒）
            
        Yields:
            (fold_idx, data_dict) for each fold
        """
        n_folds = self.cfg.cv_n_splits
        val_ratio = getattr(self.cfg, 'val_ratio', 0.1)
        test_ratio = getattr(self.cfg, 'test_ratio', 0.1)
        
        pretty_title(f"Step 3-5 [CV] 时序滚动交叉验证 (Folds={n_folds}, Val={val_ratio:.0%}, Test={test_ratio:.0%})")
        
        max_gap_sec = sampling_sec * self.cfg.max_gap_fill_multiplier
        n_total = len(df)
        all_numeric_cols = self.input_cols + [self.target_col]
        raw_clip_cols = self._get_raw_clip_cols(df)
        
        # 使用 raw_seq_len 计算 buffer (不再使用粒化参数)
        raw_seq_len = int(self.cfg.raw_seq_len)
        forecast_steps = int(self.cfg.forecast_steps)
        
        # scale_param 用于 _internal_parallel_build，保持兼容
        scale_param = {"buf_needed": raw_seq_len, "w_rows": 1, "step_rows": 1}
        tgt_w = 1  # 不再使用粒化窗口
        K = 1      # 不再使用粒化序列，K=1 表示单一时间步
        total_buf = raw_seq_len + forecast_steps
        
        # gap 应该至少覆盖 prediction_offset + forecast_steps，防止数据泄漏
        prediction_offset = int(getattr(self.cfg, "prediction_offset", 0))
        gap_min = prediction_offset + forecast_steps
        gap_rows = gap_min if self.cfg.optimize_gap_size else max(total_buf, gap_min)
        logging.info(
            f"[CV] gap_rows={gap_rows} (prediction_offset={prediction_offset}, forecast_steps={forecast_steps}, total_buf={total_buf})"
        )
        
        # 计算固定的 val 和 test 大小（基于总数据量）
        val_size = int(n_total * val_ratio)
        test_size = int(n_total * test_ratio)
        min_train_ratio = 0.1  # 至少 10% 数据用于训练 (支持8折CV)
        
        logging.info(f"[CV] 数据总量: {n_total}, Val大小: {val_size}, Test大小: {test_size}")
        
        for fold_idx in range(n_folds):
            # 计算这一折的数据范围
            # fold_end 从 70% 逐渐增加到 100%
            if n_folds > 1:
                fold_end_ratio = 0.35 + (fold_idx / (n_folds - 1)) * 0.65
            else:
                fold_end_ratio = 1.0
            fold_end = int(n_total * fold_end_ratio)
            
            # 测试集：最后 test_size
            test_start = fold_end - test_size
            test_end = fold_end
            
            # 验证集：测试集之前的 val_size
            val_start = test_start - val_size
            val_end = test_start
            
            # 训练集：验证集之前的所有数据
            train_start = 0
            train_end = val_start - gap_rows  # 留出 gap 防止数据泄漏
            
            # 确保训练集有足够数据
            if train_end < int(n_total * min_train_ratio):
                logging.warning(f"[CV] Fold {fold_idx+1} 训练集不足 ({train_end}/{int(n_total * min_train_ratio)}), 跳过")
                continue
            
            logging.info(f"[CV] Fold {fold_idx+1}: Train[0:{train_end}], Val[{val_start}:{val_end}], Test[{test_start}:{test_end}]")
            
            # 切分数据
            df_train = df.iloc[train_start:train_end].copy()
            df_val_with_buf = df.iloc[max(0, val_start - total_buf):val_end].copy()
            df_test_with_buf = df.iloc[max(0, test_start - total_buf):test_end].copy()
            
            # 训练集预处理
            df_train = self._smart_fill_small_gaps(df_train, self.input_cols, "时间", max_gap_sec)
            df_train = self._fill_missing_values(df_train, all_numeric_cols)
            local_clip_limits: Dict[str, Tuple[float, float]] = {}
            for c in raw_clip_cols:
                q_low, q_high = df_train[c].quantile(0.01), df_train[c].quantile(0.99)
                if np.isfinite(q_low) and np.isfinite(q_high) and q_high > q_low:
                    local_clip_limits[c] = (float(q_low), float(q_high))
                    df_train[c] = df_train[c].clip(lower=q_low, upper=q_high)
            df_train = self._final_safety_net(df_train, all_numeric_cols)

            def _clean_fold_df(d_):
                d_ = self._smart_fill_small_gaps(d_, self.input_cols, "时间", max_gap_sec)
                d_ = self._fill_missing_values(d_, all_numeric_cols)
                for c in raw_clip_cols:
                    if c in local_clip_limits:
                        lower, upper = local_clip_limits[c]
                        d_[c] = d_[c].clip(lower=lower, upper=upper)
                d_ = self._final_safety_net(d_, all_numeric_cols)
                return d_

            df_val_with_buf = _clean_fold_df(df_val_with_buf)
            df_test_with_buf = _clean_fold_df(df_test_with_buf)

            X_tr_mtx = self.scaler_core.fit_transform(df_train[self.core_cols].values)
            X_val_buf_mtx = self.scaler_core.transform(df_val_with_buf[self.core_cols].values)
            X_test_buf_mtx = self.scaler_core.transform(df_test_with_buf[self.core_cols].values)
            
            # [目标平滑]
            smooth_span = getattr(self.cfg, "target_smooth_span", 0)
            
            y_tr_target = df_train[self.target_col].values
            y_tr_raw_cv = y_tr_target.copy()
            if smooth_span > 0:
                y_tr_target = pd.Series(y_tr_target).ewm(span=smooth_span, min_periods=1).mean().values
            
            y_val_target = df_val_with_buf[self.target_col].values
            y_val_raw_cv = y_val_target.copy()
            if smooth_span > 0:
                y_val_target = pd.Series(y_val_target).ewm(span=smooth_span, min_periods=1).mean().values
            
            y_test_target = df_test_with_buf[self.target_col].values
            y_test_raw_cv = y_test_target.copy()
            if smooth_span > 0:
                y_test_target = pd.Series(y_test_target).ewm(span=smooth_span, min_periods=1).mean().values
            
            # 构建训练集样本
            X_tr_mac, X_tr_raw, y_tr_norm, anc_tr, y_tr_obs = self._internal_parallel_build(
                total_buf, len(df_train), X_tr_mtx, y_tr_target, df_train["时间"].values,
                scale_param, tgt_w, K, 4 * X_tr_mtx.shape[1], max_gap_sec, sampling_sec, f"F{fold_idx}_Tr",
                y_raw_all=y_tr_raw_cv
            )
            
            self.scaler_y.fit(y_tr_norm)
            
            # 构建验证集样本
            X_val_mac, X_val_raw, y_val_norm, anc_val, y_val_obs = self._internal_parallel_build(
                total_buf + tgt_w, len(df_val_with_buf), X_val_buf_mtx, y_val_target, df_val_with_buf["时间"].values,
                scale_param, tgt_w, K, 4 * X_val_buf_mtx.shape[1], max_gap_sec, sampling_sec, f"F{fold_idx}_Val",
                y_raw_all=y_val_raw_cv
            )
            
            # 构建测试集样本
            X_test_mac, X_test_raw, y_test_norm, anc_test, y_test_obs = self._internal_parallel_build(
                total_buf + tgt_w, len(df_test_with_buf), X_test_buf_mtx, y_test_target, df_test_with_buf["时间"].values,
                scale_param, tgt_w, K, 4 * X_test_buf_mtx.shape[1], max_gap_sec, sampling_sec, f"F{fold_idx}_Test",
                y_raw_all=y_test_raw_cv
            )
            
            fold_data = {
                "X_tr": X_tr_mac, "X_tr_raw": X_tr_raw,
                "y_tr": self.scaler_y.transform(y_tr_norm), "anchor_tr": anc_tr,
                "X_val": X_val_mac, "X_val_raw": X_val_raw,
                "y_val": self.scaler_y.transform(y_val_norm), "anchor_val": anc_val,
                "y_val_raw": y_val_norm, "y_val_obs": y_val_obs,
                "X_test": X_test_mac, "X_test_raw": X_test_raw,
                "y_test": self.scaler_y.transform(y_test_norm), "anchor_test": anc_test,
                "y_test_raw": y_test_norm, "y_test_obs": y_test_obs,
                "granulated_feat_dim": X_tr_mac.shape[-1],
                "raw_feat_dim": X_tr_raw.shape[-1]
            }
            yield fold_idx, fold_data


    def process_and_split(self, df: pd.DataFrame, sampling_sec: float) -> Dict:
        pretty_title("Step 3-5  训练/验证/测试 数据划分与处理 (单尺度)")
        
        if getattr(self.cfg, "USE_DATA_CACHE", False):
            cache_path = self.get_cache_path()
            if os.path.exists(cache_path):
                logging.info(f"[Cache] 检测到缓存，正在加载: {cache_path}")
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                    # 同步预处理器的状态
                    if "preprocessor_state" in cache_data:
                        state = cache_data.pop("preprocessor_state")
                        self.scaler_core = state["scaler_core"]
                        self.scaler_aux = state["scaler_aux"]
                        self.pca_aux = state["pca_aux"]
                        self.scaler_y = state["scaler_y"]
                        self.core_cols = state["core_cols"]
                        self.aux_cols = state["aux_cols"]
                        self.input_cols = state["input_cols"]
                    return cache_data
                except Exception as e:
                    logging.warning(f"[Cache] 加载缓存失败: {e}，将重新生成数据。")

        max_gap_sec = sampling_sec * self.cfg.max_gap_fill_multiplier
        n_total_raw = len(df)
        n_test = int(n_total_raw * self.cfg.test_split)
        n_val = int(n_total_raw * self.cfg.val_split)
        n_train = n_total_raw - n_val - n_test
        
        # 1. 基础数据集切分
        df_train = df.iloc[:n_train].copy()
        df_val = df.iloc[n_train : n_train + n_val].copy()
        df_test = df.iloc[n_train + n_val:].copy()
        
        # 2. 训练集预处理与归一化拟合
        all_numeric_cols = self.input_cols + [self.target_col]
        raw_clip_cols = self._get_raw_clip_cols(df_train)
        df_train = self._smart_fill_small_gaps(df_train, self.input_cols, "时间", max_gap_sec)
        df_train = self._fill_missing_values(df_train, all_numeric_cols)
        self._compute_clip_limits(df_train, raw_clip_cols)
        df_train = self._apply_clip_limits(df_train, raw_clip_cols)
        df_train = self._final_safety_net(df_train, all_numeric_cols)
        
        self.scaler_core.fit(df_train[self.core_cols].values)
        if len(self.aux_cols) > 0:
            self.scaler_aux.fit(df_train[self.aux_cols].values)
            aux_scaled = self.scaler_aux.transform(df_train[self.aux_cols].values)
            self.pca_aux.fit(aux_scaled)

        def _clean_df(d_):
            d_ = self._smart_fill_small_gaps(d_, self.input_cols, "时间", max_gap_sec)
            d_ = self._fill_missing_values(d_, all_numeric_cols)
            d_ = self._apply_clip_limits(d_, raw_clip_cols)
            d_ = self._final_safety_net(d_, all_numeric_cols)
            return d_

        df_val = _clean_df(df_val)
        df_test = _clean_df(df_test)

        def _transform_to_matrix(d_):
            core_data = self.scaler_core.transform(d_[self.core_cols].values)
            if len(self.aux_cols) > 0:
                aux_data = self.scaler_aux.transform(d_[self.aux_cols].values)
                pca_data = self.pca_aux.transform(aux_data)
                return np.hstack([core_data, pca_data])
            else:
                return core_data

        X_tr_combined = _transform_to_matrix(df_train)
        X_val_combined = _transform_to_matrix(df_val)
        X_test_combined = _transform_to_matrix(df_test)

        X_all = np.vstack([X_tr_combined, X_val_combined, X_test_combined])
        df = self._final_safety_net(df, all_numeric_cols)
        y_raw = df[self.target_col].values.copy()
        # [目标平滑] 对训练目标做 EMA 平滑，让模型学习预测平滑信号
        smooth_span = getattr(self.cfg, "target_smooth_span", 0)
        if smooth_span > 0:
            y_all = pd.Series(y_raw).ewm(span=smooth_span, min_periods=1).mean().values
            logging.info(f"[Smooth] 目标 EMA 平滑已启用, span={smooth_span}")
        else:
            y_all = y_raw.copy()
        time_all = df["时间"].values

        # 使用 raw_seq_len 计算 buffer (不再使用粒化参数)
        raw_seq_len = int(self.cfg.raw_seq_len)
        forecast_steps = int(self.cfg.forecast_steps)
        
        # scale_param 用于 _internal_parallel_build，保持兼容
        scale_param = {"buf_needed": raw_seq_len, "w_rows": 1, "step_rows": 1}
        tgt_w = 1  # 不再使用粒化窗口
        K = 1      # 不再使用粒化序列，K=1 表示单一时间步
        total_buf = raw_seq_len + forecast_steps
        
        # gap 应该至少覆盖 prediction_offset + forecast_steps，防止数据泄漏
        prediction_offset = int(getattr(self.cfg, "prediction_offset", 0))
        gap_min = prediction_offset + forecast_steps
        gap = gap_min if self.cfg.optimize_gap_size else max(total_buf, gap_min)
        logging.info(
            f"[Split] gap={gap} (prediction_offset={prediction_offset}, forecast_steps={forecast_steps}, total_buf={total_buf})"
        )

        # 4. 构建数据 (y_all=平滑目标用于训练标签, y_raw=原始值用于观测窗)
        # [修复] 训练标签泄露：仅允许标签索引落在训练集上界内
        X_tr_mac, X_tr_raw, y_tr_raw, anc_tr, y_tr_obs = self._internal_parallel_build(
            total_buf, n_train, X_all, y_all, time_all, scale_param, tgt_w, K,
            4 * X_all.shape[1], max_gap_sec, sampling_sec, "Train", y_raw_all=y_raw,
            max_future_index=n_train - 1
        )
        self.scaler_y.fit(y_tr_raw)

        val_start_idx = n_train + gap + total_buf
        X_val_mac, X_val_raw, y_val_raw, anc_val, y_val_obs = self._internal_parallel_build(
            val_start_idx, n_train + n_val, X_all, y_all, time_all, scale_param, tgt_w, K,
            4 * X_all.shape[1], max_gap_sec, sampling_sec, "Val", y_raw_all=y_raw,
            max_future_index=n_train + n_val - 1
        )

        test_start_idx = n_train + n_val + gap + total_buf
        X_test_mac, X_test_raw, y_test_raw, anc_test, y_test_obs = self._internal_parallel_build(
            test_start_idx, len(X_all), X_all, y_all, time_all, scale_param, tgt_w, K,
            4 * X_all.shape[1], max_gap_sec, sampling_sec, "Test", y_raw_all=y_raw,
            max_future_index=len(X_all) - 1
        )

        granulated_feat_dim = X_tr_mac.shape[-1]

        # 提取测试集对应的时间戳
        # 测试集样本的时间戳对应于每个样本的最后一个时间点
        test_sample_count = X_test_raw.shape[0]
        test_time_indices = np.arange(test_start_idx, test_start_idx + test_sample_count)
        # 确保索引不越界
        test_time_indices = test_time_indices[test_time_indices < len(time_all)]
        timestamps_test = time_all[test_time_indices] if len(test_time_indices) == test_sample_count else None
        
        result = {
            "X_tr": X_tr_mac, "X_val": X_val_mac, "X_test": X_test_mac,
            "X_tr_raw": X_tr_raw, "X_val_raw": X_val_raw, "X_test_raw": X_test_raw,
            "y_tr": self.scaler_y.transform(y_tr_raw), "anchor_tr": anc_tr,
            "y_val": self.scaler_y.transform(y_val_raw), "anchor_val": anc_val,
            "y_test": self.scaler_y.transform(y_test_raw), "anchor_test": anc_test,
            "y_test_raw": y_test_raw, "y_test_obs": y_test_obs,
            "granulated_feat_dim": granulated_feat_dim,
            "raw_feat_dim": X_tr_raw.shape[-1],
            "timestamps_test": timestamps_test
        }

        if getattr(self.cfg, "USE_DATA_CACHE", False):
            cache_path = self.get_cache_path()
            logging.info(f"[Cache] 正在保存数据到缓存: {cache_path}")
            # 保存预处理器状态，以便加载缓存时能恢复
            cache_to_save = result.copy()
            cache_to_save["preprocessor_state"] = {
                "scaler_core": self.scaler_core,
                "scaler_aux": self.scaler_aux,
                "pca_aux": self.pca_aux,
                "scaler_y": self.scaler_y,
                "core_cols": self.core_cols,
                "aux_cols": self.aux_cols,
                "input_cols": self.input_cols
            }
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_to_save, f)
            except Exception as e:
                logging.warning(f"[Cache] 保存缓存失败: {e}")

        return result
    
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scaler_core, os.path.join(save_dir, "scaler_core.pkl"))
        joblib.dump(self.scaler_aux, os.path.join(save_dir, "scaler_aux.pkl"))
        joblib.dump(self.pca_aux, os.path.join(save_dir, "pca_aux.pkl"))
        joblib.dump(self.scaler_y, os.path.join(save_dir, "scaler_y.pkl"))
        logging.info(f"[Save] 预处理器已保存到 {save_dir}")

    def load(self, save_dir: str):
        self.scaler_core = joblib.load(os.path.join(save_dir, "scaler_core.pkl"))
        self.scaler_aux = joblib.load(os.path.join(save_dir, "scaler_aux.pkl"))
        self.pca_aux = joblib.load(os.path.join(save_dir, "pca_aux.pkl"))
        self.scaler_y = joblib.load(os.path.join(save_dir, "scaler_y.pkl"))
        logging.info(f"[Load] 预处理器已加载 from {save_dir}")
