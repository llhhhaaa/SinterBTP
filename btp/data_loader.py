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

# data_loader.py
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import logging

from btp.utils import pretty_title, normalize_col_name, make_unique_cols
from scipy import stats




class DataLoader:
    @staticmethod
    def load_xlsx(
        path: str,
        prefer_time_col: str = "时间",
        preview_rows: int = 10,
    ) -> Tuple[pd.DataFrame, float]:
        """
        🔧 增强版：鲁棒的采样间隔推断
        """
        pretty_title("Step 1  读取 Excel 数据（增强版）")

        logging.info(f"[INFO] 目标文件: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        df = pd.read_excel(path)
        logging.info(f"[INFO] 原始形状: {df.shape}")

        # 规范化列名
        df.columns = [normalize_col_name(c) for c in df.columns]
        df.columns = make_unique_cols(df.columns)

        # 查找时间列
        time_col = None
        if prefer_time_col in df.columns:
            time_col = prefer_time_col
        else:
            for c in df.columns:
                if "时间" in c:
                    time_col = c
                    break

        if time_col is None:
            raise ValueError("找不到时间列")

        logging.info(f"[INFO] 时间列: {time_col}")

        # 解析时间
        raw_sample = str(df[time_col].iloc[0])
        if re.match(r"^\d{2}:\d{2}:\d{2}$", raw_sample):
            df["时间"] = pd.to_timedelta(df[time_col].astype(str), errors="coerce")
            base_date = pd.to_datetime("2000-01-01")
            df["时间"] = base_date + df["时间"]
        else:
            df["时间"] = pd.to_datetime(df[time_col], errors="coerce")

        # 丢弃时间NaN
        before = len(df)
        df = df.dropna(subset=["时间"]).copy()
        after = len(df)
        logging.info(f"[INFO] 去除时间NaN: {before} -> {after}")

        if after < 2:
            raise ValueError("有效时间行不足2行")

        # 排序
        df = df.sort_values("时间").reset_index(drop=True)

        # 去重（保留最后一条）
        dup_count = df["时间"].duplicated().sum()
        if dup_count > 0:
            logging.warning(f"[WARN] 检测到 {dup_count} 个重复时间戳，保留最后一条")
            df = df.drop_duplicates(subset=["时间"], keep="last").reset_index(drop=True)

        # 🔧 增强的采样间隔推断
        dt = df["时间"].diff().dt.total_seconds()
        dt_valid = dt[dt > 0]
        
        if dt_valid.empty:
            logging.error("[ERROR] 无法推断采样间隔（所有时间差≤0），请检查数据源")
            raise ValueError("数据时间序列异常")
        
        # 使用众数而非中位数（更鲁棒）
        mode_result = stats.mode(dt_valid.round(1), keepdims=True)  # 四舍五入到0.1秒
        sampling_sec = float(mode_result.mode[0])
        
        # 如果众数不可靠，回退到中位数
        if sampling_sec <= 0 or sampling_sec > 3600:
            sampling_sec = float(np.median(dt_valid))
            logging.warning(f"[WARN] 众数推断失败，使用中位数: {sampling_sec:.3f}秒")
        
        logging.info(f"[INFO] 推断采样间隔: {sampling_sec:.3f}秒 (众数法)")
        
        # 验证推断结果
        expected_count = (df["时间"].iloc[-1] - df["时间"].iloc[0]).total_seconds() / sampling_sec
        actual_count = len(df)
        completeness = actual_count / expected_count * 100
        logging.info(f"[INFO] 数据完整度: {completeness:.1f}% ({actual_count}/{int(expected_count)})")
        
        if completeness < 50:
            logging.warning("[WARN] 数据完整度<50%，可能存在大量间隙")

        # 检测大间隙
        large_gaps = dt_valid[dt_valid > sampling_sec * 10]
        if len(large_gaps) > 0:
            logging.warning(
                f"[WARN] 检测到 {len(large_gaps)} 个异常间隙（最大: {large_gaps.max():.1f}秒）"
            )

        logging.info(f"[OK] 数据加载完成，形状: {df.shape}\n")
        return df, sampling_sec
