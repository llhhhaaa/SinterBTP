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

# utils.py
import re
import logging
import sys


def pretty_title(text: str, width: int = 80):
    """打印美化的标题栏"""
    line = "=" * width
    logging.info(f"\n{line}")
    logging.info(f"{text.center(width)}")
    logging.info(f"{line}\n")


def normalize_col_name(col_name: str) -> str:
    """
    规范化列名：
    1. 去除前后空格
    2. 将连续空格替换为单个下划线
    3. 移除特殊字符（保留中文、英文、数字、下划线）
    """
    s = str(col_name).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", s)
    return s


def make_unique_cols(cols: list) -> list:
    """
    处理重复列名，追加数字后缀
    例：['A', 'A', 'B', 'A'] -> ['A', 'A_1', 'B', 'A_2']
    """
    seen = {}
    result = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            result.append(c)
        else:
            seen[c] += 1
            result.append(f"{c}_{seen[c]}")
    return result

def setup_logging(log_path: str, reset_handlers: bool = True):
    """
    ✅ 修复版日志配置：支持 GUI 模式（不重置 handler）
    
    Args:
        log_path: 日志文件路径
        reset_handlers: 是否清空现有 handler（GUI 模式下设为 False）
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # ✅ 修改：根据参数决定是否清空
    if reset_handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # ✅ 文件处理器（立即刷新）
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # ✅ 控制台处理器（只在非 GUI 模式下添加）
    if reset_handlers:  # GUI 模式下不添加控制台 handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # ✅ 写入测试日志，确认文件正常工作
    logging.info(f"日志系统初始化成功，日志文件: {log_path}")