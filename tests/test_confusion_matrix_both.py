#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试混淆矩阵的 2D 和 3D 版本
"""
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btp.visualizer import Visualizer

def test_both_confusion_matrix_versions():
    """测试 2D 和 3D 混淆矩阵函数"""
    
    # 创建保存目录
    save_dir = "screenshots"
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建 Visualizer 实例
    viz = Visualizer(save_dir=save_dir)
    
    # 模拟数据：3分类状态 (0:过烧, 1:正常, 2:欠烧)
    np.random.seed(42)
    n_samples = 500
    
    # 生成真实标签
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.5, 0.3])
    
    # 生成预测标签 (添加一些噪声，模拟真实预测)
    y_pred = y_true.copy()
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    for idx in noise_idx:
        # 随机改变预测
        y_pred[idx] = np.random.choice([0, 1, 2])
    
    print("=" * 60)
    print("测试混淆矩阵可视化功能")
    print("=" * 60)
    
    # 测试 2D 热力图版本
    print("\n1. 测试 2D 热力图版本...")
    try:
        viz.plot_diagnosis_confusion_matrix_2d(
            y_true, y_pred, 
            "test_confusion_matrix_2d.png"
        )
        print("   ✅ 2D 热力图版本测试成功!")
    except Exception as e:
        print(f"   ❌ 2D 热力图版本测试失败: {e}")
    
    # 测试 3D 柱状图版本
    print("\n2. 测试 3D 柱状图版本...")
    try:
        viz.plot_diagnosis_confusion_matrix_3d(
            y_true, y_pred, 
            "test_confusion_matrix_3d.png"
        )
        print("   ✅ 3D 柱状图版本测试成功!")
    except Exception as e:
        print(f"   ❌ 3D 柱状图版本测试失败: {e}")
    
    # 测试包装函数 (默认 2D)
    print("\n3. 测试包装函数 (mode='2d')...")
    try:
        viz.plot_diagnosis_confusion_matrix(
            y_true, y_pred, 
            "test_confusion_matrix_wrapper_2d.png",
            mode='2d'
        )
        print("   ✅ 包装函数 (mode='2d') 测试成功!")
    except Exception as e:
        print(f"   ❌ 包装函数 (mode='2d') 测试失败: {e}")
    
    # 测试包装函数 (3D)
    print("\n4. 测试包装函数 (mode='3d')...")
    try:
        viz.plot_diagnosis_confusion_matrix(
            y_true, y_pred, 
            "test_confusion_matrix_wrapper_3d.png",
            mode='3d'
        )
        print("   ✅ 包装函数 (mode='3d') 测试成功!")
    except Exception as e:
        print(f"   ❌ 包装函数 (mode='3d') 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！请查看 screenshots 目录中的图片文件:")
    print("  - test_confusion_matrix_2d.png")
    print("  - test_confusion_matrix_3d.png")
    print("  - test_confusion_matrix_wrapper_2d.png")
    print("  - test_confusion_matrix_wrapper_3d.png")
    print("=" * 60)

if __name__ == "__main__":
    test_both_confusion_matrix_versions()