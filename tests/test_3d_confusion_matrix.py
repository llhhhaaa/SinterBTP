"""
测试三维混淆矩阵可视化
"""
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btp.visualizer import Visualizer

def test_3d_confusion_matrix():
    """测试三维混淆矩阵绘制"""
    
    # 创建模拟数据：3分类状态 (0:过烧, 1:正常, 2:欠烧)
    np.random.seed(42)
    n_samples = 500
    
    # 生成真实状态 (有一定分布)
    true_states = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.5, 0.3])
    
    # 生成预测状态 (模拟真实场景，有一定噪声)
    pred_states = true_states.copy()
    noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    
    for idx in noise_indices:
        # 随机添加噪声
        pred_states[idx] = np.random.choice([0, 1, 2])
    
    # 创建保存目录
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'screenshots')
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建可视化器
    viz = Visualizer(save_dir=save_dir)
    
    # 绘制三维混淆矩阵
    viz.plot_diagnosis_confusion_matrix(
        y_true_states=true_states,
        y_pred_states=pred_states,
        fname='test_3d_confusion_matrix.png'
    )
    
    print(f"✅ 三维混淆矩阵已保存到: {os.path.join(save_dir, 'test_3d_confusion_matrix.png')}")
    print(f"真实状态分布: 过烧={sum(true_states==0)}, 正常={sum(true_states==1)}, 欠烧={sum(true_states==2)}")
    print(f"预测状态分布: 过烧={sum(pred_states==0)}, 正常={sum(pred_states==1)}, 欠烧={sum(pred_states==2)}")

if __name__ == "__main__":
    test_3d_confusion_matrix()