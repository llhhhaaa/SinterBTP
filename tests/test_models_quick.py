"""
快速测试脚本：验证所有模型能否正常前向传播和反向传播
运行方式: python quick_test_models.py
"""
import torch
import torch.nn as nn
import sys

# 添加当前目录到路径
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from btp.config import TrainConfig
from btp.model import build_model, QuantileLoss

def quick_test_all_models():
    """快速测试所有模型能否跑通"""
    
    # 测试配置
    cfg = TrainConfig()
    cfg.raw_seq_len = 360
    cfg.hidden_size = 128
    cfg.dropout = 0.3
    cfg.forecast_steps = 3
    cfg.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    cfg.enable_revin = True
    cfg.attn_heads = 4
    
    # 模拟输入数据
    batch_size = 4
    seq_len = 360
    input_dim = 15  # 假设15个特征
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("="*60)
    
    # 要测试的模型列表
    models_to_test = [
        "enhanced_transformer",
        "baseline_transformer", 
        "baseline_lstm",
        "baseline_gru"
    ]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n测试模型: {model_name}")
        print("-"*40)
        
        try:
            # 1. 创建模型
            model = build_model(cfg, input_dim, model_type=model_name)
            model = model.to(device)
            print(f"  ✓ 模型创建成功")
            
            # 统计参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  参数量: {total_params:,} (可训练: {trainable_params:,})")
            
            # 2. 创建测试数据
            x = torch.randn(batch_size, seq_len, input_dim).to(device)
            y = torch.randn(batch_size, cfg.forecast_steps).to(device)
            
            # 3. 前向传播
            model.train()
            output = model(x)
            expected_shape = (batch_size, cfg.forecast_steps, len(cfg.quantiles))
            
            if output.shape != expected_shape:
                raise ValueError(f"输出形状错误: {output.shape} != {expected_shape}")
            print(f"  ✓ 前向传播成功, 输出形状: {output.shape}")
            
            # 4. 计算损失
            criterion = QuantileLoss(cfg).to(device)
            loss = criterion(output, y)
            print(f"  ✓ 损失计算成功, loss={loss.item():.6f}")
            
            # 5. 反向传播
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"  ✓ 反向传播成功")
            
            # 6. 推理模式测试
            model.eval()
            with torch.no_grad():
                output_eval = model(x)
            print(f"  ✓ 推理模式成功")
            
            results[model_name] = "✓ PASS"
            print(f"\n  >>> {model_name}: 全部测试通过! <<<")
            
        except Exception as e:
            results[model_name] = f"✗ FAIL: {str(e)}"
            print(f"\n  >>> {model_name}: 测试失败! <<<")
            print(f"  错误信息: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总:")
    print("="*60)
    
    all_passed = True
    for model_name, result in results.items():
        status = "✓" if "PASS" in result else "✗"
        print(f"  {status} {model_name}: {result}")
        if "FAIL" in result:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 所有模型测试通过！可以开始正式实验。")
        return True
    else:
        print("⚠️ 部分模型测试失败，请修复后再运行正式实验。")
        return False


if __name__ == "__main__":
    success = quick_test_all_models()
    sys.exit(0 if success else 1)
