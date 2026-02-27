import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch

from btp.config import TrainConfig
from btp.model import EnhancedTransformer


def test_model():
    print("=" * 50)
    print("测试 DecompTransformer 架构")
    print("=" * 50)

    # 配置
    cfg = TrainConfig()
    cfg.enable_decomp = True
    cfg.enable_freq_enhance = True

    # 模型参数
    input_dim = 12  # 输入特征数(需 >= 11，模型内部会取索引 8 和 10)
    seq_len = 60  # 序列长度
    batch_size = 4
    quantiles = getattr(cfg, "quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])
    forecast_steps = getattr(cfg, "forecast_steps", 1)

    # 实例化模型
    print("\n1. 实例化模型...")
    try:
        model = EnhancedTransformer(cfg, input_dim)
        print("   ✓ 模型实例化成功")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ✗ 模型实例化失败: {e}")
        return False

    # 前向传播测试
    print("\n2. 前向传播测试...")
    try:
        x = torch.randn(batch_size, seq_len, input_dim)
        model.eval()
        with torch.no_grad():
            out = model(x)

        print(f"   输入形状: x={x.shape}")
        print(f"   输出形状: {out.shape}")
        print(f"   期望形状: ({batch_size}, {forecast_steps}, {len(quantiles)})")

        if out.shape == (batch_size, forecast_steps, len(quantiles)):
            print("   ✓ 输出形状正确")
        else:
            print("   ✗ 输出形状错误")
            return False
    except Exception as e:
        print(f"   ✗ 前向传播失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 测试消融开关
    print("\n3. 测试消融开关...")

    # 关闭分解
    try:
        cfg.enable_decomp = False
        cfg.enable_freq_enhance = True
        model_no_decomp = EnhancedTransformer(cfg, input_dim)
        model_no_decomp.eval()
        with torch.no_grad():
            out = model_no_decomp(x)
        print("   ✓ enable_decomp=False 正常")
    except Exception as e:
        print(f"   ✗ enable_decomp=False 失败: {e}")
        return False

    # 关闭频域增强
    try:
        cfg.enable_decomp = True
        cfg.enable_freq_enhance = False
        model_no_freq = EnhancedTransformer(cfg, input_dim)
        model_no_freq.eval()
        with torch.no_grad():
            out = model_no_freq(x)
        print("   ✓ enable_freq_enhance=False 正常")
    except Exception as e:
        print(f"   ✗ enable_freq_enhance=False 失败: {e}")
        return False

    print("\n" + "=" * 50)
    print("所有测试通过！")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
