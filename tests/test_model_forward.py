
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import logging
from btp.config import TrainConfig
from btp.model import build_model

def test_model_forward():
    logging.basicConfig(level=logging.INFO)
    
    # 1. Setup Config
    cfg = TrainConfig()
    cfg.enable_revin = True
    cfg.target_column = "北侧_计算BTP位置"
    cfg.hidden_size = 32
    cfg.forecast_steps = 5
    
    # 2. Mock Data
    batch_size = 4
    seq_len = 60
    input_dim = 96 # 24 features * 4
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 3. Build Model
    model = build_model(cfg, input_dim=input_dim)
    model.eval()
    
    # 4. Forward Pass
    print("\n[Test] Running Forward Pass...")
    with torch.no_grad():
        preds = model(x)
    
    print(f"[Test] Output Shape: {preds.shape}")
    expected_shape = (batch_size, cfg.forecast_steps, 5) # 5 quantiles
    
    assert preds.shape == expected_shape, f"Shape Mismatch: {preds.shape} vs {expected_shape}"
    print("[Test] Shape Check Passed.")
    
    # 5. Check Denormalization Logic
    # If RevIN works, the output shouldn't be standardized to 0/1 if the input isn't.
    # Let's try with a shifted input
    shift = 100.0
    x_shifted = x + shift
    
    print(f"\n[Test] Running Forward Pass with Shifted Input (+{shift})...")
    with torch.no_grad():
        preds_shifted = model(x_shifted)
    
    # The output should roughly reflect the shift (since we denormalize)
    # Note: The model is untrained, so it might not predict x_t+1 = x_t perfectly, 
    # but the mean of preds should be shifted.
    
    mean_diff = preds_shifted.mean() - preds.mean()
    print(f"[Test] Prediction Mean Shift: {mean_diff.item():.4f}")
    
    if abs(mean_diff - shift) < 50.0: # Loose check for untrained weights
        print("[Test] Denormalization seems to be working (Output shifted with input).")
    else:
        print("[Test] WARNING: Output shift is significantly different from input shift.")

if __name__ == "__main__":
    test_model_forward()
