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

import pandas as pd
import os

ablation_root = 'outputs/Ablation_Study_20260216_093853'
configs = ['full_model', 'no_positional_encoding', 'no_mc_dropout', 'no_revin', 'no_cqr', 'layers_1', 'layers_3', 'heads_2', 'heads_8']

results = []
for cfg in configs:
    csv_path = os.path.join(ablation_root, f'Ablation_{cfg}', 'cv_results', 'cv_summary_report.csv')
    df = pd.read_csv(csv_path)
    # 只取 test 集且 cqr=off 的行
    test_df = df[(df['split'] == 'test') & (df['cqr'] == 'off')]
    mae_mean = test_df['MAE'].mean()
    mae_std = test_df['MAE'].std()
    results.append({'config': cfg, 'MAE_mean': mae_mean, 'MAE_std': mae_std})

results_df = pd.DataFrame(results).sort_values('MAE_mean')
print('='*60)
print('消融实验结果汇总 (8折CV TEST集 MAE)')
print('='*60)
for _, row in results_df.iterrows():
    print(f"{row['config']:<30} MAE: {row['MAE_mean']:.5f} +/- {row['MAE_std']:.5f}")
print('='*60)
print(f"最优配置: {results_df.iloc[0]['config']}")
print(f"最优MAE: {results_df.iloc[0]['MAE_mean']:.5f}")
