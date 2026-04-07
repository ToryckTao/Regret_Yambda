#!/usr/bin/env python
"""测试 YambdaUserResponse 模型输出的实际值范围"""

import torch
import sys
sys.path.insert(0, '/root/autodl-tmp/data/HSRL/HSRL')

from model.YambdaUserResponse import YambdaUserResponse
from reader.YambdaDataReader import YambdaDataReader
from argparse import Namespace
import numpy as np

# 加载模型
model_path = "/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/env/yambda_user_env_lr0.001_reg0.0001.model.checkpoint"
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

# 从 log 文件中恢复参数
log_path = "/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/env/log/yambda_user_env_lr0.001_reg0.0001.model.log"
with open(log_path, 'r') as f:
    class_args = eval(f.readline())
    model_args = eval(f.readline())

print(f"Model args: {model_args}")

# 创建 reader
reader = YambdaDataReader(model_args)
print(f"Reader stats: {reader.get_statistics()}")

# 创建模型
args = Namespace(**{**vars(model_args), 'device': 'cpu'})
model = YambdaUserResponse(args, reader, 'cpu')

# 加载权重
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# 测试几个样本
print("\n=== Testing model output ===")
all_preds = []

for i in range(10):
    data = reader[i]
    
    # 准备 feed_dict
    feed_dict = {
        'history': torch.tensor(data['history']).unsqueeze(0).float(),
        'history_features': torch.tensor(data['history_features']).unsqueeze(0).float(),
        'exposure': torch.tensor(data['exposure']).unsqueeze(0).float(),
        'exposure_features': torch.tensor(data['exposure_features']).unsqueeze(0).float(),
        'user_profile': torch.tensor(data['user_profile']).unsqueeze(0).float(),
    }
    
    with torch.no_grad():
        output = model(feed_dict)
        preds = output['preds'].numpy()
        all_preds.append(preds)
        
        print(f"Sample {i}:")
        print(f"  Exposure: {data['exposure']}")
        print(f"  Feedback (target): {data['feedback']}")
        print(f"  Predictions: {preds}")
        print(f"  Mean reward: {np.mean(preds)}")
        print()

# 汇总
all_preds = np.concatenate([p.flatten() for p in all_preds])
print(f"=== Summary ===")
print(f"Min prediction: {all_preds.min():.4f}")
print(f"Max prediction: {all_preds.max():.4f}")
print(f"Mean prediction: {all_preds.mean():.4f}")
print(f"Std prediction: {all_preds.std():.4f}")
