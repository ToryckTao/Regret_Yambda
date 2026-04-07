#!/usr/bin/env python
"""
将原始RL4RS数据集转换为项目需要的格式
原始格式: timestamp@session_id@sequence_id@exposed_items@user_feedback@...
目标格式: tsv格式，按session_id分割train/val/test
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 配置
DATA_DIR = Path("/root/autodl-tmp/data/HSRL/dataset/rl4rs/rl4rs-dataset")
OUTPUT_DIR = Path("/root/autodl-tmp/data/HSRL/dataset")

# 读取原始数据
print("读取原始数据...")
df = pd.read_csv(DATA_DIR / "rl4rs_dataset_a_sl.csv", sep='@', header=0)
print(f"原始数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# 查看数据基本信息
print(f"\n总session数: {df['session_id'].nunique()}")
print(f"总记录数: {len(df)}")

# 按session_id分组，每个session保留80%作为训练，10%验证，10%测试
np.random.seed(42)
all_sessions = df['session_id'].unique()
np.random.shuffle(all_sessions)

n_train = int(len(all_sessions) * 0.8)
n_val = int(len(all_sessions) * 0.9)

train_sessions = all_sessions[:n_train]
val_sessions = all_sessions[n_train:n_val]
test_sessions = all_sessions[n_val:]

print(f"\n训练集session数: {len(train_sessions)}")
print(f"验证集session数: {len(val_sessions)}")
print(f"测试集session数: {len(test_sessions)}")

# 分割数据
train_df = df[df['session_id'].isin(train_sessions)]
val_df = df[df['session_id'].isin(val_sessions)]
test_df = df[df['session_id'].isin(test_sessions)]

print(f"\n训练集记录数: {len(train_df)}")
print(f"验证集记录数: {len(val_df)}")
print(f"测试集记录数: {len(test_df)}")

# 保存为tsv格式
output_dir = OUTPUT_DIR / "rl4rs_processed"
output_dir.mkdir(parents=True, exist_ok=True)

# 保存处理后的数据（只需保存原始列，reader会自动处理）
train_df.to_csv(output_dir / "train.csv", sep='\t', index=False)
val_df.to_csv(output_dir / "val.csv", sep='\t', index=False)
test_df.to_csv(output_dir / "test.csv", sep='\t', index=False)

print(f"\n数据已保存到: {output_dir}")
print("  - train.csv")
print("  - val.csv") 
print("  - test.csv")

# 同时复制item_info.csv（重命名以匹配代码期望）
import shutil
shutil.copy(DATA_DIR / "item_info.csv", output_dir / "item_info.csv")
print("  - item_info.csv")

print("\n数据预处理完成！")
