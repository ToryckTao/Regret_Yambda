#!/usr/bin/env python3
"""
预处理脚本：生成用户后悔事件 + 后悔池初始化数据

从原始交互数据中提取：
1. 用户的后悔事件 (like -> unlike, dislike -> undislike)
2. 后悔池初始化所需的数据

输出文件放在 SARA_Yambda/dataset/ 目录下
"""

import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm

# ================= 配置路径 =================
# 原始数据路径
RAW_DATA_DIR = '/root/autodl-tmp/data/HSRL/dataset/yambda_50m/raw/sequential/50m'

# 输出路径
OUT_DIR = '/root/autodl-tmp/data/HSRL/HSRL/SARA_Yambda/dataset'
os.makedirs(OUT_DIR, exist_ok=True)

# 预处理参数 (与 regret_config.py 保持一致)
from regret_config import (
    LAMBDA_UNLIKE, LAMBDA_UNDISLIKE, REGRET_GAMMA, REGRET_POOL_SIZE
)

# ================= 加载原始数据 =================
print("=" * 60)
print("Step 1: 加载原始交互数据")
print("=" * 60)

multi_event_df = pd.read_parquet(os.path.join(RAW_DATA_DIR, 'multi_event.parquet'))
print(f"Loaded {len(multi_event_df)} users from multi_event.parquet")

# 检查数据结构
print(f"Columns: {multi_event_df.columns.tolist()}")
print(f"Example row: {multi_event_df.iloc[0].to_dict()}")

# ================= 提取后悔事件 =================
print("\n" + "=" * 60)
print("Step 2: 提取后悔事件")
print("=" * 60)

def extract_regret_events(user_item_ids, user_event_types, user_played_ratios, user_timestamps):
    """
    遍历用户交互序列，识别后悔事件
    
    后悔事件类型：
    1. like -> unlike: 喜欢后取消喜欢 (惩罚)
    2. dislike -> undislike: 取消不喜欢 (奖励)
    
    返回: list of (decision_time, regret_time, regret_type, item_id, phi)
        - decision_time: 原始推荐时间
        - regret_time: 后悔发生时间
        - regret_type: 'unlike' 或 'undislike'
        - item_id: 物品ID
        - phi: 惩罚值 Φ = γ^Δt * ψ_t
    """
    events = []
    n = len(user_event_types)
    
    for t in range(n):
        current_event = user_event_types[t]
        
        # Case 1: unlike (喜欢 -> 取消喜欢)
        if current_event == 'unlike':
            item_id = user_item_ids[t]
            # 向前找最近的 like 事件
            for prev_t in range(t - 1, -1, -1):
                if user_event_types[prev_t] == 'like' and user_item_ids[prev_t] == item_id:
                    # 找到原始决策时间
                    delta_t = t - prev_t
                    psi = -LAMBDA_UNLIKE  # 惩罚
                    phi = (REGRET_GAMMA ** delta_t) * psi
                    events.append({
                        'decision_time': prev_t,
                        'regret_time': t,
                        'regret_type': 'unlike',
                        'item_id': item_id,
                        'delta_t': delta_t,
                        'psi': psi,
                        'phi': phi,
                        'timestamp': user_timestamps[t] if user_timestamps is not None else None
                    })
                    break
        
        # Case 2: undislike (不喜欢 -> 取消不喜欢)
        elif current_event == 'undislike':
            item_id = user_item_ids[t]
            # 向前找最近的 dislike 事件
            for prev_t in range(t - 1, -1, -1):
                if user_event_types[prev_t] == 'dislike' and user_item_ids[prev_t] == item_id:
                    # 找到原始决策时间
                    delta_t = t - prev_t
                    psi = LAMBDA_UNDISLIKE  # 奖励
                    phi = (REGRET_GAMMA ** delta_t) * psi
                    events.append({
                        'decision_time': prev_t,
                        'regret_time': t,
                        'regret_type': 'undislike',
                        'item_id': item_id,
                        'delta_t': delta_t,
                        'psi': psi,
                        'phi': phi,
                        'timestamp': user_timestamps[t] if user_timestamps is not None else None
                    })
                    break
    
    return events

# 处理所有用户
all_regret_events = {}
stats = {
    'total_users': 0,
    'users_with_unlike': 0,
    'users_with_undislike': 0,
    'total_unlike_events': 0,
    'total_undislike_events': 0,
}

for idx, row in tqdm(multi_event_df.iterrows(), total=len(multi_event_df), desc="Extracting regret events"):
    uid = row['uid']
    item_ids = row['item_id']
    event_types = row['event_type']
    played_ratios = row.get('played_ratio_pct', [np.nan] * len(item_ids))
    timestamps = row.get('timestamp', None)
    
    events = extract_regret_events(item_ids, event_types, played_ratios, timestamps)
    
    if events:
        all_regret_events[uid] = events
        stats['total_users'] += 1
        
        unlike_count = sum(1 for e in events if e['regret_type'] == 'unlike')
        undislike_count = sum(1 for e in events if e['regret_type'] == 'undislike')
        
        if unlike_count > 0:
            stats['users_with_unlike'] += 1
            stats['total_unlike_events'] += unlike_count
        if undislike_count > 0:
            stats['users_with_undislike'] += 1
            stats['total_undislike_events'] += undislike_count

print(f"\n统计信息:")
print(f"  总用户数: {len(multi_event_df)}")
print(f"  有后悔事件的用户: {stats['total_users']}")
print(f"  有 unlike 事件的用户: {stats['users_with_unlike']}")
print(f"  有 undislike 事件的用户: {stats['users_with_undislike']}")
print(f"  总 unlike 事件: {stats['total_unlike_events']}")
print(f"  总 undislike 事件: {stats['total_undislike_events']}")

# ================= 生成后悔池初始化数据 =================
print("\n" + "=" * 60)
print("Step 3: 生成后悔池初始化数据")
print("=" * 60)

# 需要加载 item2sid 映射
SID_PATH = '/root/autodl-tmp/data/HSRL/HSRL/dataset/yambda_item2sid.pkl'
if os.path.exists(SID_PATH):
    with open(SID_PATH, 'rb') as f:
        item2sid = pickle.load(f)
    print(f"Loaded item2sid mapping for {len(item2sid)} items")
else:
    print(f"Warning: item2sid not found at {SID_PATH}, will use None")
    item2sid = None

def build_regret_pool_data(user_id, regret_events, item2sid, max_pool_size=REGRET_POOL_SIZE):
    """
    为单个用户构建后悔池数据
    
    返回: {
        'user_id': user_id,
        'regret_paths': list of (sid_tokens, phi)
            - sid_tokens: tuple of (token_l1, token_l2, token_l3)
            - phi: 惩罚值
    }
    """
    # 按时间排序，取最近的后悔事件
    sorted_events = sorted(regret_events, key=lambda x: x['regret_time'], reverse=True)
    
    regret_paths = []
    for event in sorted_events[:max_pool_size]:
        item_id = event['item_id']
        phi = event['phi']
        
        if item2sid is not None and item_id in item2sid:
            sid_tokens = tuple(item2sid[item_id])
        else:
            sid_tokens = None
        
        if sid_tokens is not None:
            regret_paths.append({
                'sid_tokens': sid_tokens,
                'phi': phi,
                'regret_type': event['regret_type'],
                'delta_t': event['delta_t']
            })
    
    return {
        'user_id': user_id,
        'regret_paths': regret_paths,
        'n_regrets': len(regret_paths)
    }

# 为所有有后悔事件的用户生成数据
regret_pool_data = {}
for uid, events in tqdm(all_regret_events.items(), desc="Building regret pool data"):
    pool_data = build_regret_pool_data(uid, events, item2sid)
    if pool_data['n_regrets'] > 0:
        regret_pool_data[uid] = pool_data

print(f"\n生成后悔池用户数: {len(regret_pool_data)}")

# ================= 保存结果 =================
print("\n" + "=" * 60)
print("Step 4: 保存结果")
print("=" * 60)

# 1. 后悔事件列表
regret_events_path = os.path.join(OUT_DIR, 'regret_events.pkl')
with open(regret_events_path, 'wb') as f:
    pickle.dump(all_regret_events, f)
print(f"Saved regret events to {regret_events_path}")

# 2. 后悔池初始化数据
regret_pool_path = os.path.join(OUT_DIR, 'regret_pool_init.pkl')
with open(regret_pool_path, 'wb') as f:
    pickle.dump(regret_pool_data, f)
print(f"Saved regret pool init data to {regret_pool_path}")

# 3. 统计信息
stats_path = os.path.join(OUT_DIR, 'regret_stats.txt')
with open(stats_path, 'w') as f:
    f.write("Regret Events Statistics\n")
    f.write("=" * 40 + "\n")
    for k, v in stats.items():
        f.write(f"{k}: {v}\n")
    f.write(f"\nParameters:\n")
    f.write(f"  LAMBDA_UNLIKE: {LAMBDA_UNLIKE}\n")
    f.write(f"  LAMBDA_UNDISLIKE: {LAMBDA_UNDISLIKE}\n")
    f.write(f"  REGRET_GAMMA: {REGRET_GAMMA}\n")
    f.write(f"  REGRET_POOL_SIZE: {REGRET_POOL_SIZE}\n")
print(f"Saved stats to {stats_path}")

# 4. 简单统计 CSV
stats_df = pd.DataFrame([stats])
stats_csv_path = os.path.join(OUT_DIR, 'regret_stats.csv')
stats_df.to_csv(stats_csv_path, index=False)
print(f"Saved stats CSV to {stats_csv_path}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
print(f"\n输出文件:")
print(f"  1. {regret_events_path}")
print(f"  2. {regret_pool_path}")
print(f"  3. {stats_path}")
print(f"  4. {stats_csv_path}")
