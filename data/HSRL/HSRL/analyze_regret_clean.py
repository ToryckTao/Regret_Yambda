"""
Regret Interval Analysis - Clean visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

DATA_PATH = "/root/autodl-tmp/data/HSRL/dataset/yambda_50m/raw/flat/50m"
OUTPUT_PATH = "/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hac/analysis"

print("Loading data...")

likes = pd.read_parquet(f"{DATA_PATH}/likes.parquet")[['uid', 'item_id', 'timestamp']]
unlikes = pd.read_parquet(f"{DATA_PATH}/unlikes.parquet")[['uid', 'item_id', 'timestamp']]
dislikes = pd.read_parquet(f"{DATA_PATH}/dislikes.parquet")[['uid', 'item_id', 'timestamp']]
undislikes = pd.read_parquet(f"{DATA_PATH}/undislikes.parquet")[['uid', 'item_id', 'timestamp']]

for df in [likes, unlikes, dislikes, undislikes]:
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df.sort_values('timestamp', inplace=True)

# Like -> Unlike
like_unlike = pd.merge_asof(
    unlikes, 
    likes.rename(columns={'timestamp': 't_lk'}),
    left_on='timestamp', 
    right_on='t_lk',
    by=['uid', 'item_id'],
    direction='backward'
)
like_unlike = like_unlike.dropna(subset=['t_lk'])
like_unlike['dt'] = (like_unlike['timestamp'] - like_unlike['t_lk']) / 3600.0

# Dislike -> Undislike
dislike_undislike = pd.merge_asof(
    undislikes, 
    dislikes.rename(columns={'timestamp': 't_dlk'}),
    left_on='timestamp', 
    right_on='t_dlk',
    by=['uid', 'item_id'],
    direction='backward'
)
dislike_undislike = dislike_undislike.dropna(subset=['t_dlk'])
dislike_undislike['dt'] = (dislike_undislike['timestamp'] - dislike_undislike['t_dlk']) / 3600.0

lk_intervals = like_unlike[like_unlike['dt'] > 0]['dt']
dlk_intervals = dislike_undislike[dislike_undislike['dt'] > 0]['dt']

print(f"Like->Unlike: {len(lk_intervals)}, Dislike->Undislike: {len(dlk_intervals)}")

# ========== 新的可视化 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

blue = '#4A90D9'
orange = '#F5A623'

# 1. 线性直方图（0-7天）
ax1 = axes[0, 0]
ax1.hist(lk_intervals[lk_intervals <= 168], bins=50, edgecolor='black', alpha=0.7, color=blue)
ax1.set_xlabel('Time Interval (Hours)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Like->Unlike: Linear Scale (0-7 days)', fontsize=14)
ax1.axvline(lk_intervals.median(), color='red', linestyle='--', linewidth=2, 
            label=f'Median: {lk_intervals.median():.1f}h ({lk_intervals.median()/24:.1f}d)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.95, 0.95, f'n={len(lk_intervals[lk_intervals <= 168])}/{len(lk_intervals)}', 
         transform=ax1.transAxes, ha='right', va='top', fontsize=10)

# 2. 线性直方图（0-30天）
ax2 = axes[0, 1]
lk_days = lk_intervals / 24
ax2.hist(lk_days[lk_days <= 30], bins=30, edgecolor='black', alpha=0.7, color=blue)
ax2.set_xlabel('Time Interval (Days)', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Like->Unlike: Linear Scale (0-30 days)', fontsize=14)
ax2.axvline(lk_days.median(), color='red', linestyle='--', linewidth=2, 
            label=f'Median: {lk_days.median():.1f}d')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.text(0.95, 0.95, f'n={len(lk_days[lk_days <= 30])}/{len(lk_days)}', 
         transform=ax2.transAxes, ha='right', va='top', fontsize=10)

# 3. 真正的对数直方图（X轴对数，Y轴线性）
ax3 = axes[1, 0]
log_lk = np.log10(lk_intervals)
ax3.hist(log_lk, bins=50, edgecolor='black', alpha=0.7, color=blue)
ax3.set_xlabel('log10(Time Interval / Hours)', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Like->Unlike: X-axis Log Scale (Linear Y)', fontsize=14)
# 添加次要X轴显示实际时间
ax3_top = ax3.twiny()
ax3_top.set_xlim(ax3.get_xlim())
tick_positions = [-2, -1, 0, 1, 2, 3]
tick_labels = ['0.01h\n(36s)', '0.1h\n(6min)', '1h', '10h', '100h\n(4d)', '1000h\n(42d)']
ax3_top.set_xticks(tick_positions)
ax3_top.set_xticklabels(tick_labels, fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. 对比CDF
ax4 = axes[1, 1]
sorted_lk = np.sort(lk_intervals)
cdf_lk = np.arange(1, len(sorted_lk) + 1) / len(sorted_lk)
ax4.plot(sorted_lk, cdf_lk * 100, linewidth=2, color=blue, label='Like->Unlike')

sorted_dlk = np.sort(dlk_intervals)
cdf_dlk = np.arange(1, len(sorted_dlk) + 1) / len(sorted_dlk)
ax4.plot(sorted_dlk, cdf_dlk * 100, linewidth=2, color=orange, label='Dislike->Undislike')

ax4.set_xlabel('Time Interval (Hours)', fontsize=12)
ax4.set_ylabel('Cumulative %', fontsize=12)
ax4.set_title('CDF Comparison', fontsize=14)
ax4.axhline(50, color='gray', linestyle='--', alpha=0.5)
ax4.axhline(90, color='gray', linestyle='--', alpha=0.5)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 168)

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/regret_interval_clean.png", bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nSaved: {OUTPUT_PATH}/regret_interval_clean.png")

# 打印详细统计
print("\n========== 关键统计 ==========")
print(f"<=1小时: {len(lk_intervals[lk_intervals <= 1])} ({len(lk_intervals[lk_intervals <= 1])/len(lk_intervals)*100:.1f}%)")
print(f"<=24小时(1天): {len(lk_intervals[lk_intervals <= 24])} ({len(lk_intervals[lk_intervals <= 24])/len(lk_intervals)*100:.1f}%)")
print(f"<=168小时(7天): {len(lk_intervals[lk_intervals <= 168])} ({len(lk_intervals[lk_intervals <= 168])/len(lk_intervals)*100:.1f}%)")
print(f"<=720小时(30天): {len(lk_intervals[lk_intervals <= 720])} ({len(lk_intervals[lk_intervals <= 720])/len(lk_intervals)*100:.1f}%)")
