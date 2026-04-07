"""
Regret Interval Analysis - Using merge_asof for strict sequential matching
Like->Unlike and Dislike->Undislike time intervals
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

sns.set_theme(style="ticks")

DATA_PATH = "/root/autodl-tmp/data/HSRL/dataset/yambda_50m/raw/flat/50m"
OUTPUT_PATH = "/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hac/analysis"

print("=" * 60)
print("Loading data for regret interval analysis...")
print("=" * 60)

# 1. Load data
likes = pd.read_parquet(f"{DATA_PATH}/likes.parquet")[['uid', 'item_id', 'timestamp']]
unlikes = pd.read_parquet(f"{DATA_PATH}/unlikes.parquet")[['uid', 'item_id', 'timestamp']]
dislikes = pd.read_parquet(f"{DATA_PATH}/dislikes.parquet")[['uid', 'item_id', 'timestamp']]
undislikes = pd.read_parquet(f"{DATA_PATH}/undislikes.parquet")[['uid', 'item_id', 'timestamp']]

print(f"Likes: {len(likes)}, Unlikes: {len(unlikes)}")
print(f"Dislikes: {len(dislikes)}, Undislikes: {len(undislikes)}")

# 2. Data type conversion and sorting (required for merge_asof)
for df in [likes, unlikes, dislikes, undislikes]:
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df.sort_values('timestamp', inplace=True)

# 3. Strict sequential matching using merge_asof
# Match each unlike to the most recent like before it
like_unlike = pd.merge_asof(
    unlikes, 
    likes.rename(columns={'timestamp': 't_lk'}),
    left_on='timestamp', 
    right_on='t_lk',
    by=['uid', 'item_id'],
    direction='backward'
)
like_unlike = like_unlike.dropna(subset=['t_lk'])
like_unlike['dt'] = (like_unlike['timestamp'] - like_unlike['t_lk']) / 3600.0  # hours

# Match each undislike to the most recent dislike before it
dislike_undislike = pd.merge_asof(
    undislikes, 
    dislikes.rename(columns={'timestamp': 't_dlk'}),
    left_on='timestamp', 
    right_on='t_dlk',
    by=['uid', 'item_id'],
    direction='backward'
)
dislike_undislike = dislike_undislike.dropna(subset=['t_dlk'])
dislike_undislike['dt'] = (dislike_undislike['timestamp'] - dislike_undislike['t_dlk']) / 3600.0  # hours

# Extract valid intervals (dt > 0)
lk_intervals = like_unlike[like_unlike['dt'] > 0]['dt']
dlk_intervals = dislike_undislike[dislike_undislike['dt'] > 0]['dt']

print(f"\n匹配成功: Like->Unlike {len(lk_intervals)} 条, Dislike->Undislike {len(dlk_intervals)} 条")

# Convert to days for easier interpretation
lk_days = lk_intervals / 24
dlk_days = dlk_intervals / 24

print(f"\n========== Like->Unlike 后悔间隔统计 ==========")
print(f"  记录数: {len(lk_intervals)}")
print(f"  最小值: {lk_intervals.min():.2f} 小时 ({lk_days.min():.2f} 天)")
print(f"  最大值: {lk_intervals.max():.2f} 小时 ({lk_days.max():.2f} 天)")
print(f"  中位数: {lk_intervals.median():.2f} 小时 ({lk_days.median():.2f} 天)")
print(f"  平均值: {lk_intervals.mean():.2f} 小时 ({lk_days.mean():.2f} 天)")

for p in [10, 25, 50, 75, 90, 95]:
    val_h = np.percentile(lk_intervals, p)
    val_d = val_h / 24
    print(f"  {p}%分位: {val_h:.2f} 小时 ({val_d:.2f} 天)")

print(f"\n========== Dislike->Undislike 后悔间隔统计 ==========")
print(f"  记录数: {len(dlk_intervals)}")
print(f"  最小值: {dlk_intervals.min():.2f} 小时 ({dlk_days.min():.2f} 天)")
print(f"  最大值: {dlk_intervals.max():.2f} 小时 ({dlk_days.max():.2f} 天)")
print(f"  中位数: {dlk_intervals.median():.2f} 小时 ({dlk_days.median():.2f} 天)")
print(f"  平均值: {dlk_intervals.mean():.2f} 小时 ({dlk_days.mean():.2f} 天)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Color scheme
blue = '#4A90D9'
orange = '#F5A623'
green = '#7ED321'
purple = '#9013FE'

# 1. Histogram of Like->Unlike (hours, <= 7 days)
ax1 = axes[0, 0]
ax1.hist(lk_intervals[lk_intervals <= 168], bins=50, edgecolor='black', alpha=0.7, color=blue)
ax1.set_xlabel('Regret Interval (Hours)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Like->Unlike Distribution (Hours, <=7 days)', fontsize=14)
ax1.axvline(lk_intervals.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {lk_intervals.median():.1f}h')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Histogram of Like->Unlike (days, <= 30 days)
ax2 = axes[0, 1]
ax2.hist(lk_days[lk_days <= 30], bins=30, edgecolor='black', alpha=0.7, color=blue)
ax2.set_xlabel('Regret Interval (Days)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Like->Unlike Distribution (Days, <=30 days)', fontsize=14)
ax2.axvline(lk_days.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {lk_days.median():.1f}d')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Log scale comparison
ax3 = axes[1, 0]
log_lk = np.log10(lk_intervals[lk_intervals > 0])
log_dlk = np.log10(dlk_intervals[dlk_intervals > 0])
ax3.hist(log_lk, bins=40, alpha=0.6, color=blue, edgecolor='black', label=f'Like->Unlike (n={len(lk_intervals)})')
ax3.hist(log_dlk, bins=40, alpha=0.6, color=orange, edgecolor='black', label=f'Dislike->Undislike (n={len(dlk_intervals)})')
ax3.set_xlabel('log10(Regret Interval / Hours)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Regret Interval Distribution (Log Scale)', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. CDF comparison
ax4 = axes[1, 1]
sorted_lk = np.sort(lk_intervals)
cdf_lk = np.arange(1, len(sorted_lk) + 1) / len(sorted_lk)
ax4.plot(sorted_lk, cdf_lk, linewidth=2, color=blue, label='Like->Unlike')

sorted_dlk = np.sort(dlk_intervals)
cdf_dlk = np.arange(1, len(sorted_dlk) + 1) / len(sorted_dlk)
ax4.plot(sorted_dlk, cdf_dlk, linewidth=2, color=orange, label='Dislike->Undislike')

ax4.set_xlabel('Regret Interval (Hours)', fontsize=12)
ax4.set_ylabel('Cumulative Probability', fontsize=12)
ax4.set_title('CDF Comparison', fontsize=14)
ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax4.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 168)

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/regret_interval_correct.png", bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n图表已保存到: {OUTPUT_PATH}/regret_interval_correct.png")
print("=" * 60)
print("Done.")
