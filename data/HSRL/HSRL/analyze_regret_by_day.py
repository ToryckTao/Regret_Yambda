"""
Regret Interval Analysis - Full range by days
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

for df in [likes, unlikes]:
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
like_unlike['dt'] = (like_unlike['timestamp'] - like_unlike['t_lk']) / 3600.0  # hours

lk_intervals = like_unlike[like_unlike['dt'] > 0]['dt']
lk_days = lk_intervals / 24  # convert to days

max_days = int(np.ceil(lk_days.max()))
print(f"Max days: {max_days}")

# Create histogram with 1-day bins
bins = np.arange(0, max_days + 2) - 0.5  # center bins on integers
counts, bin_edges = np.histogram(lk_days, bins=bins)

# Create DataFrame for detailed view
day_df = pd.DataFrame({
    'day': range(1, max_days + 1),
    'count': counts[1:]  # exclude bin 0 (day 0)
})

print("\n========== 每天的后悔数量 ==========")
print(f"{'天数':<8} {'数量':<12} {'累计占比':<12}")
print("-" * 35)

cumsum = 0
for i, row in day_df.iterrows():
    cumsum += row['count']
    pct = cumsum / len(lk_days) * 100
    if row['count'] > 0:  # 只打印有数据的
        print(f"第{int(row['day']):>3}天   {int(row['count']):>8}   {pct:.1f}%")

print("-" * 35)
print(f"总计: {len(lk_days)}")

# 可视化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# 1. 全范围天数直方图
ax1.bar(day_df['day'], day_df['count'], width=1, edgecolor='black', alpha=0.7, color='#4A90D9')
ax1.set_xlabel('Days', fontsize=14)
ax1.set_ylabel('Count', fontsize=14)
ax1.set_title(f'Like->Unlike Regret Interval Distribution by Day (Full Range: 0-{max_days} days)', fontsize=15)
ax1.set_xlim(-1, max_days + 1)
ax1.grid(True, alpha=0.3, axis='y')

# 添加中位数线
median_day = np.median(lk_days)
ax1.axvline(median_day, color='red', linestyle='--', linewidth=2, 
            label=f'Median: {median_day:.1f} days')
ax1.legend(fontsize=12)
ax1.text(0.98, 0.95, f'Total: {len(lk_days):,}', transform=ax1.transAxes, 
         ha='right', va='top', fontsize=12, fontweight='bold')

# 2. 前60天的详细图（更容易看清）
ax2.bar(day_df[day_df['day'] <= 60]['day'], 
        day_df[day_df['day'] <= 60]['count'], 
        width=1, edgecolor='black', alpha=0.7, color='#4A90D9')
ax2.set_xlabel('Days', fontsize=14)
ax2.set_ylabel('Count', fontsize=14)
ax2.set_title('Like->Unlike Regret Interval (First 60 Days)', fontsize=15)
ax2.set_xlim(-1, 61)
ax2.grid(True, alpha=0.3, axis='y')

# 添加一些关键天数标注
key_days = [1, 7, 14, 30, 60]
for d in key_days:
    count_at_d = day_df[day_df['day'] == d]['count'].values[0] if d <= max_days else 0
    ax2.annotate(f'{int(count_at_d)}', xy=(d, count_at_d), xytext=(d, count_at_d + 500),
                ha='center', fontsize=9)

ax2.axvline(median_day, color='red', linestyle='--', linewidth=2, 
            label=f'Median: {median_day:.1f} days')
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/regret_interval_by_day.png", bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n已保存: {OUTPUT_PATH}/regret_interval_by_day.png")

# 打印简洁的统计
print("\n========== 关键统计 ==========")
print(f"中位数: {median_day:.1f} 天")
print(f"平均: {lk_days.mean():.1f} 天")
print(f"25%分位: {np.percentile(lk_days, 25):.1f} 天")
print(f"75%分位: {np.percentile(lk_days, 75):.1f} 天")
