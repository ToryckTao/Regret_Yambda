"""
Yambda Dataset Descriptive Statistics - Professional Academic Figures
Update: Two CR lines in Fig 2, Removed Mean/Median dots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================================
# Configuration 
# ============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 15,          
    'axes.labelsize': 16,     
    'axes.titlesize': 17,     
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,    
    'legend.fontsize': 14,    
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
})

sns.set_theme(style="ticks")

COLOR_SCHEMES = {
    "Scheme1_Soft": {
        'base_blue': '#8EBAE5',   
        'accent_flesh': '#FFDAB9',
        'base_green': '#A1D99B',  
        'accent_red': '#F4A261',  
        'name': 'fig_v1_soft_blue_peach.png'
    },
    "Scheme2_Pastel": {
        'base_blue': '#B3CDE3',   
        'accent_flesh': '#FBB4AE',
        'base_green': '#CCEBC5',  
        'accent_red': '#FB8072',  
        'name': 'fig_v2_pastel_tones.png'
    },
    "Scheme3_Neutral": {
        'base_blue': '#C6DBEF',   
        'accent_flesh': '#FDD0A2',
        'base_green': '#C7E9C0',  
        'accent_red': '#FDAE6B',  
        'name': 'fig_v3_neutral_light.png'
    }
}

DATA_PATH = "/root/autodl-tmp/data/HSRL/dataset/yambda_50m/raw/flat/50m"
OUTPUT_PATH = "/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hac/analysis"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 60)
print("Data Preprocessing (Strict Sequential Matching)...")
print("=" * 60)

# 1. 加载数据
multi_event = pd.read_parquet(f"{DATA_PATH}/multi_event.parquet")

likes = pd.read_parquet(f"{DATA_PATH}/likes.parquet")[['uid', 'item_id', 'timestamp']]
unlikes = pd.read_parquet(f"{DATA_PATH}/unlikes.parquet")[['uid', 'item_id', 'timestamp']]
dislikes = pd.read_parquet(f"{DATA_PATH}/dislikes.parquet")[['uid', 'item_id', 'timestamp']]
undislikes = pd.read_parquet(f"{DATA_PATH}/undislikes.parquet")[['uid', 'item_id', 'timestamp']]

# 2. 强制排序
for df in [likes, unlikes, dislikes, undislikes]:
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df.sort_values('timestamp', inplace=True)

# 3. 严格匹配
like_unlike = pd.merge_asof(
    unlikes, likes.rename(columns={'timestamp': 't_lk'}),
    left_on='timestamp', right_on='t_lk', by=['uid', 'item_id'], direction='backward'
)
like_unlike = like_unlike.dropna(subset=['t_lk'])
like_unlike['dt'] = (like_unlike['timestamp'] - like_unlike['t_lk']) / 3600.0

dislike_undislike = pd.merge_asof(
    undislikes, dislikes.rename(columns={'timestamp': 't_dlk'}),
    left_on='timestamp', right_on='t_dlk', by=['uid', 'item_id'], direction='backward'
)
dislike_undislike = dislike_undislike.dropna(subset=['t_dlk'])
dislike_undislike['dt'] = (dislike_undislike['timestamp'] - dislike_undislike['t_dlk']) / 3600.0

lk_intervals = like_unlike[like_unlike['dt'] > 0]['dt']
dlk_intervals = dislike_undislike[dislike_undislike['dt'] > 0]['dt']

print(f"匹配成功: Like->Unlike {len(lk_intervals)} 条, Dislike->Undislike {len(dlk_intervals)} 条")

listens = pd.read_parquet(f"{DATA_PATH}/listens.parquet")
played_ratio = listens['played_ratio_pct'].dropna()
played_ratio_filtered = played_ratio[played_ratio > 0]

print("Data loaded. Generating schemes...")

# ============================================================================
# Plotting Loop
# ============================================================================
for scheme_key, colors in COLOR_SCHEMES.items():
    print(f"\nGenerating {scheme_key}...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- [Figure 1] Feedback Action Distribution ---
    feedback_events = multi_event[multi_event['event_type'] != 'listen']
    event_counts = feedback_events['event_type'].value_counts().sort_index()

    order = ['like', 'dislike', 'unlike', 'undislike']
    labels = ['Like', 'Dislike', 'Unlike', 'Undislike']
    
    bar_colors = [colors['base_blue'], colors['base_blue'], colors['accent_red'], colors['accent_red']]
    values = [event_counts.get(e, 0) for e in order]
    total = sum(values)
    original_count = values[0] + values[1]
    regret_count = values[2] + values[3]

    bars = ax1.bar(range(len(values)), values, color=bar_colors, edgecolor='black', linewidth=0.8, width=0.6)

    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02, 
                 f'{val/1e6:.2f}M\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=13) 

    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=15) 
    ax1.set_ylabel('Count', fontsize=16)     
    ax1.set_ylim(0, max(values) * 1.35)
    
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax1.yaxis.get_offset_text().set_fontsize(11) 
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)

    ax_in = ax1.inset_axes([0.5, 0.5, 0.45, 0.45])
    ax_in.pie([original_count, regret_count], labels=['Original', 'Regret'], 
              colors=[colors['base_blue'], colors['accent_red']], autopct='%1.1f%%', 
              startangle=90, textprops={'fontsize': 11}) 

    # --- [Figure 2] Regret Interval Distribution ---
    log10_lk_intervals = np.log10(lk_intervals)
    log10_dlk_intervals = np.log10(dlk_intervals)
    
    min_log_dt = min(log10_lk_intervals.min(), log10_dlk_intervals.min())
    max_log_dt = max(log10_lk_intervals.max(), log10_dlk_intervals.max())
    
    bins_fig2 = np.linspace(min_log_dt, max_log_dt, 38) 
    
    counts_lk, _ = np.histogram(log10_lk_intervals, bins=bins_fig2)
    counts_dlk, _ = np.histogram(log10_dlk_intervals, bins=bins_fig2)
    
    log_lk = np.zeros_like(counts_lk, dtype=float)
    log_lk[counts_lk > 0] = np.log10(counts_lk[counts_lk > 0])
    
    log_dlk = np.zeros_like(counts_dlk, dtype=float)
    log_dlk[counts_dlk > 0] = np.log10(counts_dlk[counts_dlk > 0])
    
    widths = np.diff(bins_fig2)
    
    ax2.bar(bins_fig2[:-1], log_lk, width=widths, align='edge', 
            color=colors['base_blue'], edgecolor='black', linewidth=0.5, alpha=0.7, label='Like → Unlike')
    ax2.bar(bins_fig2[:-1], log_dlk, width=widths, align='edge', 
            color=colors['accent_flesh'], edgecolor='black', linewidth=0.5, alpha=0.85, label='Dislike → Undislike')

    ax2.set_xlim(np.floor(min_log_dt), np.ceil(max_log_dt))
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
    
    ax2.set_xlabel(r'$\log_{10}(\mathrm{Time\ Interval})$ (Hours)', fontsize=16) 
    ax2.set_ylabel(r'$\log_{10}(\mathrm{Count})$', fontsize=16) 
    
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    tick_positions = [-2, -1, 0, 1, 2, 3]
    tick_labels = ['36s', '6min', '1h', '10h', '4d', '42d']
    ax2_top.set_xticks(tick_positions)
    ax2_top.set_xticklabels(tick_labels, fontsize=10)
    
    max_log_count = max(log_lk.max(), log_dlk.max()) if len(log_lk) > 0 else 4
    y_upper_limit = np.ceil(max_log_count) + 0.5
    
    ax2.set_yticks(np.arange(1, y_upper_limit + 0.5, 1))
    ax2.set_ylim(1, y_upper_limit) 
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4)
    
    # Cumulative Records (CR) - 两条折线
    ax2_cr = ax2.twinx()
    
    sorted_lk = np.sort(lk_intervals)
    cumulative_cr_lk = np.arange(1, len(sorted_lk) + 1)
    
    sorted_dlk = np.sort(dlk_intervals)
    cumulative_cr_dlk = np.arange(1, len(sorted_dlk) + 1)
    
    # 1. Like -> Unlike (深灰色)
    ax2_cr.plot(np.log10(sorted_lk), cumulative_cr_lk, color='#4F4F4F', linestyle='--', linewidth=1.5, alpha=0.9, label='CR (Like → Unlike)')
    # 2. Dislike -> Undislike (深蓝色，可区分)
    ax2_cr.plot(np.log10(sorted_dlk), cumulative_cr_dlk, color='#1565C0', linestyle='--', linewidth=1.5, alpha=0.9, label='CR (Dislike → Undislike)')
    
    ax2_cr.set_ylabel('Cumulative Records', fontsize=16, color='black') 
    ax2_cr.tick_params(axis='y', colors='black')
    
    # 取两个数组中最大长度来设置右侧Y轴的上限
    ax2_cr.set_ylim(0, max(len(sorted_lk), len(sorted_dlk)) * 1.05)
    
    ax2_cr.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax2_cr.yaxis.get_offset_text().set_fontsize(11)

    # 聚合所有图例 (删除了散点代码，保持画面纯净)
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_cr.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=False, fontsize=11)

    # --- [Figure 3] Played Ratio Distribution ---
    bins_fig3 = np.linspace(-5, 120, 51)
    
    counts_pr, _ = np.histogram(played_ratio_filtered, bins=bins_fig3)
    widths_pr = np.diff(bins_fig3)
    
    ax3.bar(bins_fig3[:-1], counts_pr, width=widths_pr, align='edge', 
            color=colors['base_green'], edgecolor='black', linewidth=0.5, alpha=0.7, label='Count')

    ax3.set_xlabel('Played Ratio (%)', fontsize=16) 
    ax3.set_xlim(-5, 120)
    ax3.set_ylabel('Count', fontsize=16)            
    
    max_count = max(counts_pr) if len(counts_pr) > 0 else 1e6
    ax3.set_ylim(0, max_count * 1.15) 
    
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax3.yaxis.get_offset_text().set_fontsize(11) 
    ax3.yaxis.grid(True, linestyle='--', alpha=0.4)

    ax3_cr = ax3.twinx()
    sorted_ratio = np.sort(played_ratio)
    cumulative_cr3 = np.arange(1, len(sorted_ratio) + 1)
    
    ax3_cr.plot(sorted_ratio, cumulative_cr3, color=colors['accent_red'], linewidth=2.5, label='Cumulative Records')
    ax3_cr.set_ylabel('Cumulative Records', fontsize=16, color='black') 
    ax3_cr.tick_params(axis='y', colors='black')
    ax3_cr.set_ylim(0, len(sorted_ratio) * 1.05)
    
    ax3_cr.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
    ax3_cr.yaxis.get_offset_text().set_fontsize(11)
    
    ax3_cr.axvline(100, color='#969696', linestyle='--', linewidth=1.2, alpha=0.8)
    
    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_3, labels_3 = ax3_cr.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_3, labels_1 + labels_3, loc='upper left', frameon=False, fontsize=14) 

    # ============================================================================
    # Final Layout and Save
    # ============================================================================
    plt.tight_layout()
    save_path = f"{OUTPUT_PATH}/{colors['name']}"
    fig.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close(fig) 
    print(f"  ✓ Saved: {colors['name']}")

print("\n" + "=" * 60)
print("Done.")