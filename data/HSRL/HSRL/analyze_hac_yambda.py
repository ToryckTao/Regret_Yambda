#!/usr/bin/env python3
"""
HAC (Hierarchical Actor-Critic) Training Log Analyzer - Academic Publication Quality Visualization
Parses HAC training logs and generates training curves and evaluation dashboards.
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from typing import Dict, List, Any, Optional
import os


def parse_train_log(filepath: str) -> Dict[str, Any]:
    """
    Parse HAC training log file.

    Args:
        filepath: Path to the training log file (model.report)

    Returns:
        Dictionary containing:
            - hyperparameters: dict of extracted hyperparameters
            - metrics: DataFrame with step-level metrics
            - summary: dict of aggregated metrics
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    # Extract hyperparameters from Namespace lines
    hyperparameters = {
        'agent_class': None,
        'actor_lr': None,
        'critic_lr': None,
        'noise_var': None,
        'behavior_lr': None,
        'hyper_actor_coef': None,
        'n_iter': None,
        'gamma': None,
        'buffer_size': None,
        'episode_batch_size': None,
        'batch_size': None
    }

    # Find agent_class
    for line in lines:
        if line.startswith('Namespace(') and "agent_class='" in line:
            agent_match = re.search(r"agent_class='(\w+)'", line)
            if agent_match:
                hyperparameters['agent_class'] = agent_match.group(1)
            break

    # Find hyperparameters from the agent setup Namespace
    for line in lines:
        if line.startswith('Namespace(') and 'actor_lr=' in line:
            # Extract actor_lr
            actor_lr_match = re.search(r'actor_lr=([0-9.e\-]+)', line)
            if actor_lr_match:
                try:
                    hyperparameters['actor_lr'] = float(actor_lr_match.group(1))
                except ValueError:
                    pass

            # Extract critic_lr
            critic_lr_match = re.search(r'critic_lr=([0-9.e\-]+)', line)
            if critic_lr_match:
                try:
                    hyperparameters['critic_lr'] = float(critic_lr_match.group(1))
                except ValueError:
                    pass

            # Extract noise_var
            noise_match = re.search(r'noise_var=([0-9.e\-]+)', line)
            if noise_match:
                try:
                    hyperparameters['noise_var'] = float(noise_match.group(1))
                except ValueError:
                    pass

            # Extract behavior_lr
            behavior_lr_match = re.search(r'behavior_lr=([0-9.e\-]+)', line)
            if behavior_lr_match:
                try:
                    hyperparameters['behavior_lr'] = float(behavior_lr_match.group(1))
                except ValueError:
                    pass

            # Extract hyper_actor_coef
            hyper_actor_coef_match = re.search(r'hyper_actor_coef=([0-9.e\-]+)', line)
            if hyper_actor_coef_match:
                try:
                    hyperparameters['hyper_actor_coef'] = float(hyper_actor_coef_match.group(1))
                except ValueError:
                    pass

            # Extract n_iter
            n_iter_match = re.search(r'n_iter=\[?([0-9]+)\]?', line)
            if n_iter_match:
                try:
                    hyperparameters['n_iter'] = int(n_iter_match.group(1))
                except ValueError:
                    pass

            # Extract gamma
            gamma_match = re.search(r'gamma=([0-9.e\-]+)', line)
            if gamma_match:
                try:
                    hyperparameters['gamma'] = float(gamma_match.group(1))
                except ValueError:
                    pass

            # Extract episode_batch_size
            ep_bs_match = re.search(r'episode_batch_size=([0-9]+)', line)
            if ep_bs_match:
                try:
                    hyperparameters['episode_batch_size'] = int(ep_bs_match.group(1))
                except ValueError:
                    pass

            # Extract batch_size
            batch_size_match = re.search(r'batch_size=([0-9]+)', line)
            if batch_size_match:
                try:
                    hyperparameters['batch_size'] = int(batch_size_match.group(1))
                except ValueError:
                    pass

            break

    # Extract training metrics
    step_data = []
    # Pattern for HAC logs (includes max_n_step, min_n_step)
    pattern = r"step:\s*(\d+)\s*@\s*episode report:\s*\{([^}]+)\}"

    for match in re.finditer(pattern, content):
        step = int(match.group(1))
        metrics_str = match.group(2)

        # Extract metrics using regex - support both np.float32 and np.float64
        avg_reward_match = re.search(r"'average_total_reward':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        reward_var_match = re.search(r"'reward_variance':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        max_reward_match = re.search(r"'max_total_reward':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        min_reward_match = re.search(r"'min_total_reward':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        avg_n_step_match = re.search(r"'average_n_step':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        max_n_step_match = re.search(r"'max_n_step':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        min_n_step_match = re.search(r"'min_n_step':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        buffer_size_match = re.search(r"'buffer_size':\s*(\d+)", metrics_str)

        # Extract losses
        critic_loss_match = re.search(r"'critic_loss':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        actor_loss_match = re.search(r"'actor_loss':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        behavior_loss_match = re.search(r"'behavior_loss':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)

        if avg_reward_match and avg_n_step_match:
            avg_reward = float(avg_reward_match.group(1))
            avg_n_step = float(avg_n_step_match.group(1))

            if avg_reward > 0:  # Filter out initial step with 0 rewards
                step_data.append({
                    'step': step,
                    'average_total_reward': avg_reward,
                    'reward_variance': float(reward_var_match.group(1)) if reward_var_match else np.nan,
                    'max_total_reward': float(max_reward_match.group(1)) if max_reward_match else np.nan,
                    'min_total_reward': float(min_reward_match.group(1)) if min_reward_match else np.nan,
                    'average_n_step': avg_n_step,
                    'max_n_step': float(max_n_step_match.group(1)) if max_n_step_match else np.nan,
                    'min_n_step': float(min_n_step_match.group(1)) if min_n_step_match else np.nan,
                    'buffer_size': int(buffer_size_match.group(1)) if buffer_size_match else 0,
                    'critic_loss': float(critic_loss_match.group(1)) if critic_loss_match and critic_loss_match.group(1) != 'nan' else np.nan,
                    'actor_loss': float(actor_loss_match.group(1)) if actor_loss_match and actor_loss_match.group(1) != 'nan' else np.nan,
                    'behavior_loss': float(behavior_loss_match.group(1)) if behavior_loss_match and behavior_loss_match.group(1) != 'nan' else np.nan,
                })

    df = pd.DataFrame(step_data)

    # Compute summary statistics
    summary = {}
    if not df.empty:
        summary['mean_reward'] = df['average_total_reward'].mean()
        summary['peak_reward'] = df['max_total_reward'].max()
        summary['mean_depth'] = df['average_n_step'].mean()
        summary['std_reward'] = df['average_total_reward'].std()
        summary['std_depth'] = df['average_n_step'].std()
        summary['final_reward'] = df['average_total_reward'].iloc[-1] if len(df) > 0 else 0
        summary['final_buffer'] = df['buffer_size'].iloc[-1] if len(df) > 0 else 0
        summary['mean_max_n_step'] = df['max_n_step'].mean()
        summary['mean_min_n_step'] = df['min_n_step'].mean()

        # Compute reward trend (last 10% vs first 10%)
        n = len(df)
        first_10 = df['average_total_reward'].iloc[:max(1, n//10)].mean()
        last_10 = df['average_total_reward'].iloc[-max(1, n//10):].mean()
        summary['reward_improvement'] = last_10 - first_10

    return {
        'hyperparameters': hyperparameters,
        'metrics': df,
        'summary': summary
    }


def plot_training_curves(data: Dict, output_path: str = 'hac_training_curves.png'):
    """
    Generate training curve visualization for HAC.

    Args:
        data: Output of parse_train_log
        output_path: Path to save the output image
    """
    # Set academic publication style
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.dpi'] = 150

    df = data['metrics']
    hp = data['hyperparameters']
    summary = data['summary']

    if df.empty:
        print("Warning: No data to plot!")
        return

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Color palette
    primary_color = '#2E86AB'  # Steel blue
    secondary_color = '#A23B72'  # Raspberry
    tertiary_color = '#F18F01'  # Orange
    quaternary_color = '#C73E1D'  # Red

    # ========== Subplot 1: Reward Curve ==========
    ax1 = axes[0, 0]

    # Raw data with moving average
    window = max(5, len(df) // 20)
    df['reward_ma'] = df['average_total_reward'].rolling(window=window, min_periods=1).mean()

    ax1.plot(df['step'], df['average_total_reward'], alpha=0.3, color=primary_color, linewidth=0.8)
    ax1.plot(df['step'], df['reward_ma'], color=primary_color, linewidth=2, label=f'Moving Avg (window={window})')

    # Add confidence band
    df['reward_std'] = df['average_total_reward'].rolling(window=window, min_periods=1).std()
    ax1.fill_between(df['step'],
                      df['reward_ma'] - df['reward_std'],
                      df['reward_ma'] + df['reward_std'],
                      alpha=0.2, color=primary_color)

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Average Total Reward')
    ax1.set_title('(a) Training Reward Curve')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # ========== Subplot 2: Episode Depth ==========
    ax2 = axes[0, 1]

    df['depth_ma'] = df['average_n_step'].rolling(window=window, min_periods=1).mean()

    ax2.plot(df['step'], df['average_n_step'], alpha=0.3, color=secondary_color, linewidth=0.8)
    ax2.plot(df['step'], df['depth_ma'], color=secondary_color, linewidth=2, label=f'Moving Avg (window={window})')

    # Plot max/min depth range
    if 'max_n_step' in df.columns and df['max_n_step'].notna().any():
        ax2.fill_between(df['step'], df['min_n_step'], df['max_n_step'],
                        alpha=0.15, color=secondary_color, label='Min-Max Range')

    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Average Episode Depth (n_step)')
    ax2.set_title('(b) Episode Depth over Training')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    # ========== Subplot 3: Loss Curves ==========
    ax3 = axes[1, 0]

    # Plot losses if available
    has_valid_loss = False
    if 'critic_loss' in df.columns and df['critic_loss'].notna().any():
        df['critic_loss_ma'] = df['critic_loss'].rolling(window=window, min_periods=1).mean()
        ax3.plot(df['step'], df['critic_loss'], alpha=0.3, color=tertiary_color, linewidth=0.8)
        ax3.plot(df['step'], df['critic_loss_ma'], color=tertiary_color, linewidth=2, label='Critic Loss')
        has_valid_loss = True

    if 'actor_loss' in df.columns and df['actor_loss'].notna().any():
        df['actor_loss_scaled'] = -df['actor_loss']  # Flip sign since actor loss is negative
        df['actor_loss_ma'] = df['actor_loss_scaled'].rolling(window=window, min_periods=1).mean()
        ax3.plot(df['step'], df['actor_loss_scaled'], alpha=0.3, color=quaternary_color, linewidth=0.8)
        ax3.plot(df['step'], df['actor_loss_ma'], color=quaternary_color, linewidth=2, label='Actor Loss (inverted)')
        has_valid_loss = True

    if not has_valid_loss:
        ax3.text(0.5, 0.5, 'Loss data not available', ha='center', va='center', transform=ax3.transAxes)

    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss Value')
    ax3.set_title('(c) Training Losses')
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(left=0)

    # ========== Subplot 4: Replay Buffer Size ==========
    ax4 = axes[1, 1]

    ax4.plot(df['step'], df['buffer_size'], color=primary_color, linewidth=2)
    ax4.fill_between(df['step'], 0, df['buffer_size'], alpha=0.3, color=primary_color)

    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Replay Buffer Size')
    ax4.set_title('(d) Replay Buffer Growth')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(left=0)
    ax4.set_ylim(bottom=0)

    # ========== Global Title ==========
    agent_name = hp.get('agent_class', 'HAC')
    iter_count = hp.get('n_iter', 'N/A')
    hyper_actor_coef = hp.get('hyper_actor_coef', 'N/A')
    fig.suptitle(f'HAC ({agent_name}) Training Analysis\n'
                 f'Iterations: {iter_count}, Actor LR: {hp.get("actor_lr", "N/A")}, '
                 f'Hyper Actor Coef: {hyper_actor_coef}',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Training curves saved to: {output_path}")


def plot_evaluation_summary(data: Dict, output_path: str = 'hac_eval_summary.png'):
    """
    Generate evaluation summary visualization for HAC.

    Args:
        data: Output of parse_train_log
        output_path: Path to save the output image
    """
    # Set academic publication style
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.dpi'] = 150

    df = data['metrics']
    hp = data['hyperparameters']
    summary = data['summary']

    if df.empty:
        print("Warning: No data to plot!")
        return

    fig = plt.figure(figsize=(14, 6))

    # Color palette
    primary_color = '#2E86AB'

    # ========== Subplot 1: Reward Distribution (Histogram + KDE) ==========
    ax1 = fig.add_subplot(1, 2, 1)

    rewards = df['average_total_reward'].values

    # Histogram with KDE overlay
    sns.histplot(rewards, bins=30, kde=True, ax=ax1, color=primary_color,
                 edgecolor='white', alpha=0.7)

    # Add mean line
    mean_reward = np.mean(rewards)
    ax1.axvline(mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')

    ax1.set_xlabel('Average Total Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(a) Reward Distribution')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ========== Subplot 2: Summary Table ==========
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')

    # Helper to format hyperparameter values
    def format_hp(value, suffix=''):
        if value is None:
            return 'N/A'
        if isinstance(value, float):
            if value == 0:
                return '0'
            return f'{value:.2e}{suffix}' if value < 0.01 else f'{value:.4f}{suffix}'
        if isinstance(value, int):
            return f'{value:,}'
        return str(value)

    # Prepare table data
    table_data = [
        # Test Results Section
        ['', 'Value'],
        ['Mean Reward', f"{summary.get('mean_reward', 0):.2f}"],
        ['Peak Reward', f"{summary.get('peak_reward', 0):.2f}"],
        ['Reward Std', f"{summary.get('std_reward', 0):.2f}"],
        ['Mean Depth (n_step)', f"{summary.get('mean_depth', 0):.1f}"],
        ['Depth Std', f"{summary.get('std_depth', 0):.1f}"],
        ['Mean Max Depth', f"{summary.get('mean_max_n_step', 0):.1f}"],
        ['Mean Min Depth', f"{summary.get('mean_min_n_step', 0):.1f}"],
        ['', ''],  # Divider
        # Training Summary
        ['Final Reward', f"{summary.get('final_reward', 0):.2f}"],
        ['Reward Improvement', f"{summary.get('reward_improvement', 0):.2f}"],
        ['Final Buffer Size', f"{summary.get('final_buffer', 0):,}"],
        ['', ''],  # Divider
        # Hyperparameters
        ['Agent', hp.get('agent_class', 'N/A')],
        ['Actor LR', format_hp(hp.get('actor_lr'))],
        ['Critic LR', format_hp(hp.get('critic_lr'))],
        ['Behavior LR (BC)', format_hp(hp.get('behavior_lr'))],
        ['Hyper Actor Coef', format_hp(hp.get('hyper_actor_coef'))],
        ['Noise Var', format_hp(hp.get('noise_var'))],
        ['Gamma', format_hp(hp.get('gamma'))],
        ['Episode Batch Size', format_hp(hp.get('episode_batch_size'))],
    ]

    # Create table
    table = ax2.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.4, 0.35]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Header row styling
    table[(0, 0)].set_facecolor('#404040')
    table[(0, 0)].set_text_props(color='white', fontweight='bold')
    table[(0, 1)].set_facecolor('#404040')
    table[(0, 1)].set_text_props(color='white', fontweight='bold')

    # Section dividers (find divider rows)
    for i, row in enumerate(table_data):
        if row[0] == '' and row[1] == '':
            for j in range(2):
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(2)

    # Styling for data rows
    for i in range(1, len(table_data)):
        if table_data[i][0] == '' and table_data[i][1] == '':
            continue  # Skip dividers
        for j in range(2):
            if i <= 9:  # Test results section
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F5F5F5')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')
            else:  # Hyperparameters section
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E8F4F8')
                else:
                    table[(i, j)].set_facecolor('#F0F8FF')

    # Highlight key metrics
    table[(1, 1)].set_text_props(fontweight='bold', color=primary_color)
    table[(2, 1)].set_text_props(fontweight='bold', color='#28A745')

    ax2.set_title('(b) Evaluation Summary', pad=20, fontweight='bold', fontsize=12)

    # ========== Global Title ==========
    fig.suptitle(f'HAC Training Evaluation Summary',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Evaluation summary saved to: {output_path}")


def plot_convergence_analysis(data: Dict, output_path: str = 'hac_convergence.png'):
    """
    Generate convergence analysis plot.

    Args:
        data: Output of parse_train_log
        output_path: Path to save the output image
    """
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.dpi'] = 150

    df = data['metrics']

    if df.empty or len(df) < 10:
        print("Warning: Not enough data for convergence analysis!")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    primary_color = '#2E86AB'

    # ========== Subplot 1: Cumulative Mean Reward ==========
    ax1 = axes[0]

    df['cumsum_reward'] = df['average_total_reward'].cumsum()
    df['cummean_reward'] = df['cumsum_reward'] / (df.index + 1)

    ax1.plot(df['step'], df['cummean_reward'], color=primary_color, linewidth=2)
    ax1.fill_between(df['step'], 0, df['cummean_reward'], alpha=0.3, color=primary_color)

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Cumulative Mean Reward')
    ax1.set_title('(a) Cumulative Mean Reward')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)

    # ========== Subplot 2: Reward Variance (Stability) ==========
    ax2 = axes[1]

    window = max(10, len(df) // 10)
    df['rolling_std'] = df['average_total_reward'].rolling(window=window, min_periods=1).std()

    ax2.plot(df['step'], df['rolling_std'], color='#E94F37', linewidth=2)
    ax2.fill_between(df['step'], 0, df['rolling_std'], alpha=0.3, color='#E94F37')

    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Rolling Std (window={})'.format(window))
    ax2.set_title('(b) Reward Stability (Lower is Better)')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)

    # ========== Subplot 3: Reward Range ==========
    ax3 = axes[2]

    df['rolling_max'] = df['max_total_reward'].rolling(window=window, min_periods=1).max()
    df['rolling_min'] = df['min_total_reward'].rolling(window=window, min_periods=1).min()

    ax3.fill_between(df['step'], df['rolling_min'], df['rolling_max'],
                     alpha=0.3, color=primary_color, label='Reward Range')
    ax3.plot(df['step'], df['rolling_max'], color='green', linewidth=1.5, label='Max Reward')
    ax3.plot(df['step'], df['rolling_min'], color='red', linewidth=1.5, label='Min Reward')

    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Reward Value')
    ax3.set_title('(c) Reward Range over Training')
    ax3.legend(loc='lower right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Convergence analysis saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse HAC training logs and generate visualization'
    )
    parser.add_argument('--log', type=str, required=True,
                        help='Path to HAC training log file (model.report)')
    parser.add_argument('--output_dir', type=str, default='output/yambda_hac/analysis',
                        help='Output directory for visualizations')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Parsing log: {args.log}")
    data = parse_train_log(args.log)

    hp = data['hyperparameters']
    summary = data['summary']

    print(f"\n=== Training Summary ===")
    print(f"Agent: {hp.get('agent_class', 'N/A')}")
    print(f"Actor LR: {hp.get('actor_lr', 'N/A')}")
    print(f"Critic LR: {hp.get('critic_lr', 'N/A')}")
    print(f"Behavior LR: {hp.get('behavior_lr', 'N/A')}")
    print(f"Hyper Actor Coef: {hp.get('hyper_actor_coef', 'N/A')}")
    print(f"Noise Var: {hp.get('noise_var', 'N/A')}")
    print(f"Iterations: {hp.get('n_iter', 'N/A')}")
    print(f"\n=== Results ===")
    print(f"Mean Reward: {summary.get('mean_reward', 0):.2f}")
    print(f"Peak Reward: {summary.get('peak_reward', 0):.2f}")
    print(f"Mean Depth: {summary.get('mean_depth', 0):.1f}")
    print(f"Reward Improvement: {summary.get('reward_improvement', 0):.2f}")
    print(f"Final Buffer Size: {summary.get('final_buffer', 0):,}")

    print(f"\nGenerating visualizations...")

    # Generate all plots
    base_name = os.path.splitext(os.path.basename(args.log))[0]

    plot_training_curves(data, os.path.join(args.output_dir, f'{base_name}_training_curves.png'))
    plot_evaluation_summary(data, os.path.join(args.output_dir, f'{base_name}_eval_summary.png'))
    plot_convergence_analysis(data, os.path.join(args.output_dir, f'{base_name}_convergence.png'))

    print(f"\nAll visualizations saved to: {args.output_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
