#!/usr/bin/env python3
"""
HAC Test Log Analyzer - Academic Publication Quality Visualization
Parses HAC and baseline test logs and generates comparison dashboards.
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from typing import Dict, List, Any, Optional


def parse_log_file(filepath: str) -> Dict[str, Any]:
    """
    Parse a single RL test log file (supports both DDPG and HAC).

    Args:
        filepath: Path to the test log file

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
        'entropy_coef': None,
        'behavior_lr': None,
        'hyper_actor_coef': None
    }

    # First pass: find agent_class from the first Namespace
    for line in lines:
        if line.startswith('Namespace(') and "agent_class='" in line:
            agent_match = re.search(r"agent_class='(\w+)'", line)
            if agent_match:
                hyperparameters['agent_class'] = agent_match.group(1)
            break

    # Second pass: find hyperparameters from the agent setup Namespace
    for line in lines:
        if line.startswith('Namespace(') and 'actor_lr=' in line:
            # Extract actor_lr (supports scientific notation like 3e-05 and decimal like 0.0001)
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

            # Extract entropy_coef
            entropy_match = re.search(r'entropy_coef=([0-9.e\-]+)', line)
            if entropy_match:
                try:
                    hyperparameters['entropy_coef'] = float(entropy_match.group(1))
                except ValueError:
                    pass

            # Extract behavior_lr
            behavior_lr_match = re.search(r'behavior_lr=([0-9.e\-]+)', line)
            if behavior_lr_match:
                try:
                    hyperparameters['behavior_lr'] = float(behavior_lr_match.group(1))
                except ValueError:
                    pass

            # Extract hyper_actor_coef (HAC specific)
            hyper_actor_coef_match = re.search(r'hyper_actor_coef=([0-9.e\-]+)', line)
            if hyper_actor_coef_match:
                try:
                    hyperparameters['hyper_actor_coef'] = float(hyper_actor_coef_match.group(1))
                except ValueError:
                    pass

            break

    # Extract evaluation metrics
    step_data = []
    pattern = r"step:\s*(\d+)\s*@\s*episode report:\s*\{([^}]+)\}"

    for match in re.finditer(pattern, content):
        step = int(match.group(1))
        metrics_str = match.group(2)

        # Extract metrics using regex - support both np.float32 and np.float64
        avg_reward_match = re.search(r"'average_total_reward':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        max_reward_match = re.search(r"'max_total_reward':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        avg_n_step_match = re.search(r"'average_n_step':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        max_n_step_match = re.search(r"'max_n_step':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        min_n_step_match = re.search(r"'min_n_step':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)

        if avg_reward_match and avg_n_step_match:
            avg_reward = float(avg_reward_match.group(1))
            max_reward = float(max_reward_match.group(1)) if max_reward_match else np.nan
            avg_n_step = float(avg_n_step_match.group(1))
            max_n_step = float(max_n_step_match.group(1)) if max_n_step_match else np.nan
            min_n_step = float(min_n_step_match.group(1)) if min_n_step_match else np.nan

            if avg_reward > 0:  # Filter out initial step with 0 rewards
                step_data.append({
                    'step': step,
                    'average_total_reward': avg_reward,
                    'max_total_reward': max_reward,
                    'average_n_step': avg_n_step,
                    'max_n_step': max_n_step,
                    'min_n_step': min_n_step
                })

    df = pd.DataFrame(step_data)

    # Compute summary statistics
    summary = {}
    if not df.empty:
        summary['mean_reward'] = df['average_total_reward'].mean()
        summary['peak_reward'] = df['max_total_reward'].max()
        summary['mean_depth'] = df['average_n_step'].mean()
        summary['std_reward'] = df['average_total_reward'].std()
        summary['std_depth'] = df['average_n_step'].mean()
        # HAC specific
        if 'max_n_step' in df.columns and df['max_n_step'].notna().any():
            summary['mean_max_n_step'] = df['max_n_step'].mean()
            summary['mean_min_n_step'] = df['min_n_step'].mean()

    return {
        'hyperparameters': hyperparameters,
        'metrics': df,
        'summary': summary
    }


def plot_comparison_dashboard(data_dict: Dict[str, Dict], output_path: str = 'comparison_dashboard.png'):
    """
    Generate a 1x3 academic-style comparison dashboard.

    Args:
        data_dict: Dictionary with keys for different models, each containing
                   the output of parse_log_file
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

    # Color palette - high contrast academic colors
    colors = {
        'HAC': '#2E86AB',       # Steel blue
        'DDPG': '#E94F37',      # Vermilion red
        'Yambda': '#F18F01',    # Orange
    }

    # Use default colors for additional models
    default_colors = ['#2E86AB', '#E94F37', '#F18F01', '#28A745', '#6F4E7C', '#00CED1']
    model_names = list(data_dict.keys())
    for i, name in enumerate(model_names):
        if name not in colors:
            colors[name] = default_colors[i % len(default_colors)]

    fig = plt.figure(figsize=(14, 5))

    # ========== Subplot 1: Total Reward Distribution (KDE) ==========
    ax1 = fig.add_subplot(1, 3, 1)

    for name, data in data_dict.items():
        rewards = data['metrics']['average_total_reward'].values
        sns.kdeplot(rewards, ax=ax1, color=colors.get(name, '#2E86AB'),
                   fill=True, alpha=0.3, linewidth=2, label=name)

    ax1.set_xlabel('Average Total Reward')
    ax1.set_ylabel('Density')
    ax1.set_title('Total Reward Distribution')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)

    # ========== Subplot 2: Depth Distribution (KDE) ==========
    ax2 = fig.add_subplot(1, 3, 2)

    for name, data in data_dict.items():
        depths = data['metrics']['average_n_step'].values
        sns.kdeplot(depths, ax=ax2, color=colors.get(name, '#2E86AB'),
                   fill=True, alpha=0.3, linewidth=2, label=name)

    ax2.set_xlabel('Average Episode Depth (n_step)')
    ax2.set_ylabel('Density')
    ax2.set_title('Depth Distribution')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)

    # ========== Subplot 3: Summary Table ==========
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis('off')

    # Prepare table data - determine max columns
    col_names = ['Metric'] + model_names

    # Calculate mean rewards and depths for comparison
    table_data = [col_names]
    table_data.append(['Mean Reward'] + [f"{data_dict[name]['summary'].get('mean_reward', 0):.2f}" for name in model_names])
    table_data.append(['Peak Reward'] + [f"{data_dict[name]['summary'].get('peak_reward', 0):.2f}" for name in model_names])
    table_data.append(['Mean Depth'] + [f"{data_dict[name]['summary'].get('mean_depth', 0):.1f}" for name in model_names])

    # Find best values for highlighting
    mean_rewards = {name: data_dict[name]['summary'].get('mean_reward', 0) for name in model_names}
    best_model = max(mean_rewards, key=mean_rewards.get)

    # Create table
    table = ax3.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.25] + [0.25] * len(model_names)
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Header row styling
    for j in range(len(col_names)):
        table[(0, j)].set_facecolor('#404040')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Alternating row colors
    for i in range(1, len(table_data)):
        for j in range(len(col_names)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    # Highlight best model values
    if best_model in model_names:
        col_idx = model_names.index(best_model) + 1
        table[(1, col_idx)].set_text_props(fontweight='bold', color=colors.get(best_model, '#2E86AB'))

    ax3.set_title('Test Results', pad=20, fontweight='bold', fontsize=12)

    # ========== Global Title ==========
    fig.suptitle('Evaluation Comparison: HAC vs Baselines', 
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Dashboard saved to: {output_path}")


def plot_comparison_dashboard_with_hyperparams(data_dict: Dict[str, Dict],
                                                output_path: str = 'comparison_dashboard.png'):
    """
    Generate a 2x2 academic-style comparison dashboard with hyperparameter table.

    Args:
        data_dict: Dictionary with keys for different models, each containing
                   the output of parse_log_file
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

    # Color palette
    colors = {
        'HAC': '#2E86AB',
        'DDPG': '#E94F37',
        'Yambda': '#F18F01',
    }
    default_colors = ['#2E86AB', '#E94F37', '#F18F01', '#28A745', '#6F4E7C', '#00CED1']
    model_names = list(data_dict.keys())
    for i, name in enumerate(model_names):
        if name not in colors:
            colors[name] = default_colors[i % len(default_colors)]

    fig = plt.figure(figsize=(14, 8))

    # ========== Subplot 1: Total Reward Distribution (KDE) ==========
    ax1 = fig.add_subplot(2, 2, 1)

    for name, data in data_dict.items():
        rewards = data['metrics']['average_total_reward'].values
        sns.kdeplot(rewards, ax=ax1, color=colors.get(name, '#2E86AB'),
                   fill=True, alpha=0.3, linewidth=2, label=name)

    ax1.set_xlabel('Average Total Reward')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) Total Reward Distribution')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)

    # ========== Subplot 2: Depth Distribution (KDE) ==========
    ax2 = fig.add_subplot(2, 2, 2)

    for name, data in data_dict.items():
        depths = data['metrics']['average_n_step'].values
        sns.kdeplot(depths, ax=ax2, color=colors.get(name, '#2E86AB'),
                   fill=True, alpha=0.3, linewidth=2, label=name)

    ax2.set_xlabel('Average Episode Depth (n_step)')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) Depth Distribution')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)

    # ========== Subplot 3: Summary Table (Full Width) ==========
    ax3 = fig.add_subplot(2, 2, (3, 4))
    ax3.axis('off')

    # Helper to format hyperparameter values
    def format_hp(value, suffix=''):
        if value is None:
            return 'N/A'
        if isinstance(value, float):
            if value == 0:
                return '0'
            return f'{value:.2e}{suffix}' if value < 0.01 else f'{value:.4f}{suffix}'
        return str(value)

    # Prepare table data
    col_names = ['Metric'] + model_names
    table_data = [col_names]

    # Test Results Section
    table_data.append(['Mean Reward'] + [f"{data_dict[name]['summary'].get('mean_reward', 0):.2f}" for name in model_names])
    table_data.append(['Peak Reward'] + [f"{data_dict[name]['summary'].get('peak_reward', 0):.2f}" for name in model_names])
    table_data.append(['Mean Depth (n_step)'] + [f"{data_dict[name]['summary'].get('mean_depth', 0):.1f}" for name in model_names])
    table_data.append(['Reward Std'] + [f"{data_dict[name]['summary'].get('std_reward', 0):.2f}" for name in model_names])

    # Hyperparameters Section
    table_data.append(['Actor LR'] + [format_hp(data_dict[name]['hyperparameters'].get('actor_lr')) for name in model_names])
    table_data.append(['Critic LR'] + [format_hp(data_dict[name]['hyperparameters'].get('critic_lr')) for name in model_names])
    table_data.append(['Noise Var'] + [format_hp(data_dict[name]['hyperparameters'].get('noise_var')) for name in model_names])
    table_data.append(['Behavior LR (BC)'] + [format_hp(data_dict[name]['hyperparameters'].get('behavior_lr')) for name in model_names])
    table_data.append(['Agent Class'] + [data_dict[name]['hyperparameters'].get('agent_class', 'N/A')[:15] for name in model_names])

    # HAC specific - add hyper_actor_coef if available
    has_hac = any('hyper_actor_coef' in data_dict[name]['hyperparameters'] and
                  data_dict[name]['hyperparameters']['hyper_actor_coef'] is not None
                  for name in model_names)
    if has_hac:
        table_data.append(['Hyper Actor Coef'] + [format_hp(data_dict[name]['hyperparameters'].get('hyper_actor_coef')) for name in model_names])

    # Create table
    table = ax3.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.25] + [0.2] * len(model_names)
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Header row styling
    for j in range(len(col_names)):
        table[(0, j)].set_facecolor('#404040')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Section divider - mark end of test results (row 4)
    for j in range(len(col_names)):
        table[(4, j)].set_edgecolor('black')
        table[(4, j)].set_linewidth(2)

    # Styling for test results section (rows 1-4)
    for i in range(1, 5):
        for j in range(len(col_names)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8E8E8')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')

    # Styling for hyperparameters section
    for i in range(5, len(table_data)):
        for j in range(len(col_names)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F4F8')
            else:
                table[(i, j)].set_facecolor('#F0F8FF')

    # Highlight best values in test results
    mean_rewards = {name: data_dict[name]['summary'].get('mean_reward', 0) for name in model_names}
    best_model = max(mean_rewards, key=mean_rewards.get)
    if best_model in model_names:
        col_idx = model_names.index(best_model) + 1
        table[(1, col_idx)].set_text_props(fontweight='bold', color=colors.get(best_model, '#2E86AB'))

    # ========== Global Title ==========
    fig.suptitle('Evaluation Comparison: HAC vs Baselines on RL4RS',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Dashboard saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse HAC and baseline test logs and generate comparison visualization'
    )
    parser.add_argument('--log1', type=str, required=True,
                        help='Path to first model test log file (e.g., HAC)')
    parser.add_argument('--log2', type=str, required=True,
                        help='Path to second model test log file (e.g., DDPG)')
    parser.add_argument('--name1', type=str, default='HAC',
                        help='Name for first model (default: HAC)')
    parser.add_argument('--name2', type=str, default='DDPG',
                        help='Name for second model (default: DDPG)')
    parser.add_argument('--output', type=str, default='evaluation_comparison.png',
                        help='Output image path (default: evaluation_comparison.png)')

    args = parser.parse_args()

    print(f"Parsing {args.name1} log: {args.log1}")
    data1 = parse_log_file(args.log1)
    print(f"  - Agent: {data1['hyperparameters']['agent_class']}")
    print(f"  - Actor LR: {data1['hyperparameters']['actor_lr']}")
    print(f"  - Mean Reward: {data1['summary'].get('mean_reward', 0):.2f}")
    print(f"  - Mean Depth: {data1['summary'].get('mean_depth', 0):.1f}")

    print(f"\nParsing {args.name2} log: {args.log2}")
    data2 = parse_log_file(args.log2)
    print(f"  - Agent: {data2['hyperparameters']['agent_class']}")
    print(f"  - Actor LR: {data2['hyperparameters']['actor_lr']}")
    print(f"  - Mean Reward: {data2['summary'].get('mean_reward', 0):.2f}")
    print(f"  - Mean Depth: {data2['summary'].get('mean_depth', 0):.1f}")

    print("\nGenerating visualization...")
    data_dict = {
        args.name1: data1,
        args.name2: data2
    }

    plot_comparison_dashboard_with_hyperparams(data_dict, args.output)
    print("\nDone!")


if __name__ == '__main__':
    main()
