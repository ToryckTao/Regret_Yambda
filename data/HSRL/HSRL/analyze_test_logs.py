#!/usr/bin/env python3
"""
RL Test Log Analyzer - Academic Publication Quality Visualization
Parses DDPG and HSRL test logs and generates comparison dashboards.
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
    Parse a single RL test log file.
    
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
    # There are multiple Namespace lines - we need the one with agent hyperparameters
    # (appears after "Setup agent" or contains actor_lr, noise_var, etc.)
    hyperparameters = {
        'agent_class': None,
        'actor_lr': None,
        'noise_var': None,
        'entropy_coef': None,
        'behavior_lr': None
    }
    
    # First pass: find agent_class from the first Namespace
    for line in lines:
        if line.startswith('Namespace(') and "agent_class='" in line:
            agent_match = re.search(r"agent_class='(\w+)'", line)
            if agent_match:
                hyperparameters['agent_class'] = agent_match.group(1)
            break
    
    # Second pass: find hyperparameters from the agent setup Namespace
    # This typically appears after "Setup agent" and contains actor_lr, noise_var, etc.
    for line in lines:
        if line.startswith('Namespace(') and 'actor_lr=' in line:
            # Extract actor_lr (supports scientific notation like 3e-05 and decimal like 0.0001)
            actor_lr_match = re.search(r'actor_lr=([0-9.e\-]+)', line)
            if actor_lr_match:
                try:
                    hyperparameters['actor_lr'] = float(actor_lr_match.group(1))
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
            
            break
    
    # Extract evaluation metrics
    step_data = []
    pattern = r"step:\s*(\d+)\s*@\s*episode report:\s*\{([^}]+)\}"
    
    for match in re.finditer(pattern, content):
        step = int(match.group(1))
        metrics_str = match.group(2)
        
        # Extract metrics using regex
        avg_reward_match = re.search(r"'average_total_reward':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        max_reward_match = re.search(r"'max_total_reward':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        avg_n_step_match = re.search(r"'average_n_step':\s*np\.float(?:32|64)\(([0-9e.\-]+)\)", metrics_str)
        
        if avg_reward_match and avg_n_step_match:
            avg_reward = float(avg_reward_match.group(1))
            max_reward = float(max_reward_match.group(1)) if max_reward_match else np.nan
            avg_n_step = float(avg_n_step_match.group(1))
            
            if avg_reward > 0:  # Filter out initial step with 0 rewards
                step_data.append({
                    'step': step,
                    'average_total_reward': avg_reward,
                    'max_total_reward': max_reward,
                    'average_n_step': avg_n_step
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
    
    return {
        'hyperparameters': hyperparameters,
        'metrics': df,
        'summary': summary
    }


def plot_comparison_dashboard(data_dict: Dict[str, Dict], output_path: str = 'comparison_dashboard.png'):
    """
    Generate a 1x3 academic-style comparison dashboard.
    
    Args:
        data_dict: Dictionary with keys 'DDPG' and 'HSRL', each containing
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
        'DDPG': '#2E86AB',    # Steel blue
        'HSRL': '#E94F37'     # Vermilion red
    }
    
    fig = plt.figure(figsize=(14, 5))
    
    # ========== Subplot 1: Total Reward Distribution (KDE) ==========
    ax1 = fig.add_subplot(1, 3, 1)
    
    ddpg_rewards = data_dict['DDPG']['metrics']['average_total_reward'].values
    hsrl_rewards = data_dict['HSRL']['metrics']['average_total_reward'].values
    
    # KDE plots with fill
    sns.kdeplot(ddpg_rewards, ax=ax1, color=colors['DDPG'], 
                fill=True, alpha=0.4, linewidth=2, label='DDPG')
    sns.kdeplot(hsrl_rewards, ax=ax1, color=colors['HSRL'], 
                fill=True, alpha=0.4, linewidth=2, label='HSRL')
    
    ax1.set_xlabel('Average Total Reward')
    ax1.set_ylabel('Density')
    ax1.set_title('Total Reward Distribution')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    
    # ========== Subplot 2: Depth Distribution (KDE) ==========
    ax2 = fig.add_subplot(1, 3, 2)
    
    ddpg_depths = data_dict['DDPG']['metrics']['average_n_step'].values
    hsrl_depths = data_dict['HSRL']['metrics']['average_n_step'].values
    
    sns.kdeplot(ddpg_depths, ax=ax2, color=colors['DDPG'], 
                fill=True, alpha=0.4, linewidth=2, label='DDPG')
    sns.kdeplot(hsrl_depths, ax=ax2, color=colors['HSRL'], 
                fill=True, alpha=0.4, linewidth=2, label='HSRL')
    
    ax2.set_xlabel('Average Episode Depth (n_step)')
    ax2.set_ylabel('Density')
    ax2.set_title('Depth Distribution')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)
    
    # ========== Subplot 3: Summary Table ==========
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis('off')
    
    ddpg_hp = data_dict['DDPG']['hyperparameters']
    hsrl_hp = data_dict['HSRL']['hyperparameters']
    ddpg_sum = data_dict['DDPG']['summary']
    hsrl_sum = data_dict['HSRL']['summary']
    
    # Prepare table data
    # Upper part: Test Results
    table_data = [
        ['Metric', 'DDPG', 'HSRL'],
        ['Mean Reward', f"{ddpg_sum.get('mean_reward', 0):.2f}", f"{hsrl_sum.get('mean_reward', 0):.2f}"],
        ['Peak Reward', f"{ddpg_sum.get('peak_reward', 0):.2f}", f"{hsrl_sum.get('peak_reward', 0):.2f}"],
        ['Mean Depth', f"{ddpg_sum.get('mean_depth', 0):.1f}", f"{hsrl_sum.get('mean_depth', 0):.1f}"],
    ]
    
    # Create table
    table = ax3.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.35, 0.3, 0.3]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Header row styling
    for j in range(3):
        table[(0, j)].set_facecolor('#404040')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternating row colors
    for i in range(1, 4):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    # Highlight best values
    if ddpg_sum.get('mean_reward', 0) > hsrl_sum.get('mean_reward', 0):
        table[(1, 1)].set_text_props(fontweight='bold', color=colors['DDPG'])
    else:
        table[(1, 2)].set_text_props(fontweight='bold', color=colors['HSRL'])
    
    if hsrl_sum.get('peak_reward', 0) > ddpg_sum.get('peak_reward', 0):
        table[(2, 2)].set_text_props(fontweight='bold', color=colors['HSRL'])
    else:
        table[(2, 1)].set_text_props(fontweight='bold', color=colors['DDPG'])
    
    if hsrl_sum.get('mean_depth', 0) > ddpg_sum.get('mean_depth', 0):
        table[(3, 2)].set_text_props(fontweight='bold', color=colors['HSRL'])
    else:
        table[(3, 1)].set_text_props(fontweight='bold', color=colors['DDPG'])
    
    # Add title for test results section
    ax3.set_title('Test Results', pad=20, fontweight='bold', fontsize=12)
    
    # ========== Global Title ==========
    fig.suptitle('Evaluation Comparison: HSRL vs DDPG on RL4RS', 
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
        data_dict: Dictionary with keys 'DDPG' and 'HSRL', each containing
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
        'DDPG': '#2E86AB',    # Steel blue
        'HSRL': '#E94F37'     # Vermilion red
    }
    
    fig = plt.figure(figsize=(14, 6))
    
    # ========== Subplot 1: Total Reward Distribution (KDE) ==========
    ax1 = fig.add_subplot(2, 2, 1)
    
    ddpg_rewards = data_dict['DDPG']['metrics']['average_total_reward'].values
    hsrl_rewards = data_dict['HSRL']['metrics']['average_total_reward'].values
    
    # KDE plots with fill
    sns.kdeplot(ddpg_rewards, ax=ax1, color=colors['DDPG'], 
                fill=True, alpha=0.4, linewidth=2, label='DDPG')
    sns.kdeplot(hsrl_rewards, ax=ax1, color=colors['HSRL'], 
                fill=True, alpha=0.4, linewidth=2, label='HSRL')
    
    ax1.set_xlabel('Average Total Reward')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) Total Reward Distribution')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    
    # ========== Subplot 2: Depth Distribution (KDE) ==========
    ax2 = fig.add_subplot(2, 2, 2)
    
    ddpg_depths = data_dict['DDPG']['metrics']['average_n_step'].values
    hsrl_depths = data_dict['HSRL']['metrics']['average_n_step'].values
    
    sns.kdeplot(ddpg_depths, ax=ax2, color=colors['DDPG'], 
                fill=True, alpha=0.4, linewidth=2, label='DDPG')
    sns.kdeplot(hsrl_depths, ax=ax2, color=colors['HSRL'], 
                fill=True, alpha=0.4, linewidth=2, label='HSRL')
    
    ax2.set_xlabel('Average Episode Depth (n_step)')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) Depth Distribution')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)
    
    # ========== Subplot 3: Summary Table (Full Width) ==========
    ax3 = fig.add_subplot(2, 2, (3, 4))
    ax3.axis('off')
    
    ddpg_hp = data_dict['DDPG']['hyperparameters']
    hsrl_hp = data_dict['HSRL']['hyperparameters']
    ddpg_sum = data_dict['DDPG']['summary']
    hsrl_sum = data_dict['HSRL']['summary']
    
    # Helper to format hyperparameter values
    def format_hp(value, suffix=''):
        if value is None:
            return 'N/A'
        if isinstance(value, float):
            if value == 0:
                return '0'
            return f'{value:.2e}{suffix}' if value < 0.01 else f'{value:.4f}{suffix}'
        return str(value)
    
    # Prepare table data - 2 sections
    # Section 1: Test Results (4 rows)
    # Section 2: Hyperparameters (5 rows)
    table_data = [
        # Test Results Section
        ['', 'DDPG', 'HSRL'],
        ['Mean Reward', f"{ddpg_sum.get('mean_reward', 0):.2f}", f"{hsrl_sum.get('mean_reward', 0):.2f}"],
        ['Peak Reward', f"{ddpg_sum.get('peak_reward', 0):.2f}", f"{hsrl_sum.get('peak_reward', 0):.2f}"],
        ['Mean Depth (n_step)', f"{ddpg_sum.get('mean_depth', 0):.1f}", f"{hsrl_sum.get('mean_depth', 0):.1f}"],
        ['Reward Std', f"{ddpg_sum.get('std_reward', 0):.2f}", f"{hsrl_sum.get('std_reward', 0):.2f}"],
        # Hyperparameters Section
        ['Actor LR', format_hp(ddpg_hp.get('actor_lr')), format_hp(hsrl_hp.get('actor_lr'))],
        ['Noise Var', format_hp(ddpg_hp.get('noise_var')), format_hp(hsrl_hp.get('noise_var'))],
        ['Entropy Coef', format_hp(ddpg_hp.get('entropy_coef')), format_hp(hsrl_hp.get('entropy_coef'))],
        ['Behavior LR (BC)', format_hp(ddpg_hp.get('behavior_lr')), format_hp(hsrl_hp.get('behavior_lr'))],
        ['Agent Class', ddpg_hp.get('agent_class', 'N/A')[:15], hsrl_hp.get('agent_class', 'N/A')[:15]],
    ]
    
    # Create table
    table = ax3.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.2, 0.2]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    
    # Header row styling
    for j in range(3):
        table[(0, j)].set_facecolor('#404040')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Section divider - mark end of test results (row 4)
    for j in range(3):
        table[(4, j)].set_edgecolor('black')
        table[(4, j)].set_linewidth(2)
    
    # Styling for test results section (rows 1-4, index 0-4)
    for i in range(1, 5):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8E8E8')
            else:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    # Styling for hyperparameters section (rows 5-9, index 5-9)
    for i in range(5, 10):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F4F8')
            else:
                table[(i, j)].set_facecolor('#F0F8FF')
    
    # Highlight best values in test results
    if ddpg_sum.get('mean_reward', 0) > hsrl_sum.get('mean_reward', 0):
        table[(1, 1)].set_text_props(fontweight='bold', color=colors['DDPG'])
    else:
        table[(1, 2)].set_text_props(fontweight='bold', color=colors['HSRL'])
    
    if hsrl_sum.get('peak_reward', 0) > ddpg_sum.get('peak_reward', 0):
        table[(2, 2)].set_text_props(fontweight='bold', color=colors['HSRL'])
    else:
        table[(2, 1)].set_text_props(fontweight='bold', color=colors['DDPG'])
    
    if hsrl_sum.get('mean_depth', 0) > ddpg_sum.get('mean_depth', 0):
        table[(3, 2)].set_text_props(fontweight='bold', color=colors['HSRL'])
    else:
        table[(3, 1)].set_text_props(fontweight='bold', color=colors['DDPG'])
    
    # ========== Global Title ==========
    fig.suptitle('Evaluation Comparison: HSRL vs DDPG on RL4RS', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Dashboard saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse RL test logs and generate comparison visualization'
    )
    parser.add_argument('--ddpg_log', type=str, required=True,
                        help='Path to DDPG test log file')
    parser.add_argument('--hsrl_log', type=str, required=True,
                        help='Path to HSRL test log file')
    parser.add_argument('--output', type=str, default='evaluation_comparison.png',
                        help='Output image path (default: evaluation_comparison.png)')
    
    args = parser.parse_args()
    
    print("Parsing DDPG log...")
    ddpg_data = parse_log_file(args.ddpg_log)
    print(f"  - Agent: {ddpg_data['hyperparameters']['agent_class']}")
    print(f"  - Actor LR: {ddpg_data['hyperparameters']['actor_lr']}")
    print(f"  - Mean Reward: {ddpg_data['summary'].get('mean_reward', 0):.2f}")
    print(f"  - Mean Depth: {ddpg_data['summary'].get('mean_depth', 0):.1f}")
    
    print("\nParsing HSRL log...")
    hsrl_data = parse_log_file(args.hsrl_log)
    print(f"  - Agent: {hsrl_data['hyperparameters']['agent_class']}")
    print(f"  - Actor LR: {hsrl_data['hyperparameters']['actor_lr']}")
    print(f"  - Entropy Coef: {hsrl_data['hyperparameters']['entropy_coef']}")
    print(f"  - Mean Reward: {hsrl_data['summary'].get('mean_reward', 0):.2f}")
    print(f"  - Mean Depth: {hsrl_data['summary'].get('mean_depth', 0):.1f}")
    
    print("\nGenerating visualization...")
    data_dict = {
        'DDPG': ddpg_data,
        'HSRL': hsrl_data
    }
    
    plot_comparison_dashboard_with_hyperparams(data_dict, args.output)
    print("\nDone!")


if __name__ == '__main__':
    main()
