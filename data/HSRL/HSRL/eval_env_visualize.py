#!/usr/bin/env python3
"""
RL4RS User Response Model Environment Evaluation Visualization
================================================================
This script provides comprehensive visualization for evaluating the trained
user response model used in the HSRL reinforcement learning environment.

Author: HSRL Team
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

import argparse
import os
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'output_dir': 'output/rl4rs/env',
    'log_file': 'output/rl4rs/env/log/rl4rs_user_env_lr0.0003_reg0.0001.model.log',
    'pred_file': 'output/rl4rs/env/rl4rs_user_env.model.output',
    'sample_file': 'output/rl4rs/env/rl4rs_user_env.model.ber',
    'save_dir': 'output/rl4rs/env/figures',
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def parse_log_file(log_path):
    """Parse training log file to extract loss and AUC metrics."""
    losses = []
    aucs = []
    epochs = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Find validation AUC lines
    for i, line in enumerate(lines):
        if 'validating; auc:' in line:
            try:
                auc = float(line.split('auc:')[1].strip())
                aucs.append(auc)
                # Find corresponding epoch
                for j in range(i-1, -1, -1):
                    if 'epoch' in lines[j] and 'training' in lines[j]:
                        epoch = int(lines[j].split('epoch')[1].split('training')[0].strip())
                        epochs.append(epoch)
                        break
            except (IndexError, ValueError):
                continue
        elif 'Iteration' in line and 'loss:' in line:
            try:
                loss = float(line.split('loss:')[1].strip())
                losses.append(loss)
            except (IndexError, ValueError):
                continue
    
    return losses, aucs, epochs


def load_predictions(pred_path, max_samples=None):
    """Load model predictions from output file."""
    predictions = []
    with open(pred_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                predictions.append(float(line.strip()))
            except ValueError:
                continue
    return np.array(predictions)


def load_behavior_samples(sample_path, max_samples=None):
    """Load behavior sampling results (number of clicks per sample)."""
    samples = []
    with open(sample_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                samples.append(float(line.strip()))
            except ValueError:
                continue
    return np.array(samples)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_curve(losses, aucs, epochs, save_path):
    """
    Plot training loss curve and validation AUC.
    
    This figure shows:
    - Left: Training loss over iterations
    - Right: Validation AUC per epoch
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    ax1 = axes[0]
    iterations = np.arange(1, len(losses) + 1)
    ax1.plot(iterations, losses, 'b-', alpha=0.6, linewidth=0.8, label='Raw Loss')
    
    # Moving average for smoother curve
    window = min(100, len(losses) // 10)
    if window > 1:
        losses_ma = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax1.plot(np.arange(window//2, len(losses)-window//2+1), losses_ma, 
                'r-', linewidth=2, label=f'Moving Avg (window={window})')
    
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, len(losses)])
    
    # Plot 2: Validation AUC
    ax2 = axes[1]
    ax2.bar(epochs, aucs, color='steelblue', alpha=0.7, edgecolor='navy')
    ax2.plot(epochs, aucs, 'ro-', markersize=8, linewidth=2)
    
    for i, (ep, auc) in enumerate(zip(epochs, aucs)):
        ax2.annotate(f'{auc:.4f}', (ep, auc), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation AUC')
    ax2.set_title('Validation AUC per Epoch')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([min(aucs)*0.95, max(aucs)*1.02])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_prediction_distribution(predictions, save_path):
    """
    Plot prediction probability distribution with comprehensive statistics.
    
    This figure shows:
    - Histogram of predicted probabilities
    - Cumulative distribution
    - Key statistics annotation
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Histogram
    ax1 = axes[0]
    bins = np.linspace(0, 1, 51)
    n, bins_edges, patches = ax1.hist(predictions, bins=bins, color='steelblue', 
                                        alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Color gradient based on probability
    for i, patch in enumerate(patches):
        bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
        if bin_center < 0.3:
            patch.set_facecolor('#2ecc71')  # Green for low
        elif bin_center < 0.7:
            patch.set_facecolor('#f39c12')  # Orange for medium
        else:
            patch.set_facecolor('#e74c3c')  # Red for high
    
    ax1.axvline(np.mean(predictions), color='navy', linestyle='--', linewidth=2, 
                label=f'Mean = {np.mean(predictions):.3f}')
    ax1.axvline(np.median(predictions), color='purple', linestyle=':', linewidth=2,
                label=f'Median = {np.median(predictions):.3f}')
    
    ax1.set_xlabel('Predicted Click Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Predicted Probabilities')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Cumulative Distribution
    ax2 = axes[1]
    sorted_preds = np.sort(predictions)
    cdf = np.arange(1, len(sorted_preds) + 1) / len(sorted_preds)
    ax2.plot(sorted_preds, cdf, 'b-', linewidth=2, label='CDF')
    
    # Mark key percentiles
    percentiles = [25, 50, 75, 90]
    colors = ['green', 'purple', 'orange', 'red']
    for p, c in zip(percentiles, colors):
        val = np.percentile(predictions, p)
        ax2.axhline(p/100, color=c, linestyle='--', alpha=0.5)
        ax2.axvline(val, color=c, linestyle='--', alpha=0.5)
        ax2.plot(val, p/100, 'o', color=c, markersize=8)
        ax2.annotate(f'P{p}={val:.2f}', (val, p/100), textcoords="offset points",
                    xytext=(5, 5), fontsize=9, color=c)
    
    ax2.set_xlabel('Predicted Click Probability')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution Function')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_behavior_analysis(predictions, behavior_samples, save_path):
    """
    Analyze the relationship between predictions and actual behavior sampling.
    
    This figure shows:
    - Behavior distribution (clicks per sample)
    - Prediction vs Behavior comparison
    """
    # Align arrays to same length
    min_len = min(len(predictions), len(behavior_samples))
    predictions = predictions[:min_len]
    behavior_samples = behavior_samples[:min_len]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Behavior Distribution
    ax1 = axes[0]
    unique, counts = np.unique(behavior_samples, return_counts=True)
    ax1.bar(unique, counts, color='teal', alpha=0.7, edgecolor='darkslategray')
    ax1.set_xlabel('Number of Clicks per Sample')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Sampled Clicks')
    
    # Add statistics
    stats_text = f'Mean: {np.mean(behavior_samples):.2f}\nStd: {np.std(behavior_samples):.2f}\nMax: {int(np.max(behavior_samples))}'
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Prediction vs Behavior
    ax2 = axes[1]
    
    # Bin predictions and compute average behavior
    n_bins = 20
    pred_bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (pred_bins[:-1] + pred_bins[1:]) / 2
    avg_behavior = []
    
    for i in range(n_bins):
        mask = (predictions >= pred_bins[i]) & (predictions < pred_bins[i+1])
        if np.sum(mask) > 0:
            avg_behavior.append(np.mean(behavior_samples[mask]))
        else:
            avg_behavior.append(0)
    
    ax2.bar(bin_centers - 0.02, bin_centers, width=0.04, alpha=0.5, 
            color='blue', label='Ideal (prediction = behavior)')
    ax2.bar(bin_centers + 0.02, avg_behavior, width=0.04, alpha=0.7,
            color='red', label='Actual Average')
    
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Average Sampled Clicks')
    ax2.set_title('Prediction Calibration Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_summary(predictions, behavior_samples, aucs, save_path):
    """
    Create a comprehensive summary dashboard.
    
    This figure provides a quick overview of all key metrics.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Title and Info
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    title_text = """
    RL4RS User Response Model Evaluation Report
    ──────────────────────────────────────────────────
    Environment: rl4rs_user_env_lr0.0003_reg0.0001
    Dataset: RL4RS (Retail Recommendation)
    """
    ax_title.text(0.5, 0.6, title_text, transform=ax_title.transAxes,
                 fontsize=14, ha='center', va='center', family='monospace')
    
    # 2. Key Metrics Table
    ax_table = fig.add_subplot(gs[1, 0])
    ax_table.axis('off')
    
    metrics = [
        ['Metric', 'Value'],
        ['─' * 15, '─' * 15],
        ['Total Samples', f'{len(predictions):,}'],
        ['Mean Prediction', f'{np.mean(predictions):.4f}'],
        ['Std Prediction', f'{np.std(predictions):.4f}'],
        ['Median Prediction', f'{np.median(predictions):.4f}'],
        ['Best Val AUC', f'{max(aucs):.4f}'],
        ['Mean Clicks', f'{np.mean(behavior_samples):.2f}'],
    ]
    
    table_text = '\n'.join([f'{m:<16} {v:>15}' for m, v in metrics])
    ax_table.text(0, 1, table_text, transform=ax_table.transAxes,
                 fontsize=11, va='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # 3. Prediction Distribution
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_hist.hist(predictions, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax_hist.axvline(np.mean(predictions), color='red', linestyle='--', linewidth=2)
    ax_hist.set_xlabel('Probability')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Prediction Distribution')
    ax_hist.grid(True, alpha=0.3)
    
    # 4. Behavior Distribution  
    ax_beh = fig.add_subplot(gs[1, 2])
    unique, counts = np.unique(behavior_samples, return_counts=True)
    ax_beh.bar(unique, counts, color='teal', alpha=0.7)
    ax_beh.set_xlabel('Clicks')
    ax_beh.set_ylabel('Count')
    ax_beh.set_title('Behavior Distribution')
    ax_beh.grid(True, alpha=0.3, axis='y')
    
    # 5. Percentile Analysis
    ax_pct = fig.add_subplot(gs[2, 0])
    percentiles = [10, 25, 50, 75, 90]
    values = [np.percentile(predictions, p) for p in percentiles]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    ax_pct.barh([str(p) for p in percentiles], values, color=colors, alpha=0.7)
    ax_pct.set_xlabel('Probability')
    ax_pct.set_ylabel('Percentile')
    ax_pct.set_title('Prediction Percentiles')
    ax_pct.grid(True, alpha=0.3, axis='x')
    
    # 6. AUC Progression
    ax_auc = fig.add_subplot(gs[2, 1:])
    ax_auc.plot(aucs, 'bo-', markersize=10, linewidth=2)
    for i, auc in enumerate(aucs):
        ax_auc.annotate(f'Epoch {i+1}\n{auc:.4f}', (i, auc), 
                       textcoords="offset points", xytext=(0, 10), ha='center')
    ax_auc.set_xlabel('Epoch')
    ax_auc.set_ylabel('Validation AUC')
    ax_auc.set_title('Validation AUC Progression')
    ax_auc.grid(True, alpha=0.3)
    ax_auc.set_ylim([min(aucs)*0.98, max(aucs)*1.01])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main evaluation and visualization pipeline."""
    parser = argparse.ArgumentParser(description='RL4RS Environment Evaluation')
    parser.add_argument('--log', type=str, default=CONFIG['log_file'],
                       help='Path to training log file')
    parser.add_argument('--pred', type=str, default=CONFIG['pred_file'],
                       help='Path to prediction output file')
    parser.add_argument('--sample', type=str, default=CONFIG['sample_file'],
                       help='Path to behavior sample file')
    parser.add_argument('--output', type=str, default=CONFIG['save_dir'],
                       help='Output directory for figures')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to load for visualization')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("RL4RS User Response Model Evaluation")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading training log...")
    losses, aucs, epochs = parse_log_file(args.log)
    print(f"  - Loaded {len(losses)} training iterations")
    print(f"  - Loaded {len(aucs)} validation epochs")
    
    print("\n[2/4] Loading predictions...")
    predictions = load_predictions(args.pred, args.max_samples)
    print(f"  - Loaded {len(predictions):,} predictions")
    
    print("\n[3/4] Loading behavior samples...")
    behavior_samples = load_behavior_samples(args.sample, args.max_samples)
    print(f"  - Loaded {len(behavior_samples):,} samples")
    print(f"  - Mean clicks: {np.mean(behavior_samples):.4f}")
    
    # Generate visualizations
    print("\n[4/4] Generating visualizations...")
    
    # 1. Training curves
    plot_training_curve(
        losses, aucs, epochs,
        os.path.join(args.output, '01_training_curves.png')
    )
    
    # 2. Prediction distribution
    plot_prediction_distribution(
        predictions,
        os.path.join(args.output, '02_prediction_distribution.png')
    )
    
    # 3. Behavior analysis
    plot_behavior_analysis(
        predictions, behavior_samples,
        os.path.join(args.output, '03_behavior_analysis.png')
    )
    
    # 4. Summary dashboard
    plot_model_summary(
        predictions, behavior_samples, aucs,
        os.path.join(args.output, '04_summary_dashboard.png')
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Figures saved to: {args.output}")
    print("=" * 60)
    
    # Print summary statistics
    print("\n" + "─" * 40)
    print("SUMMARY STATISTICS")
    print("─" * 40)
    print(f"  Best Validation AUC: {max(aucs):.4f}")
    print(f"  Final Validation AUC: {aucs[-1]:.4f}")
    print(f"  Final Training Loss: {losses[-1]:.4f}")
    print(f"  Prediction Mean: {np.mean(predictions):.4f}")
    print(f"  Prediction Median: {np.median(predictions):.4f}")
    print(f"  Prediction Std: {np.std(predictions):.4f}")
    print(f"  Mean Sampled Clicks: {np.mean(behavior_samples):.4f}")
    print("─" * 40)


if __name__ == '__main__':
    main()
