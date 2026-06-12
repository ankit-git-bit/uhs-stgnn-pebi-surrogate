"""
Generate diagnostic plots for the pressure surrogate dataset.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    data_dir = Path(__file__).parent
    
    # Load dataset
    dataset_file = data_dir / 'pressure_surrogate_dataset.npz'
    data = np.load(dataset_file)
    
    # Load stats
    with open(data_dir / 'normalization_stats.json', 'r') as f:
        stats = json.load(f)
    
    print("Generating diagnostic plots...")
    
    # Plot 1: Feature distributions
    X_train = data['X_train']
    feature_names = stats['feature_names']
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle('Input Feature Distributions (Training Set)', fontsize=14, fontweight='bold')
    
    for feat_idx in range(9):
        ax = axes[feat_idx // 3, feat_idx % 3]
        X_feat = X_train[:, :, feat_idx].reshape(-1)
        ax.hist(X_feat, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(f'{feature_names[feat_idx]}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Feature {feat_idx+1}: {feature_names[feat_idx]}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(str(data_dir / 'feature_distributions.png'), dpi=150, bbox_inches='tight')
    print("✓ Saved: feature_distributions.png")
    plt.close()
    
    # Plot 2: Target pressure distributions
    Y_train = data['Y_train']
    Y_val = data['Y_val']
    Y_test = data['Y_test']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Pressure Target Distributions (Raw Data)', fontsize=14, fontweight='bold')
    
    splits = [
        ('Training Set', Y_train, 'steelblue'),
        ('Validation Set', Y_val, 'darkorange'),
        ('Test Set', Y_test, 'green')
    ]
    
    for ax, (split_name, Y, color) in zip(axes, splits):
        Y_flat = Y.reshape(-1)
        ax.hist(Y_flat, bins=100, edgecolor='black', alpha=0.7, color=color)
        ax.set_xlabel('Pressure (bar)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{split_name} (n={Y.shape[0]})')
        ax.grid(True, alpha=0.3)
        
        stats_text = f"μ: {Y_flat.mean():.1f}\nσ: {Y_flat.std():.1f}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    fig.savefig(str(data_dir / 'target_distributions.png'), dpi=150, bbox_inches='tight')
    print("✓ Saved: target_distributions.png")
    plt.close()
    
    # Plot 3: Data split visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Dataset Overview', fontsize=14, fontweight='bold')
    
    # Pie chart: split sizes
    ax = axes[0, 0]
    sizes = [Y_train.shape[0], Y_val.shape[0], Y_test.shape[0]]
    labels = [f'Training\n{sizes[0]}', f'Validation\n{sizes[1]}', f'Test\n{sizes[2]}']
    colors = ['steelblue', 'darkorange', 'green']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Train/Val/Test Split')
    
    # Bar plot: sample counts
    ax = axes[0, 1]
    split_names = ['Training', 'Validation', 'Test']
    sample_counts = sizes
    bars = ax.bar(split_names, sample_counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Counts per Split')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, sample_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom')
    
    # Box plot: pressure by split
    ax = axes[1, 0]
    pressure_data = [Y_train.reshape(-1), Y_val.reshape(-1), Y_test.reshape(-1)]
    bp = ax.boxplot(pressure_data, labels=split_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Pressure (bar)')
    ax.set_title('Pressure Distribution by Split')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_data = []
    for i, (name, Y_split) in enumerate([('Training', Y_train), ('Validation', Y_val), ('Test', Y_test)]):
        Y_flat = Y_split.reshape(-1)
        stats_data.append([
            name,
            f"{Y_split.shape[0]}",
            f"{Y_flat.min():.2f}",
            f"{Y_flat.max():.2f}",
            f"{Y_flat.mean():.2f}",
            f"{Y_flat.std():.2f}"
        ])
    
    table = ax.table(cellText=stats_data,
                     colLabels=['Split', 'Samples', 'Min (bar)', 'Max (bar)', 'Mean (bar)', 'Std (bar)'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(6):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Pressure Statistics by Split', pad=20, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(str(data_dir / 'dataset_overview.png'), dpi=150, bbox_inches='tight')
    print("✓ Saved: dataset_overview.png")
    plt.close()
    
    print("\n✓ All diagnostic plots generated successfully!")


if __name__ == "__main__":
    main()
