"""
Pressure Surrogate Dataset - Loading and Usage Guide

This script demonstrates how to:
1. Load the preprocessed dataset
2. Access train/val/test splits
3. Use normalized vs raw data
4. Denormalize predictions
5. Compute basic statistics

Run this to verify the dataset is ready for model training.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================

def load_dataset(data_dir: Path = None):
    """
    Load the preprocessed pressure surrogate dataset.
    
    Returns:
    --------
    data : dict
        Contains keys: X_train, Y_train, X_val, Y_val, X_test, Y_test,
                      X_train_norm, Y_train_norm, etc.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent
    
    dataset_file = data_dir / 'pressure_surrogate_dataset.npz'
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_file}")
    
    print(f"Loading dataset from: {dataset_file}")
    data = np.load(dataset_file)
    
    # Convert to regular dict (npz format is a zip-like object)
    data_dict = {key: data[key] for key in data.files}
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Available keys: {list(data_dict.keys())}")
    
    return data_dict


def load_normalization_stats(data_dir: Path = None):
    """
    Load normalization parameters for reproducibility.
    
    Returns:
    --------
    stats : dict
        X_mean, X_std, Y_mean, Y_std, feature_names
    """
    if data_dir is None:
        data_dir = Path(__file__).parent
    
    stats_file = data_dir / 'normalization_stats.json'
    
    if not stats_file.exists():
        raise FileNotFoundError(f"Normalization stats not found at {stats_file}")
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print(f"✓ Normalization statistics loaded")
    print(f"  Pressure mean: {stats['Y_mean']:.4f} bar")
    print(f"  Pressure std:  {stats['Y_std']:.4f} bar")
    
    return stats


# ============================================================================
# STEP 2: DATA INSPECTION
# ============================================================================

def inspect_dataset(data: dict):
    """
    Print detailed information about the loaded dataset.
    """
    print("\n" + "="*70)
    print("DATASET INSPECTION")
    print("="*70)
    
    # Training data
    X_train = data['X_train']
    Y_train = data['Y_train']
    
    print(f"\nTraining Data (NORMALIZED):")
    print(f"  X_train shape: {data['X_train_norm'].shape}")
    print(f"    - Samples:       {data['X_train_norm'].shape[0]}")
    print(f"    - Spatial cells: {data['X_train_norm'].shape[1]}")
    print(f"    - Features:      {data['X_train_norm'].shape[2]}")
    print(f"  Y_train shape: {data['Y_train_norm'].shape}")
    print(f"    - Samples:       {data['Y_train_norm'].shape[0]}")
    print(f"    - Spatial cells: {data['Y_train_norm'].shape[1]}")
    
    # Validation data
    print(f"\nValidation Data (NORMALIZED):")
    print(f"  X_val shape: {data['X_val_norm'].shape}")
    print(f"  Y_val shape: {data['Y_val_norm'].shape}")
    
    # Test data
    print(f"\nTest Data (NORMALIZED):")
    print(f"  X_test shape: {data['X_test_norm'].shape}")
    print(f"  Y_test shape: {data['Y_test_norm'].shape}")
    
    # Raw data statistics
    print(f"\nRaw Data Statistics (Before Normalization):")
    print(f"  X_train raw: min={X_train.min():.4f}, max={X_train.max():.4f}")
    print(f"  Y_train raw: min={Y_train.min():.4f}, max={Y_train.max():.4f}")
    
    # Normalized data statistics
    X_train_norm = data['X_train_norm']
    Y_train_norm = data['Y_train_norm']
    
    print(f"\nNormalized Data Statistics:")
    print(f"  X_train_norm: mean≈{X_train_norm.mean():.6f}, std≈{X_train_norm.std():.6f}")
    print(f"  Y_train_norm: mean≈{Y_train_norm.mean():.6f}, std≈{Y_train_norm.std():.6f}")
    
    print(f"\nMemory Usage:")
    print(f"  X_train: {X_train.nbytes / 1e6:.2f} MB")
    print(f"  Y_train: {Y_train.nbytes / 1e6:.2f} MB")
    print(f"  Total (normalized): {(data['X_train_norm'].nbytes + data['Y_train_norm'].nbytes) / 1e6:.2f} MB")


# ============================================================================
# STEP 3: USAGE PATTERNS FOR MODEL TRAINING
# ============================================================================

def example_usage_for_training():
    """
    Example: How to use the dataset for neural network training.
    """
    print("\n" + "="*70)
    print("EXAMPLE: USING DATASET FOR MODEL TRAINING")
    print("="*70)
    
    # Load dataset
    data = load_dataset()
    
    # Extract splits
    X_train = data['X_train_norm']
    Y_train = data['Y_train_norm']
    X_val = data['X_val_norm']
    Y_val = data['Y_val_norm']
    X_test = data['X_test_norm']
    Y_test = data['Y_test_norm']
    
    print("\n1. Using PyTorch:")
    print("""
import torch
from torch.utils.data import TensorDataset, DataLoader

# Convert to PyTorch tensors
X_train_torch = torch.from_numpy(X_train).float()
Y_train_torch = torch.from_numpy(Y_train).float()

# Create dataset and loader
train_dataset = TensorDataset(X_train_torch, Y_train_torch)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
for X_batch, Y_batch in train_loader:
    # Forward pass
    Y_pred = model(X_batch)
    loss = criterion(Y_pred, Y_batch)
    # Backward pass
    loss.backward()
    optimizer.step()
""")
    
    print("\n2. Using TensorFlow/Keras:")
    print("""
import tensorflow as tf

# Create Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(10116, 9)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10116)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=32
)

# Evaluate on test set
test_loss = model.evaluate(X_test, Y_test)
""")
    
    print("\n3. Key points:")
    print("  - Use X_train_norm and Y_train_norm for training")
    print("  - Use X_val_norm and Y_val_norm for validation during training")
    print("  - Use X_test_norm and Y_test_norm for final evaluation ONLY")
    print("  - Never touch test set until final evaluation!")


# ============================================================================
# STEP 4: DENORMALIZATION (Converting Predictions Back to Physical Units)
# ============================================================================

def denormalize_predictions(Y_pred_norm: np.ndarray, stats: dict) -> np.ndarray:
    """
    Convert normalized pressure predictions back to physical units (bar).
    
    Parameters:
    -----------
    Y_pred_norm : np.ndarray
        Normalized predictions shape: [n_samples, 10116]
    stats : dict
        Normalization statistics (from load_normalization_stats)
    
    Returns:
    --------
    Y_pred_physical : np.ndarray
        Denormalized predictions in bar
    """
    Y_mean = stats['Y_mean']
    Y_std = stats['Y_std']
    
    Y_pred_physical = Y_pred_norm * Y_std + Y_mean
    
    return Y_pred_physical


def example_denormalization():
    """
    Example: How to denormalize model predictions.
    """
    print("\n" + "="*70)
    print("EXAMPLE: DENORMALIZING PREDICTIONS")
    print("="*70)
    
    # Load dataset and stats
    data = load_dataset()
    stats = load_normalization_stats()
    
    # Simulated model predictions (random for this example)
    Y_pred_norm = data['Y_test_norm'][:5]  # First 5 test samples, normalized
    Y_test = data['Y_test'][:5]  # Ground truth, raw
    
    # Denormalize
    Y_pred_physical = denormalize_predictions(Y_pred_norm, stats)
    
    print(f"\nExample denormalization (5 samples):")
    print(f"  Normalized predictions: min={Y_pred_norm.min():.4f}, max={Y_pred_norm.max():.4f}")
    print(f"  Physical predictions:   min={Y_pred_physical.min():.2f} bar, max={Y_pred_physical.max():.2f} bar")
    print(f"  Ground truth:           min={Y_test.min():.2f} bar, max={Y_test.max():.2f} bar")
    
    # Compute error
    error = np.abs(Y_pred_physical - Y_test)
    print(f"  Mean absolute error:    {error.mean():.4f} bar")


# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================

def plot_data_distributions(data: dict, stats: dict):
    """
    Plot distributions of features and targets.
    """
    print("\nGenerating diagnostic plots...")
    
    X_train = data['X_train']
    Y_train = data['Y_train']
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle('Data Distributions', fontsize=14, fontweight='bold')
    
    # Feature names
    feature_names = stats['feature_names']
    
    # Plot each feature distribution
    for feat_idx in range(9):
        ax = axes[feat_idx // 3, feat_idx % 3]
        
        # Extract feature across all samples and cells
        X_feat = X_train[:, :, feat_idx].reshape(-1)
        
        ax.hist(X_feat, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel(f'{feature_names[feat_idx]}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Feature {feat_idx}: {feature_names[feat_idx]}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_target_distribution(data: dict):
    """
    Plot pressure target distributions for train/val/test.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Pressure Target Distributions', fontsize=14, fontweight='bold')
    
    splits = [
        ('Train', data['Y_train']),
        ('Validation', data['Y_val']),
        ('Test', data['Y_test'])
    ]
    
    for ax, (split_name, Y) in zip(axes, splits):
        Y_flat = Y.reshape(-1)
        ax.hist(Y_flat, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Pressure (bar)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{split_name} Set')
        ax.grid(True, alpha=0.3)
        
        # Add stats
        stats_text = f"Mean: {Y_flat.mean():.1f}\nStd: {Y_flat.std():.1f}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN: RUN ALL CHECKS
# ============================================================================

def main():
    """
    Run complete dataset verification and provide usage examples.
    """
    print("\n" + "="*70)
    print("PRESSURE SURROGATE DATASET - VERIFICATION AND USAGE")
    print("="*70)
    
    # Load dataset and stats
    data = load_dataset()
    stats = load_normalization_stats()
    
    # Inspect
    inspect_dataset(data)
    
    # Usage examples
    example_usage_for_training()
    example_denormalization()
    
    # Plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    fig1 = plot_data_distributions(data, stats)
    fig1.savefig(Path(__file__).parent / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: feature_distributions.png")
    
    fig2 = plot_target_distribution(data)
    fig2.savefig(Path(__file__).parent / 'target_distributions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: target_distributions.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Dataset Status: ✓ READY FOR MODEL TRAINING

Key Files:
  - pressure_surrogate_dataset.npz   (725 samples, all splits)
  - normalization_stats.json          (statistics for reproducibility)
  - PHASE_1_GUIDE.md                  (educational documentation)

Next Steps:
  1. Design neural network architecture
  2. Load data using this script: data = load_dataset()
  3. Train model on X_train_norm, Y_train_norm
  4. Validate on X_val_norm, Y_val_norm
  5. Test on X_test_norm, Y_test_norm
  6. Denormalize predictions using denormalize_predictions()

Important Reminders:
  - Always use normalized data for training
  - Never modify test set during training
  - Use denormalize_predictions() to interpret results
  - Pressure range should be 50-212 bar after denormalization
""")


if __name__ == "__main__":
    main()
