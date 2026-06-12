"""
Phase 3: Pressure Change Prediction (ΔP = P(t+1) - P(t))

Reformulates the problem from direct pressure prediction to pressure-change prediction.
This forces the model to learn the physical mechanisms driving pressure evolution
rather than simply reproducing the previous state.

Models:
  A. Persistence: ΔP = 0
  B. Linear regression
  C. Small MLP (pointwise)

Analysis includes:
  - ΔP statistics and distributions
  - Phase-dependent performance (injection, production, shut-in)
  - Feature ablation for linear regression
  - Comparison with Phase 2 direct prediction
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    'Perm_initial',
    'Poro_initial',
    'P(t)',
    'inj_BHP',
    'prod_BHP',
    'inj_H2',
    'prod_H2',
    'inj_status',
    'prod_status',
]

PLOT_DIR = Path('phase3_plots')


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_sample_info: Dict


def load_case(case_path: Path) -> Dict[str, np.ndarray]:
    """Load a single .mat case file."""
    data_raw = loadmat(str(case_path))
    data = {k: v for k, v in data_raw.items() if not k.startswith('__')}

    case_data = {
        'Perm_initial': np.squeeze(data['Perm_initial']).astype(np.float32),
        'Poro_initial': np.squeeze(data['Poro_initial']).astype(np.float32),
        'P_matrix': np.array(data['P_matrix'], dtype=np.float32),
        'inj_BHP': np.squeeze(data['inj_BHP']).astype(np.float32),
        'prod_BHP': np.squeeze(data['prod_BHP']).astype(np.float32),
        'inj_H2': np.squeeze(data['inj_H2']).astype(np.float32),
        'prod_H2': np.squeeze(data['prod_H2']).astype(np.float32),
        'inj_status': np.squeeze(data['inj_status']).astype(np.float32),
        'prod_status': np.squeeze(data['prod_status']).astype(np.float32),
    }

    P = case_data['P_matrix']
    if P.ndim != 2:
        P = np.squeeze(P)
    if P.ndim != 2:
        raise ValueError(f'Unexpected P_matrix shape: {P.shape}')
    case_data['P_matrix'] = P

    return case_data


def build_case_samples(case_data: Dict[str, np.ndarray], case_id: int) -> Dict[str, np.ndarray]:
    """Build (X, ΔP) pairs from a case with ΔP = P(t+1) - P(t)."""
    perm = case_data['Perm_initial']
    poro = case_data['Poro_initial']
    P = case_data['P_matrix']
    inj_BHP = case_data['inj_BHP']
    prod_BHP = case_data['prod_BHP']
    inj_H2 = case_data['inj_H2']
    prod_H2 = case_data['prod_H2']
    inj_status = case_data['inj_status']
    prod_status = case_data['prod_status']

    n_cells, n_steps = P.shape
    n_pairs = n_steps - 1

    X = np.empty((n_pairs, n_cells, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.empty((n_pairs, n_cells), dtype=np.float32)
    
    sample_info = {
        'case_id': np.full(n_pairs, case_id, dtype=np.int32),
        'timestep': np.arange(n_pairs, dtype=np.int32),
        'inj_status': inj_status[:-1].astype(np.int32),
        'prod_status': prod_status[:-1].astype(np.int32),
    }

    for t in range(n_pairs):
        X[t, :, 0] = perm
        X[t, :, 1] = poro
        X[t, :, 2] = P[:, t]
        X[t, :, 3] = inj_BHP[t]
        X[t, :, 4] = prod_BHP[t]
        X[t, :, 5] = inj_H2[t]
        X[t, :, 6] = prod_H2[t]
        X[t, :, 7] = inj_status[t]
        X[t, :, 8] = prod_status[t]
        # TARGET: ΔP = P(t+1) - P(t)
        y[t, :] = P[:, t + 1] - P[:, t]

    return {
        'X': X,
        'y': y,
        'sample_info': sample_info,
    }


def build_dataset(data_dir: Path) -> DatasetSplit:
    """Build train/val/test splits for ΔP prediction."""
    case_files = sorted(data_dir.glob('Case_*.mat'))
    if len(case_files) != 5:
        raise FileNotFoundError('Expected 5 Case_*.mat files in the workspace root.')

    all_train_X = []
    all_train_y = []
    all_val_X = []
    all_val_y = []
    all_test_X = []
    all_test_y = []
    test_sample_info = {
        'case_id': [],
        'timestep': [],
        'inj_status': [],
        'prod_status': [],
    }

    for case_id, case_path in enumerate(case_files):
        print(f"Loading {case_path.name}...")
        case_data = load_case(case_path)
        case_samples = build_case_samples(case_data, case_id)
        
        if case_id < 4:
            X_trainval, X_val, y_trainval, y_val = train_test_split(
                case_samples['X'],
                case_samples['y'],
                test_size=0.10,
                random_state=0,
                shuffle=True,
            )
            all_train_X.append(X_trainval)
            all_train_y.append(y_trainval)
            all_val_X.append(X_val)
            all_val_y.append(y_val)
        else:
            all_test_X.append(case_samples['X'])
            all_test_y.append(case_samples['y'])
            for key in test_sample_info:
                test_sample_info[key].append(case_samples['sample_info'][key])

    X_train = np.concatenate(all_train_X, axis=0)
    y_train = np.concatenate(all_train_y, axis=0)
    X_val = np.concatenate(all_val_X, axis=0)
    y_val = np.concatenate(all_val_y, axis=0)
    X_test = np.concatenate(all_test_X, axis=0)
    y_test = np.concatenate(all_test_y, axis=0)
    test_sample_info = {k: np.concatenate(v, axis=0) for k, v in test_sample_info.items()}

    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        test_sample_info=test_sample_info,
    )


def flatten_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten from (n_pairs, n_cells, n_features) to (n_total, n_features)."""
    return X.reshape(-1, X.shape[-1]), y.reshape(-1)


def expand_sample_info(sample_info: Dict, n_cells: int) -> Dict:
    """Expand sample info from (n_pairs,) to (n_pairs * n_cells,)."""
    expanded = {}
    for key, val in sample_info.items():
        # Repeat each value n_cells times
        expanded[key] = np.repeat(val, n_cells)
    return expanded


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rel = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))))
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'relative_error': rel,
    }


def persistence_baseline(X: np.ndarray) -> np.ndarray:
    """Persistence: ΔP = 0 (no change)."""
    return np.zeros(X.shape[0])


def train_linear_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train linear regression."""
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_mlp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_size: int = 300_000,
) -> MLPRegressor:
    """Train MLP with early stopping."""
    n_total = X_train.shape[0]
    if n_total > sample_size:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_total, size=sample_size, replace=False)
        X_train_sample = X_train[idx]
        y_train_sample = y_train[idx]
    else:
        X_train_sample = X_train
        y_train_sample = y_train

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=8192,
        learning_rate_init=1e-3,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=0,
        verbose=True,
    )
    model.fit(X_train_sample, y_train_sample)
    return model


def feature_ablation_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baseline_metrics: Dict[str, float],
) -> Dict[str, Dict]:
    """One-feature ablation for linear regression."""
    ablation_results = {}
    
    for feature_idx, feature_name in enumerate(FEATURE_NAMES):
        X_train_ablated = np.delete(X_train, feature_idx, axis=1)
        X_test_ablated = np.delete(X_test, feature_idx, axis=1)
        
        model = LinearRegression(n_jobs=-1)
        model.fit(X_train_ablated, y_train)
        y_pred = model.predict(X_test_ablated)
        
        metrics = evaluate_predictions(y_test, y_pred)
        metrics['delta_rmse'] = metrics['rmse'] - baseline_metrics['rmse']
        
        ablation_results[feature_name] = metrics
        print(f"  {feature_name:20s} -> RMSE={metrics['rmse']:.4f}, Δ={metrics['delta_rmse']:+.4f}")
    
    return ablation_results


def compute_delta_p_statistics(y_flat: np.ndarray) -> Dict[str, float]:
    """Compute ΔP statistics."""
    return {
        'mean': float(np.mean(y_flat)),
        'std': float(np.std(y_flat)),
        'min': float(np.min(y_flat)),
        'max': float(np.max(y_flat)),
        'median': float(np.median(y_flat)),
    }


def plot_delta_p_distributions(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    test_sample_info_expanded: Dict,
):
    """Create histogram and phase-dependent distributions."""
    PLOT_DIR.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Overall histogram
    ax = axes[0, 0]
    ax.hist(y_train, bins=100, alpha=0.7, label='Train', density=True)
    ax.hist(y_val, bins=100, alpha=0.7, label='Val', density=True)
    ax.hist(y_test, bins=100, alpha=0.7, label='Test', density=True)
    ax.set_xlabel('ΔP (bar)')
    ax.set_ylabel('Density')
    ax.set_title('ΔP Distribution Across Splits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Injection phase
    ax = axes[0, 1]
    inj_mask = test_sample_info_expanded['inj_status'] == 1
    ax.hist(y_test[inj_mask], bins=100, alpha=0.7, color='blue', label='Injection', density=True)
    ax.set_xlabel('ΔP (bar)')
    ax.set_ylabel('Density')
    ax.set_title('ΔP Distribution: Injection Phase')
    ax.grid(True, alpha=0.3)
    
    # Production phase
    ax = axes[1, 0]
    prod_mask = test_sample_info_expanded['prod_status'] == 1
    ax.hist(y_test[prod_mask], bins=100, alpha=0.7, color='green', label='Production', density=True)
    ax.set_xlabel('ΔP (bar)')
    ax.set_ylabel('Density')
    ax.set_title('ΔP Distribution: Production Phase')
    ax.grid(True, alpha=0.3)
    
    # Shut-in phase
    ax = axes[1, 1]
    shutin_mask = (test_sample_info_expanded['inj_status'] == 0) & (test_sample_info_expanded['prod_status'] == 0)
    ax.hist(y_test[shutin_mask], bins=100, alpha=0.7, color='red', label='Shut-in', density=True)
    ax.set_xlabel('ΔP (bar)')
    ax.set_ylabel('Density')
    ax.set_title('ΔP Distribution: Shut-in Phase')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'delta_p_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {PLOT_DIR / 'delta_p_distributions.png'}")
    plt.close()


def plot_phase_dependent_performance(
    y_test: np.ndarray,
    y_pred_persistence: np.ndarray,
    y_pred_linear: np.ndarray,
    y_pred_mlp: np.ndarray,
    test_sample_info_expanded: Dict,
):
    """Plot performance metrics separated by phase."""
    PLOT_DIR.mkdir(exist_ok=True)
    
    inj_mask = test_sample_info_expanded['inj_status'] == 1
    prod_mask = test_sample_info_expanded['prod_status'] == 1
    shutin_mask = (test_sample_info_expanded['inj_status'] == 0) & (test_sample_info_expanded['prod_status'] == 0)
    
    phases = {
        'Injection': inj_mask,
        'Production': prod_mask,
        'Shut-in': shutin_mask,
    }
    
    models = {
        'Persistence': y_pred_persistence,
        'Linear': y_pred_linear,
        'MLP': y_pred_mlp,
    }
    
    metrics_by_phase = {}
    
    for phase_name, mask in phases.items():
        if np.sum(mask) == 0:
            continue
        y_phase = y_test[mask]
        metrics_by_phase[phase_name] = {}
        for model_name, y_pred in models.items():
            y_pred_phase = y_pred[mask]
            metrics = evaluate_predictions(y_phase, y_pred_phase)
            metrics_by_phase[phase_name][model_name] = metrics
    
    # Plot RMSE by phase
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    phases_list = list(metrics_by_phase.keys())
    x = np.arange(len(phases_list))
    width = 0.25
    for i, model_name in enumerate(models.keys()):
        rmse_vals = [metrics_by_phase[p][model_name]['rmse'] for p in phases_list]
        ax.bar(x + i*width, rmse_vals, width, label=model_name)
    ax.set_xlabel('Phase')
    ax.set_ylabel('RMSE (bar)')
    ax.set_title('RMSE by Operational Phase')
    ax.set_xticks(x + width)
    ax.set_xticklabels(phases_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot R² by phase
    ax = axes[1]
    for i, model_name in enumerate(models.keys()):
        r2_vals = [metrics_by_phase[p][model_name]['r2'] for p in phases_list]
        ax.bar(x + i*width, r2_vals, width, label=model_name)
    ax.set_xlabel('Phase')
    ax.set_ylabel('R²')
    ax.set_title('R² by Operational Phase')
    ax.set_xticks(x + width)
    ax.set_xticklabels(phases_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.99, color='k', linestyle='--', alpha=0.3, label='0.99')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'phase_dependent_performance.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {PLOT_DIR / 'phase_dependent_performance.png'}")
    plt.close()
    
    return metrics_by_phase


def main():
    print("=" * 80)
    print("PHASE 3: PRESSURE CHANGE PREDICTION (ΔP = P(t+1) - P(t))")
    print("=" * 80)
    
    data_dir = Path(__file__).parent
    PLOT_DIR.mkdir(exist_ok=True)
    
    # ========================================================================
    # STEP 1: BUILD DATASET
    # ========================================================================
    print("\n[1/5] Building ΔP dataset...")
    dataset = build_dataset(data_dir)
    X_train_flat, y_train_flat = flatten_dataset(dataset.X_train, dataset.y_train)
    X_val_flat, y_val_flat = flatten_dataset(dataset.X_val, dataset.y_val)
    X_test_flat, y_test_flat = flatten_dataset(dataset.X_test, dataset.y_test)
    
    # Expand sample info to match flattened test data
    n_cells = dataset.X_test.shape[1]
    test_sample_info_expanded = expand_sample_info(dataset.test_sample_info, n_cells)
    
    print(f"  Train:  {X_train_flat.shape[0]:,} samples")
    print(f"  Val:    {X_val_flat.shape[0]:,} samples")
    print(f"  Test:   {X_test_flat.shape[0]:,} samples")
    print(f"  Features: {X_train_flat.shape[1]} ({', '.join(FEATURE_NAMES)})")
    
    # ========================================================================
    # STEP 2: ANALYZE ΔP STATISTICS
    # ========================================================================
    print("\n[2/5] Analyzing ΔP statistics...")
    
    train_stats = compute_delta_p_statistics(y_train_flat)
    val_stats = compute_delta_p_statistics(y_val_flat)
    test_stats = compute_delta_p_statistics(y_test_flat)
    
    print(f"  Train ΔP: μ={train_stats['mean']:+.6f}, σ={train_stats['std']:.6f}")
    print(f"  Val ΔP:   μ={val_stats['mean']:+.6f}, σ={val_stats['std']:.6f}")
    print(f"  Test ΔP:  μ={test_stats['mean']:+.6f}, σ={test_stats['std']:.6f}")
    print(f"  Test range: [{test_stats['min']:.6f}, {test_stats['max']:.6f}]")
    
    # ========================================================================
    # STEP 3: CREATE ΔP VISUALIZATIONS
    # ========================================================================
    print("\n[3/5] Creating ΔP distribution plots...")
    plot_delta_p_distributions(y_train_flat, y_val_flat, y_test_flat, test_sample_info_expanded)
    
    # ========================================================================
    # STEP 4: TRAIN MODELS
    # ========================================================================
    print("\n[4/5] Training baseline models...")
    
    # Persistence: ΔP = 0
    print("  Training persistence baseline...")
    y_pred_persistence_test = persistence_baseline(X_test_flat)
    metrics_persistence = evaluate_predictions(y_test_flat, y_pred_persistence_test)
    print(f"    Persistence RMSE={metrics_persistence['rmse']:.6f}, "
          f"R²={metrics_persistence['r2']:.6f}")
    
    # Linear regression
    print("  Training linear regression...")
    linear_model = train_linear_model(X_train_flat, y_train_flat)
    y_pred_linear_test = linear_model.predict(X_test_flat)
    metrics_linear = evaluate_predictions(y_test_flat, y_pred_linear_test)
    print(f"    Linear RMSE={metrics_linear['rmse']:.6f}, "
          f"R²={metrics_linear['r2']:.6f}")
    
    # MLP
    print("  Training MLP...")
    mlp_model = train_mlp_model(X_train_flat, y_train_flat, X_val_flat, y_val_flat)
    y_pred_mlp_test = mlp_model.predict(X_test_flat)
    metrics_mlp = evaluate_predictions(y_test_flat, y_pred_mlp_test)
    print(f"    MLP RMSE={metrics_mlp['rmse']:.6f}, "
          f"R²={metrics_mlp['r2']:.6f}")
    
    # ========================================================================
    # STEP 5: FEATURE ABLATION
    # ========================================================================
    print("\n[5/5] Feature ablation (linear regression)...")
    ablation_results = feature_ablation_linear(
        X_train_flat, y_train_flat, X_test_flat, y_test_flat, metrics_linear
    )
    
    # ========================================================================
    # PHASE-DEPENDENT PERFORMANCE
    # ========================================================================
    print("\nPhase-dependent performance analysis...")
    metrics_by_phase = plot_phase_dependent_performance(
        y_test_flat,
        y_pred_persistence_test,
        y_pred_linear_test,
        y_pred_mlp_test,
        test_sample_info_expanded,
    )
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\nSaving results...")
    
    summary = {
        'dataset': {
            'train_samples': int(X_train_flat.shape[0]),
            'val_samples': int(X_val_flat.shape[0]),
            'test_samples': int(X_test_flat.shape[0]),
            'cells_per_sample': int(dataset.X_train.shape[1]),
            'features': FEATURE_NAMES,
        },
        'delta_p_statistics': {
            'train': train_stats,
            'val': val_stats,
            'test': test_stats,
        },
        'baseline_performance_test_set': {
            'persistence': metrics_persistence,
            'linear': metrics_linear,
            'mlp': metrics_mlp,
        },
        'feature_ablation_linear': ablation_results,
        'phase_dependent_performance': {
            phase: {model: metrics for model, metrics in models.items()}
            for phase, models in metrics_by_phase.items()
        },
    }
    
    summary_file = Path('PHASE_3_PRESSURE_CHANGE_SUMMARY.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: {summary_file}")
    
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - Summary: {summary_file}")
    print(f"  - Plots: {PLOT_DIR}/")
    
    return summary


if __name__ == '__main__':
    main()
