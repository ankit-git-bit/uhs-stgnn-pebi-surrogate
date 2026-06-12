"""
Phase 4: Spatial and Temporal Feature Engineering for ΔP Prediction

This phase determines which engineered features are essential for pressure-change
prediction by systematically comparing:
  A. Baseline (cell-local static features only)
  B. Spatially enriched (adds spatial geometry and gradients)
  C. Spatial + Temporal (adds pressure history and time-dependent features)

Models trained: Linear, Ridge, Random Forest, MLP

Output: Feature importance rankings and recommendations for architecture choice.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import uniform_filter
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[2]
PLOT_DIR = project_root / "results" / "figures"


@dataclass
class DatasetSplit:
    """Container for train/val/test splits with metadata."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    feature_set_name: str
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
        'coord_X': np.squeeze(data.get('coord_X', np.zeros(10116))).astype(np.float32),
        'coord_Z': np.squeeze(data.get('coord_Z', np.zeros(10116))).astype(np.float32),
    }

    P = case_data['P_matrix']
    if P.ndim != 2:
        P = np.squeeze(P)
    if P.ndim != 2:
        raise ValueError(f'Unexpected P_matrix shape: {P.shape}')
    case_data['P_matrix'] = P

    return case_data


def create_spatial_grid_features(
    n_cells: int,
    perm_initial: np.ndarray,
    poro_initial: np.ndarray,
    coord_X: np.ndarray,
    coord_Z: np.ndarray,
    P_matrix: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Create spatial features for the grid:
    - Cell coordinates (if available)
    - Distance to injector (assume at origin)
    - Distance to producer (assume at end of domain)
    - Spatial gradients
    - Neighbor averages
    """
    features = {}
    
    # Coordinate features
    features['coord_X'] = coord_X
    features['coord_Z'] = coord_Z
    
    # Distance to injector (assume injector at (0, 0))
    dist_to_inj = np.sqrt(coord_X**2 + coord_Z**2)
    features['dist_to_injector'] = dist_to_inj
    
    # Distance to producer (assume producer at max X, Z=0)
    max_X = np.max(np.abs(coord_X)) if np.max(np.abs(coord_X)) > 0 else 1000.0
    max_Z = np.max(np.abs(coord_Z)) if np.max(np.abs(coord_Z)) > 0 else 500.0
    prod_loc_X = max_X
    prod_loc_Z = 0.0
    dist_to_prod = np.sqrt((coord_X - prod_loc_X)**2 + (coord_Z - prod_loc_Z)**2)
    features['dist_to_producer'] = dist_to_prod
    
    # Permeability gradients (approximate using finite difference)
    # Reshape to 2D grid assuming square or rect layout
    try:
        grid_size = int(np.sqrt(n_cells))
        if grid_size * grid_size != n_cells:
            # Try other factorizations
            for gs in [int(np.sqrt(n_cells)) - 1, int(np.sqrt(n_cells)) + 1, 100, 101]:
                if n_cells % gs == 0:
                    grid_size = gs
                    break
        
        perm_grid = perm_initial.reshape(grid_size, -1) if n_cells % grid_size == 0 else perm_initial
        
        # Compute gradients using finite differences
        if perm_grid.ndim == 2:
            perm_gradX = np.gradient(perm_grid, axis=1)
            perm_gradZ = np.gradient(perm_grid, axis=0)
            features['perm_grad_magnitude'] = np.sqrt(perm_gradX**2 + perm_gradZ**2).flatten()[:n_cells]
        else:
            features['perm_grad_magnitude'] = np.zeros(n_cells)
    except:
        features['perm_grad_magnitude'] = np.zeros(n_cells)
    
    # Pressure-based spatial features (computed per timestep)
    # For now, return template - will be filled per timestep
    
    return features


def compute_pressure_gradients(P_slice: np.ndarray, n_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local pressure gradients for a pressure field at one timestep.
    P_slice: pressure at all cells for timestep t (shape: n_cells)
    """
    try:
        # Attempt 2D grid layout
        grid_size = int(np.sqrt(n_cells))
        if grid_size * grid_size != n_cells:
            for gs in [int(np.sqrt(n_cells)) - 1, int(np.sqrt(n_cells)) + 1, 100, 101]:
                if n_cells % gs == 0:
                    grid_size = gs
                    break
        
        if n_cells % grid_size == 0:
            P_grid = P_slice.reshape(grid_size, -1)
            grad_X = np.gradient(P_grid, axis=1)
            grad_Z = np.gradient(P_grid, axis=0)
            return grad_X.flatten()[:n_cells], grad_Z.flatten()[:n_cells]
    except:
        pass
    
    return np.zeros(n_cells), np.zeros(n_cells)


def compute_neighbor_averages(
    field: np.ndarray,
    n_cells: int,
    window_size: int = 3,
) -> np.ndarray:
    """
    Compute local neighbor averages for a field (pressure, permeability, etc).
    """
    try:
        grid_size = int(np.sqrt(n_cells))
        if grid_size * grid_size != n_cells:
            for gs in [int(np.sqrt(n_cells)) - 1, int(np.sqrt(n_cells)) + 1, 100, 101]:
                if n_cells % gs == 0:
                    grid_size = gs
                    break
        
        if n_cells % grid_size == 0:
            field_grid = field.reshape(grid_size, -1)
            neighbors_avg = uniform_filter(field_grid, size=window_size, mode='constant')
            return neighbors_avg.flatten()[:n_cells]
    except:
        pass
    
    return field  # Fallback: return field unchanged


def build_dataset_with_features(
    data_dir: Path,
    feature_set: str = 'baseline',
) -> DatasetSplit:
    """
    Build dataset with different feature sets:
    - 'baseline': Only cell-local static features (Phase 2/3)
    - 'spatial': Add spatial geometry and pressure gradients
    - 'spatiotemporal': Add temporal history and time markers
    """
    
    baseline_features = [
        'Perm_initial', 'Poro_initial', 'P(t)',
        'inj_BHP', 'prod_BHP', 'inj_H2', 'prod_H2', 'inj_status', 'prod_status',
    ]
    
    spatial_features = baseline_features + [
        'X_coord', 'Z_coord',
        'dist_to_injector', 'dist_to_producer',
        'perm_grad_magnitude',
        'P_grad_X', 'P_grad_Z', 'P_grad_magnitude',
        'neighbor_P_avg', 'neighbor_Perm_avg',
    ]
    
    temporal_features = spatial_features + [
        'delta_P_t_minus_1', 'delta_P_t_minus_2',
        'rolling_delta_P_avg_3',
        'time_since_inj_start',
        'time_since_prod_start',
        'phase_change_indicator',
    ]
    
    feature_map = {
        'baseline': baseline_features,
        'spatial': spatial_features,
        'spatiotemporal': temporal_features,
    }
    
    selected_features = feature_map[feature_set]
    n_features = len(selected_features)
    
    print(f"\n  Building {feature_set} dataset with {n_features} features:")
    print(f"    {selected_features[:5]}...")
    
    case_files = sorted(data_dir.glob('Case_*.mat'))
    if len(case_files) != 5:
        raise FileNotFoundError('Expected 5 Case_*.mat files')
    
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
        print(f"    Case {case_id+1}/5: {case_path.name}")
        case_data = load_case(case_path)
        
        perm = case_data['Perm_initial']
        poro = case_data['Poro_initial']
        P = case_data['P_matrix']
        inj_BHP = case_data['inj_BHP']
        prod_BHP = case_data['prod_BHP']
        inj_H2 = case_data['inj_H2']
        prod_H2 = case_data['prod_H2']
        inj_status = case_data['inj_status']
        prod_status = case_data['prod_status']
        coord_X = case_data['coord_X']
        coord_Z = case_data['coord_Z']
        
        n_cells, n_steps = P.shape
        n_pairs = n_steps - 1
        
        # Generate spatial features (time-independent)
        spatial_dict = create_spatial_grid_features(
            n_cells, perm, poro, coord_X, coord_Z, P
        )
        
        # Initialize feature arrays
        X = np.empty((n_pairs, n_cells, n_features), dtype=np.float32)
        y = np.empty((n_pairs, n_cells), dtype=np.float32)
        
        sample_info = {
            'case_id': np.full(n_pairs, case_id, dtype=np.int32),
            'timestep': np.arange(n_pairs, dtype=np.int32),
            'inj_status': inj_status[:-1].astype(np.int32),
            'prod_status': prod_status[:-1].astype(np.int32),
        }
        
        # Build feature matrix for each timestep
        for t in range(n_pairs):
            feat_idx = 0
            
            # Baseline features (always included)
            X[t, :, feat_idx] = perm
            feat_idx += 1
            X[t, :, feat_idx] = poro
            feat_idx += 1
            X[t, :, feat_idx] = P[:, t]
            feat_idx += 1
            X[t, :, feat_idx] = inj_BHP[t]
            feat_idx += 1
            X[t, :, feat_idx] = prod_BHP[t]
            feat_idx += 1
            X[t, :, feat_idx] = inj_H2[t]
            feat_idx += 1
            X[t, :, feat_idx] = prod_H2[t]
            feat_idx += 1
            X[t, :, feat_idx] = inj_status[t]
            feat_idx += 1
            X[t, :, feat_idx] = prod_status[t]
            feat_idx += 1
            
            # Spatial features (if enabled)
            if feature_set in ['spatial', 'spatiotemporal']:
                X[t, :, feat_idx] = spatial_dict['coord_X']
                feat_idx += 1
                X[t, :, feat_idx] = spatial_dict['coord_Z']
                feat_idx += 1
                X[t, :, feat_idx] = spatial_dict['dist_to_injector']
                feat_idx += 1
                X[t, :, feat_idx] = spatial_dict['dist_to_producer']
                feat_idx += 1
                X[t, :, feat_idx] = spatial_dict['perm_grad_magnitude']
                feat_idx += 1
                
                # Pressure gradients (computed per timestep)
                P_grad_X, P_grad_Z = compute_pressure_gradients(P[:, t], n_cells)
                X[t, :, feat_idx] = P_grad_X
                feat_idx += 1
                X[t, :, feat_idx] = P_grad_Z
                feat_idx += 1
                P_grad_mag = np.sqrt(P_grad_X**2 + P_grad_Z**2)
                X[t, :, feat_idx] = P_grad_mag
                feat_idx += 1
                
                # Neighbor averages
                X[t, :, feat_idx] = compute_neighbor_averages(P[:, t], n_cells)
                feat_idx += 1
                X[t, :, feat_idx] = compute_neighbor_averages(perm, n_cells)
                feat_idx += 1
            
            # Temporal features (if enabled)
            if feature_set == 'spatiotemporal':
                # ΔP(t-1)
                if t > 0:
                    X[t, :, feat_idx] = P[:, t] - P[:, t-1]
                else:
                    X[t, :, feat_idx] = 0.0
                feat_idx += 1
                
                # ΔP(t-2)
                if t > 1:
                    X[t, :, feat_idx] = P[:, t-1] - P[:, t-2]
                else:
                    X[t, :, feat_idx] = 0.0
                feat_idx += 1
                
                # Rolling average of ΔP
                if t > 2:
                    roll_avg = (P[:, t] - P[:, t-1] + P[:, t-1] - P[:, t-2] + P[:, t-2] - P[:, t-3]) / 3.0
                    X[t, :, feat_idx] = roll_avg
                else:
                    X[t, :, feat_idx] = P[:, t] - P[:, t-1] if t > 0 else 0.0
                feat_idx += 1
                
                # Time since injection started (simple: timestep index)
                X[t, :, feat_idx] = float(t)
                feat_idx += 1
                
                # Time since production started (assume after first half)
                prod_start = n_pairs // 2
                time_since_prod = max(0.0, float(t - prod_start))
                X[t, :, feat_idx] = time_since_prod
                feat_idx += 1
                
                # Phase change indicator
                phase_change = (int(inj_status[t] != inj_status[t-1]) if t > 0 else 0) + \
                               (int(prod_status[t] != prod_status[t-1]) if t > 0 else 0)
                X[t, :, feat_idx] = float(phase_change)
                feat_idx += 1
            
            # Target: ΔP
            y[t, :] = P[:, t + 1] - P[:, t]
        
        # Train/val/test split
        if case_id < 4:
            X_trainval, X_val, y_trainval, y_val = train_test_split(
                X, y, test_size=0.10, random_state=0, shuffle=True
            )
            all_train_X.append(X_trainval)
            all_train_y.append(y_trainval)
            all_val_X.append(X_val)
            all_val_y.append(y_val)
        else:
            all_test_X.append(X)
            all_test_y.append(y)
            for key in test_sample_info:
                test_sample_info[key].append(sample_info[key])
    
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
        feature_names=selected_features,
        feature_set_name=feature_set,
        test_sample_info=test_sample_info,
    )


def flatten_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten from (n_pairs, n_cells, n_features) to (n_total, n_features)."""
    return X.reshape(-1, X.shape[-1]), y.reshape(-1)


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


def train_and_evaluate_models(
    dataset_baseline: DatasetSplit,
    dataset_spatial: DatasetSplit,
    dataset_spatiotemporal: DatasetSplit,
) -> Dict:
    """Train models on all three feature sets and compare."""
    
    results = {}
    
    for dataset in [dataset_baseline, dataset_spatial, dataset_spatiotemporal]:
        feature_set = dataset.feature_set_name
        print(f"\n{'='*70}")
        print(f"Training on: {feature_set.upper()} ({len(dataset.feature_names)} features)")
        print(f"{'='*70}")
        
        X_train_flat, y_train_flat = flatten_dataset(dataset.X_train, dataset.y_train)
        X_val_flat, y_val_flat = flatten_dataset(dataset.X_val, dataset.y_val)
        X_test_flat, y_test_flat = flatten_dataset(dataset.X_test, dataset.y_test)
        
        # Subsample for efficiency
        rng = np.random.RandomState(0)
        sample_size = min(1_000_000, X_train_flat.shape[0])
        idx = rng.choice(X_train_flat.shape[0], sample_size, replace=False)
        X_train_flat_sample = X_train_flat[idx]
        y_train_flat_sample = y_train_flat[idx]
        
        # Normalize features
        scaler = StandardScaler()
        X_train_flat_sample = scaler.fit_transform(X_train_flat_sample)
        X_val_flat = scaler.transform(X_val_flat)
        X_test_flat = scaler.transform(X_test_flat)
        
        results[feature_set] = {}
        
        # Linear Regression
        print("\n  Training Linear Regression...")
        lr = LinearRegression(n_jobs=-1)
        lr.fit(X_train_flat_sample, y_train_flat_sample)
        y_pred_test = lr.predict(X_test_flat)
        metrics_lr = evaluate_predictions(y_test_flat, y_pred_test)
        results[feature_set]['linear'] = {
            'metrics': metrics_lr,
            'model': lr,
            'scaler': scaler,
        }
        print(f"    Linear RMSE={metrics_lr['rmse']:.4f}, R²={metrics_lr['r2']:.6f}")
        
        # Ridge Regression
        print("  Training Ridge Regression...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_flat_sample, y_train_flat_sample)
        y_pred_test = ridge.predict(X_test_flat)
        metrics_ridge = evaluate_predictions(y_test_flat, y_pred_test)
        results[feature_set]['ridge'] = {
            'metrics': metrics_ridge,
            'model': ridge,
            'scaler': scaler,
        }
        print(f"    Ridge RMSE={metrics_ridge['rmse']:.4f}, R²={metrics_ridge['r2']:.6f}")
        
        # Random Forest (subsampled for speed)
        print("  Training Random Forest...")
        rf_sample_size = min(300_000, X_train_flat_sample.shape[0])
        idx_rf = rng.choice(X_train_flat_sample.shape[0], rf_sample_size, replace=False)
        rf = RandomForestRegressor(
            n_estimators=20, max_depth=12, n_jobs=-1, random_state=0, verbose=0
        )
        rf.fit(X_train_flat_sample[idx_rf], y_train_flat_sample[idx_rf])
        y_pred_test = rf.predict(X_test_flat)
        metrics_rf = evaluate_predictions(y_test_flat, y_pred_test)
        results[feature_set]['random_forest'] = {
            'metrics': metrics_rf,
            'model': rf,
            'feature_importance': rf.feature_importances_,
        }
        print(f"    Random Forest RMSE={metrics_rf['rmse']:.4f}, R²={metrics_rf['r2']:.6f}")
        
        # MLP
        print("  Training MLP...")
        mlp = MLPRegressor(
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
            verbose=False,
        )
        mlp.fit(X_train_flat_sample, y_train_flat_sample)
        y_pred_test = mlp.predict(X_test_flat)
        metrics_mlp = evaluate_predictions(y_test_flat, y_pred_test)
        results[feature_set]['mlp'] = {
            'metrics': metrics_mlp,
            'model': mlp,
            'scaler': scaler,
        }
        print(f"    MLP RMSE={metrics_mlp['rmse']:.4f}, R²={metrics_mlp['r2']:.6f}")
        
        # Store feature names for importance analysis
        results[feature_set]['feature_names'] = dataset.feature_names
    
    return results


def plot_performance_comparison(results: Dict):
    """Create comparison plots across feature sets and models."""
    PLOT_DIR.mkdir(exist_ok=True)
    
    feature_sets = list(results.keys())
    models = ['linear', 'ridge', 'random_forest', 'mlp']
    
    # Prepare data for plotting
    rmse_by_model = {m: [] for m in models}
    r2_by_model = {m: [] for m in models}
    
    for fs in feature_sets:
        for m in models:
            rmse_by_model[m].append(results[fs][m]['metrics']['rmse'])
            r2_by_model[m].append(results[fs][m]['metrics']['r2'])
    
    # Plot RMSE comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    x = np.arange(len(feature_sets))
    width = 0.2
    for i, m in enumerate(models):
        ax.bar(x + i*width, rmse_by_model[m], width, label=m.replace('_', ' ').title())
    ax.set_xlabel('Feature Set')
    ax.set_ylabel('RMSE (bar)')
    ax.set_title('Model Comparison: RMSE vs Feature Set')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(fs.replace('_', ' ').title() for fs in feature_sets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1]
    for i, m in enumerate(models):
        ax.bar(x + i*width, r2_by_model[m], width, label=m.replace('_', ' ').title())
    ax.set_xlabel('Feature Set')
    ax.set_ylabel('R²')
    ax.set_title('Model Comparison: R² vs Feature Set')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(fs.replace('_', ' ').title() for fs in feature_sets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.35, color='r', linestyle='--', alpha=0.5, label='Phase 3 MLP baseline')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'model_performance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {PLOT_DIR / 'model_performance_comparison.png'}")
    plt.close()


def plot_feature_importance(results: Dict):
    """Plot feature importance from Random Forest models."""
    PLOT_DIR.mkdir(exist_ok=True)
    
    feature_sets = [fs for fs in results.keys() if 'feature_importance' in results[fs]['random_forest']]
    
    if not feature_sets:
        print("  No feature importance data available")
        return
    
    fig, axes = plt.subplots(1, len(feature_sets), figsize=(5*len(feature_sets), 6))
    if len(feature_sets) == 1:
        axes = [axes]
    
    for ax_idx, fs in enumerate(feature_sets):
        importance = results[fs]['random_forest']['feature_importance']
        feature_names = results[fs]['feature_names']
        
        # Sort by importance
        indices = np.argsort(importance)[-15:]  # Top 15
        
        ax = axes[ax_idx]
        ax.barh(range(len(indices)), importance[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'{fs.replace("_", " ").title()}\n(Top 15 Features)')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'feature_importance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR / 'feature_importance_comparison.png'}")
    plt.close()


def generate_feature_importance_rankings(results: Dict) -> Dict:
    """Generate comprehensive feature importance rankings."""
    rankings = {}
    
    for fs in results.keys():
        if 'feature_importance' not in results[fs]['random_forest']:
            continue
        
        importance = results[fs]['random_forest']['feature_importance']
        feature_names = results[fs]['feature_names']
        
        # Create ranking
        indices = np.argsort(importance)[::-1]
        ranked = [
            {
                'rank': i + 1,
                'feature': feature_names[idx],
                'importance': float(importance[idx]),
                'cumulative': float(np.sum(importance[indices[:i+1]])),
            }
            for i, idx in enumerate(indices)
        ]
        
        rankings[fs] = ranked
    
    return rankings


def main():
    print("=" * 80)
    print("PHASE 4: SPATIAL AND TEMPORAL FEATURE ENGINEERING")
    print("=" * 80)
    
    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    results_metrics_dir = project_root / "results" / "metrics"
    
    results_metrics_dir.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: BUILD THREE DATASETS WITH DIFFERENT FEATURE SETS
    # ========================================================================
    print("\n[1/4] Building datasets with different feature sets...")
    print(f"\n  Baseline feature set (Phase 3):")
    dataset_baseline = build_dataset_with_features(data_raw_dir, feature_set='baseline')
    print(f"    Shape: {dataset_baseline.X_train.shape}")
    
    print(f"\n  Spatial feature set:")
    dataset_spatial = build_dataset_with_features(data_raw_dir, feature_set='spatial')
    print(f"    Shape: {dataset_spatial.X_train.shape}")
    
    print(f"\n  Spatiotemporal feature set:")
    dataset_spatiotemporal = build_dataset_with_features(data_raw_dir, feature_set='spatiotemporal')
    print(f"    Shape: {dataset_spatiotemporal.X_train.shape}")
    
    # ========================================================================
    # STEP 2: TRAIN MODELS ON ALL FEATURE SETS
    # ========================================================================
    print("\n[2/4] Training models on all feature sets...")
    results = train_and_evaluate_models(
        dataset_baseline,
        dataset_spatial,
        dataset_spatiotemporal,
    )
    
    # ========================================================================
    # STEP 3: CREATE COMPARISON VISUALIZATIONS
    # ========================================================================
    print("\n[3/4] Creating comparison visualizations...")
    plot_performance_comparison(results)
    plot_feature_importance(results)
    
    # ========================================================================
    # STEP 4: GENERATE RANKINGS AND SUMMARY
    # ========================================================================
    print("\n[4/4] Generating feature importance rankings...")
    rankings = generate_feature_importance_rankings(results)
    
    # Prepare summary report
    summary = {
        'feature_sets': {
            'baseline': {
                'description': 'Cell-local static features only (Phase 3)',
                'n_features': len(dataset_baseline.feature_names),
                'features': dataset_baseline.feature_names,
            },
            'spatial': {
                'description': 'Baseline + spatial geometry and pressure gradients',
                'n_features': len(dataset_spatial.feature_names),
                'features': dataset_spatial.feature_names,
            },
            'spatiotemporal': {
                'description': 'Spatial + temporal history and time markers',
                'n_features': len(dataset_spatiotemporal.feature_names),
                'features': dataset_spatiotemporal.feature_names,
            },
        },
        'model_performance': {},
        'feature_importance_rankings': rankings,
    }
    
    # Add model performance for each feature set
    for fs in results.keys():
        summary['model_performance'][fs] = {}
        for model_name in ['linear', 'ridge', 'random_forest', 'mlp']:
            summary['model_performance'][fs][model_name] = results[fs][model_name]['metrics']
    
    # Save summary
    summary_file = results_metrics_dir / 'PHASE_4_FEATURE_ENGINEERING_SUMMARY.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_file}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("PHASE 4 SUMMARY")
    print("=" * 80)
    print("\nBest Performance by Feature Set:")
    for fs in ['baseline', 'spatial', 'spatiotemporal']:
        if fs not in results:
            continue
        best_r2 = -999
        best_model = None
        for model in ['linear', 'ridge', 'random_forest', 'mlp']:
            r2 = results[fs][model]['metrics']['r2']
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
        rmse = results[fs][best_model]['metrics']['rmse']
        print(f"  {fs:20s}: {best_model:15s} RMSE={rmse:.4f}, R²={best_r2:.6f}")
    
    return summary, results


if __name__ == '__main__':
    main()
