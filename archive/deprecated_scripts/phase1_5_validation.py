"""
Phase 1.5 Validation for the UHS pressure surrogate dataset.

This script analyzes the dataset representation, feature relevance,
spatial topology, and candidate structured-grid errors.

Usage:
    python phase1_5_validation.py

Outputs:
    - printed validation metrics
    - optionally saves a JSON summary
"""

from pathlib import Path
from collections import Counter
import json

import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from inspect_phase1 import load_mat_file

FEATURE_NAMES = [
    'Perm_initial',
    'Poro_initial',
    'P(t)',
    'inj_BHP',
    'prod_BHP',
    'inj_H2',
    'prod_H2',
    'dt_days',
    't_index',
]


def load_dataset(npz_path: Path):
    data = np.load(npz_path)
    return data


def raw_pressure_matrix(raw):
    values = np.array(raw['P_matrix'][1], dtype=np.float32)
    n_cells = len(raw['X_coords'][1])
    if values.size % n_cells != 0:
        raise ValueError('P_matrix length is not divisible by number of cells')
    n_timesteps = values.size // n_cells
    return values.reshape(n_timesteps, n_cells)


def summarize_spatial_topology(raw):
    X = np.array(raw['X_coords'][1], dtype=np.float64)
    Z = np.array(raw['Z_coords'][1], dtype=np.float64)
    x_counts = Counter(X)
    z_counts = Counter(Z)

    return {
        'n_cells': len(X),
        'n_unique_X': len(x_counts),
        'n_unique_Z': len(z_counts),
        'X_min': float(X.min()),
        'X_max': float(X.max()),
        'Z_min': float(Z.min()),
        'Z_max': float(Z.max()),
        'X_count_min': int(min(x_counts.values())),
        'X_count_max': int(max(x_counts.values())),
        'X_count_mean': float(np.mean(list(x_counts.values()))),
        'Z_count_min': int(min(z_counts.values())),
        'Z_count_max': int(max(z_counts.values())),
        'Z_count_mean': float(np.mean(list(z_counts.values()))),
        'X_count_histogram': Counter(x_counts.values()),
        'Z_count_histogram': Counter(z_counts.values()),
    }


def summarize_status(raw):
    inj = np.array(raw['inj_status'][1], dtype=int)
    prod = np.array(raw['prod_status'][1], dtype=int)
    dt = np.array(raw['dt_days'][1], dtype=np.float32)

    return {
        'dt_days_unique': sorted(set(dt.tolist())),
        'inj_status_unique': sorted(set(inj.tolist())),
        'prod_status_unique': sorted(set(prod.tolist())),
        'inj_status_count': int(inj.sum()),
        'prod_status_count': int(prod.sum()),
        'both_on': int(np.sum((inj == 1) & (prod == 1))),
        'both_off': int(np.sum((inj == 0) & (prod == 0))),
        'inj_changes': np.where(np.diff(inj) != 0)[0].tolist(),
        'prod_changes': np.where(np.diff(prod) != 0)[0].tolist(),
        'sequence_length': len(inj),
    }


def feature_importance(data, sample_size: int = 200000, random_seed: int = 0):
    X_train = data['X_train']
    Y_train = data['Y_train']
    n_samples, n_cells, n_features = X_train.shape
    flat_X = X_train.reshape(-1, n_features)
    flat_Y = Y_train.reshape(-1)
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(flat_X.shape[0], size=min(sample_size, flat_X.shape[0]), replace=False)
    X_sub = flat_X[idx]
    Y_sub = flat_Y[idx]
    P_t = X_sub[:, 2]
    Y_delta = Y_sub - P_t

    corr_Y = [float(pearsonr(X_sub[:, i], Y_sub)[0]) if np.std(X_sub[:, i]) > 0 else float('nan') for i in range(n_features)]
    corr_dY = [float(pearsonr(X_sub[:, i], Y_delta)[0]) if np.std(X_sub[:, i]) > 0 else float('nan') for i in range(n_features)]
    mi_Y = mutual_info_regression(X_sub, Y_sub, random_state=random_seed)
    mi_dY = mutual_info_regression(X_sub, Y_delta, random_state=random_seed)

    scaler = StandardScaler().fit(X_sub)
    X_scaled = scaler.transform(X_sub)
    model = Ridge(alpha=1.0).fit(X_scaled, Y_sub)
    base_pred = model.predict(X_scaled)
    base_rmse = float(np.sqrt(np.mean((base_pred - Y_sub) ** 2)))
    ablation = []
    for i in range(n_features):
        X_reduced = np.delete(X_scaled, i, axis=1)
        reduced_model = Ridge(alpha=1.0).fit(X_reduced, Y_sub)
        pred = reduced_model.predict(X_reduced)
        rmse = float(np.sqrt(np.mean((pred - Y_sub) ** 2)))
        ablation.append({
            'feature': FEATURE_NAMES[i],
            'rmse': rmse,
            'delta_rmse': rmse - base_rmse,
        })

    return {
        'correlation_with_Y': dict(zip(FEATURE_NAMES, corr_Y)),
        'correlation_with_delta_Y': dict(zip(FEATURE_NAMES, corr_dY)),
        'mutual_info_with_Y': dict(zip(FEATURE_NAMES, mi_Y.tolist())),
        'mutual_info_with_delta_Y': dict(zip(FEATURE_NAMES, mi_dY.tolist())),
        'ablation': {
            'base_rmse': base_rmse,
            'results': ablation,
        },
    }


def grid_interpolation_errors(raw, snapshots=None):
    X = np.array(raw['X_coords'][1], dtype=np.float32)
    Z = np.array(raw['Z_coords'][1], dtype=np.float32)
    pressures = raw_pressure_matrix(raw)
    n_timesteps = pressures.shape[0]
    if snapshots is None:
        snapshots = [0, n_timesteps // 2, n_timesteps - 1]

    def project_error(nx: int, nz: int, snapshot: int):
        xi = np.linspace(X.min(), X.max(), nx)
        zi = np.linspace(Z.min(), Z.max(), nz)
        grid_x, grid_z = np.meshgrid(xi, zi, indexing='xy')
        points = np.column_stack([X, Z])
        values = pressures[snapshot]
        grid_vals = griddata(points, values, (grid_x, grid_z), method='linear')
        if np.any(np.isnan(grid_vals)):
            grid_vals_nearest = griddata(points, values, (grid_x, grid_z), method='nearest')
            grid_vals[np.isnan(grid_vals)] = grid_vals_nearest[np.isnan(grid_vals)]
        interp = RegularGridInterpolator((xi, zi), grid_vals.T, bounds_error=False, fill_value=None)
        reconstructed = interp(points)
        rmse = float(np.sqrt(np.mean((reconstructed - values) ** 2)))
        mae = float(np.mean(np.abs(reconstructed - values)))
        return {
            'nx': nx,
            'nz': nz,
            'snapshot': int(snapshot),
            'rmse': rmse,
            'mae': mae,
        }

    results = []
    for nx, nz in [(64, 32), (128, 64), (256, 128)]:
        for snapshot in snapshots:
            results.append(project_error(nx, nz, snapshot))
    return results


def main():
    root = Path('.')
    npz_path = root / 'pressure_surrogate_dataset.npz'
    cases = sorted(root.glob('Case_*.mat'))
    if not npz_path.exists() or len(cases) == 0:
        raise FileNotFoundError('Required dataset or raw cases not found.')

    data = load_dataset(npz_path)
    raw = load_mat_file(cases[0])

    report = {
        'dataset_shape': {
            'X_train': data['X_train'].shape,
            'Y_train': data['Y_train'].shape,
            'X_val': data['X_val'].shape,
            'Y_val': data['Y_val'].shape,
            'X_test': data['X_test'].shape,
            'Y_test': data['Y_test'].shape,
        },
        'spatial_summary': summarize_spatial_topology(raw),
        'status_summary': summarize_status(raw),
        'feature_importance_summary': feature_importance(data),
        'interpolation_errors': grid_interpolation_errors(raw),
    }

    print(json.dumps(report, indent=2))
    output_path = root / 'PHASE_1_5_VALIDATION_SUMMARY.json'
    with output_path.open('w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
