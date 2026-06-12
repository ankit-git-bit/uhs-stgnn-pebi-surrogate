"""
Phase 2 Baseline Pressure Evolution Models

This script builds a scientifically defensible Phase 2 baseline using the
recommended feature set and native cell representation.

Baseline models:
  A. Persistence: P(t+1) = P(t)
  B. Linear regression
  C. Small MLP (pointwise)

Metrics:
  - RMSE
  - MAE
  - Relative Error
  - R²

The script also performs one-feature ablation for the linear model,
identifies worst cells/timesteps, and produces diagnostic plots.
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

PLOT_DIR = Path('phase2_plots')


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
    sample_ids = np.arange(n_pairs, dtype=np.int32)
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
        y[t, :] = P[:, t + 1]

    return {
        'X': X,
        'y': y,
        'sample_info': sample_info,
    }


def build_dataset(data_dir: Path) -> DatasetSplit:
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

    rng = np.random.default_rng(0)

    for case_id, case_path in enumerate(case_files):
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
    return X.reshape(-1, X.shape[-1]), y.reshape(-1)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
    return X[..., 2].reshape(-1)


def train_linear_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
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


def feature_ablation(
    X_train_flat: np.ndarray,
    y_train_flat: np.ndarray,
    X_test_flat: np.ndarray,
    y_test_flat: np.ndarray,
) -> List[Dict[str, object]]:
    results = []
    for i, feat in enumerate(FEATURE_NAMES):
        keep_cols = [j for j in range(len(FEATURE_NAMES)) if j != i]
        model = LinearRegression(n_jobs=-1)
        model.fit(X_train_flat[:, keep_cols], y_train_flat)
        y_pred = model.predict(X_test_flat[:, keep_cols])
        metrics = evaluate_predictions(y_test_flat, y_pred)
        results.append({
            'removed_feature': feat,
            'metrics': metrics,
        })
    return results


def sample_weights_by_phase(test_info: Dict[str, np.ndarray], y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    results = {}
    test_len = y_true.shape[0]
    # each row corresponds to one sample of 10116 cells
    n_cells = 10116
    y_true_samples = y_true.reshape(-1, n_cells)
    y_pred_samples = y_pred.reshape(-1, n_cells)

    for phase_name, mask in [
        ('inj_on', test_info['inj_status'] == 1),
        ('prod_on', test_info['prod_status'] == 1),
        ('both_on', (test_info['inj_status'] == 1) & (test_info['prod_status'] == 1)),
        ('both_off', (test_info['inj_status'] == 0) & (test_info['prod_status'] == 0)),
    ]:
        if np.any(mask):
            y_true_phase = y_true_samples[mask].reshape(-1)
            y_pred_phase = y_pred_samples[mask].reshape(-1)
            results[phase_name] = evaluate_predictions(y_true_phase, y_pred_phase)
        else:
            results[phase_name] = {'rmse': None, 'mae': None, 'r2': None, 'relative_error': None}
    return results


def plot_predictions_vs_true(y_true: np.ndarray, predictions: Dict[str, np.ndarray], output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    n_plot = min(y_true.shape[0], 20000)
    rng = np.random.default_rng(0)
    idx = rng.choice(y_true.shape[0], size=n_plot, replace=False)

    plt.figure(figsize=(8, 8))
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], c='k', lw=1)
    for name, y_pred in predictions.items():
        plt.scatter(y_true[idx], y_pred[idx], s=4, alpha=0.2, label=name)
    plt.xlabel('True pressure (bar)')
    plt.ylabel('Predicted pressure (bar)')
    plt.legend()
    plt.title('Predicted vs True Pressure (Test subset)')
    plt.tight_layout()
    plt.savefig(output_dir / 'predicted_vs_true.png', dpi=200)
    plt.close()


def plot_error_histograms(y_true: np.ndarray, predictions: Dict[str, np.ndarray], output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    for name, y_pred in predictions.items():
        errors = np.abs(y_pred - y_true)
        plt.hist(errors, bins=80, alpha=0.5, label=name)
    plt.xlabel('Absolute error (bar)')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Absolute Error Histogram (Test set)')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_histograms.png', dpi=200)
    plt.close()


def plot_sample_error_over_time(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    test_info: Dict[str, np.ndarray],
    output_dir: Path,
):
    output_dir.mkdir(exist_ok=True)
    n_cells = 10116
    y_true_samples = y_true.reshape(-1, n_cells)
    fig, ax = plt.subplots(figsize=(10, 5))
    sample_errors = {}
    for name, y_pred in predictions.items():
        y_pred_samples = y_pred.reshape(-1, n_cells)
        sample_rmse = np.sqrt(np.mean((y_pred_samples - y_true_samples) ** 2, axis=1))
        sample_errors[name] = sample_rmse
        ax.plot(sample_rmse, label=name)
    ax.set_xlabel('Test sample index')
    ax.set_ylabel('Sample RMSE (bar)')
    ax.set_title('Test-sample RMSE over timesteps')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_rmse_over_time.png', dpi=200)
    plt.close()

    # phase-specific lines for injection/production status
    fig, ax = plt.subplots(figsize=(10, 5))
    for phase_name, mask in [
        ('inj_on', test_info['inj_status'] == 1),
        ('prod_on', test_info['prod_status'] == 1),
        ('both_off', (test_info['inj_status'] == 0) & (test_info['prod_status'] == 0)),
    ]:
        if np.any(mask):
            ax.scatter(np.where(mask)[0], test_info['timestep'][mask], s=10, label=phase_name)
    ax.set_xlabel('Test sample index')
    ax.set_ylabel('Original timestep')
    ax.set_title('Test sample phase labeling')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'test_sample_phase_labels.png', dpi=200)
    plt.close()


def plot_spatial_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_coords: Tuple[np.ndarray, np.ndarray],
    output_dir: Path,
):
    output_dir.mkdir(exist_ok=True)
    n_cells = 10116
    y_true_samples = y_true.reshape(-1, n_cells)
    y_pred_samples = y_pred.reshape(-1, n_cells)
    mean_error = np.mean(np.abs(y_pred_samples - y_true_samples), axis=0)
    X_coords, Z_coords = raw_coords

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X_coords, Z_coords, c=mean_error, cmap='viridis', s=12, alpha=0.8)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Z coordinate')
    ax.set_title('Mean Absolute Error by Cell (Test set)')
    fig.colorbar(sc, ax=ax, label='Mean abs error (bar)')
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_error_distribution.png', dpi=200)
    plt.close()


def plot_pressure_evolution_examples(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_dir: Path,
    n_examples: int = 3,
):
    output_dir.mkdir(exist_ok=True)
    n_cells = 10116
    y_true_samples = y_true.reshape(-1, n_cells)
    y_pred_samples = {name: pred.reshape(-1, n_cells) for name, pred in predictions.items()}
    sample_rmse = np.sqrt(np.mean((y_pred_samples[next(iter(predictions))] - y_true_samples) ** 2, axis=1))
    selected = np.argsort(-sample_rmse)[:n_examples]

    for sample_idx in selected:
        plt.figure(figsize=(10, 5))
        plt.plot(y_true_samples[sample_idx], label='True', alpha=0.8)
        for name, y_pred in y_pred_samples.items():
            plt.plot(y_pred[sample_idx], label=f'Predicted {name}', alpha=0.8)
        plt.xlabel('Cell index')
        plt.ylabel('Pressure (bar)')
        plt.title(f'Pressure evolution example: test sample {sample_idx}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'pressure_evolution_sample_{sample_idx}.png', dpi=200)
        plt.close()


def compute_cell_and_timestep_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_coords: Tuple[np.ndarray, np.ndarray],
) -> Dict[str, object]:
    n_cells = 10116
    y_true_samples = y_true.reshape(-1, n_cells)
    y_pred_samples = y_pred.reshape(-1, n_cells)
    abs_error = np.abs(y_pred_samples - y_true_samples)
    cell_mean_error = np.mean(abs_error, axis=0)
    timestep_mean_error = np.mean(abs_error, axis=1)
    X_coords, Z_coords = raw_coords

    top_cells = np.argsort(-cell_mean_error)[:10]
    top_cell_entries = []
    for cell_idx in top_cells:
        top_cell_entries.append({
            'cell_index': int(cell_idx),
            'mean_abs_error': float(cell_mean_error[cell_idx]),
            'X': float(X_coords[cell_idx]),
            'Z': float(Z_coords[cell_idx]),
        })

    top_timesteps = np.argsort(-timestep_mean_error)[:10]
    top_timestep_entries = []
    for sample_idx in top_timesteps:
        top_timestep_entries.append({
            'test_sample_index': int(sample_idx),
            'mean_abs_error': float(timestep_mean_error[sample_idx]),
        })

    return {
        'top_cells': top_cell_entries,
        'top_timesteps': top_timestep_entries,
        'cell_mean_error': cell_mean_error.tolist(),
        'timestep_mean_error': timestep_mean_error.tolist(),
    }


def load_raw_coords(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    case_path = next(data_dir.glob('Case_*.mat'))
    raw = loadmat(str(case_path))
    return np.squeeze(raw['X_coords']).astype(np.float32), np.squeeze(raw['Z_coords']).astype(np.float32)


def save_json(data: object, path: Path):
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def main():
    data_dir = Path('.')
    summary_path = data_dir / 'PHASE_2_BASELINE_SUMMARY.json'
    report_path = data_dir / 'PHASE_2_REPORT.md'

    print('Building dataset...')
    split = build_dataset(data_dir)
    print('Dataset shapes:')
    print('  X_train', split.X_train.shape)
    print('  X_val', split.X_val.shape)
    print('  X_test', split.X_test.shape)

    X_train_flat, y_train_flat = flatten_dataset(split.X_train, split.y_train)
    X_val_flat, y_val_flat = flatten_dataset(split.X_val, split.y_val)
    X_test_flat, y_test_flat = flatten_dataset(split.X_test, split.y_test)

    print('Flattened shapes:')
    print('  X_train_flat', X_train_flat.shape)

    print('Evaluating persistence baseline...')
    y_pred_persistence = persistence_baseline(X_test_flat.reshape(-1, 9))
    persistence_metrics = evaluate_predictions(y_test_flat, y_pred_persistence)

    print('Training linear regression...')
    linear_model = train_linear_model(X_train_flat, y_train_flat)
    y_pred_linear = linear_model.predict(X_test_flat)
    linear_metrics = evaluate_predictions(y_test_flat, y_pred_linear)

    print('Training MLP model...')
    mlp = train_mlp_model(X_train_flat, y_train_flat, X_val_flat, y_val_flat)
    y_pred_mlp = mlp.predict(X_test_flat)
    mlp_metrics = evaluate_predictions(y_test_flat, y_pred_mlp)

    print('Running feature ablation (linear regression)...')
    ablation_results = feature_ablation(X_train_flat, y_train_flat, X_test_flat, y_test_flat)

    print('Computing phase-specific performance...')
    phase_performance = {
        'persistence': sample_weights_by_phase(split.test_sample_info, y_test_flat, y_pred_persistence),
        'linear': sample_weights_by_phase(split.test_sample_info, y_test_flat, y_pred_linear),
        'mlp': sample_weights_by_phase(split.test_sample_info, y_test_flat, y_pred_mlp),
    }

    print('Computing cell/timestep failures...')
    raw_coords = load_raw_coords(data_dir)
    failure_analysis = {
        'persistence': compute_cell_and_timestep_errors(y_test_flat, y_pred_persistence, raw_coords),
        'linear': compute_cell_and_timestep_errors(y_test_flat, y_pred_linear, raw_coords),
        'mlp': compute_cell_and_timestep_errors(y_test_flat, y_pred_mlp, raw_coords),
    }

    print('Saving plots...')
    plot_preds = {
        'Persistence': y_pred_persistence,
        'Linear': y_pred_linear,
        'MLP': y_pred_mlp,
    }
    plot_predictions_vs_true(y_test_flat, plot_preds, PLOT_DIR)
    plot_error_histograms(y_test_flat, plot_preds, PLOT_DIR)
    plot_sample_error_over_time(y_test_flat, plot_preds, split.test_sample_info, PLOT_DIR)
    plot_spatial_error_distribution(y_test_flat, y_pred_mlp, raw_coords, PLOT_DIR)
    plot_pressure_evolution_examples(y_test_flat, plot_preds, PLOT_DIR)

    summary = {
        'dataset': {
            'train_samples': int(split.X_train.shape[0]),
            'val_samples': int(split.X_val.shape[0]),
            'test_samples': int(split.X_test.shape[0]),
            'cells_per_sample': int(split.X_train.shape[1]),
            'features': len(FEATURE_NAMES),
            'feature_names': FEATURE_NAMES,
        },
        'metrics': {
            'persistence': persistence_metrics,
            'linear': linear_metrics,
            'mlp': mlp_metrics,
        },
        'ablation': ablation_results,
        'phase_performance': phase_performance,
        'failure_analysis': failure_analysis,
    }

    save_json(summary, summary_path)
    print(f'Saved summary to {summary_path}')

    report_lines = [
        '# Phase 2 Baseline Report',
        '',
        '## Summary',
        '',
        'This report compares persistence, linear regression, and a small MLP using the native cell representation.',
        '',
        '## Dataset',
        '',
        f'- Train samples: {summary["dataset"]["train_samples"]}',
        f'- Validation samples: {summary["dataset"]["val_samples"]}',
        f'- Test samples: {summary["dataset"]["test_samples"]}',
        f'- Cells per sample: {summary["dataset"]["cells_per_sample"]}',
        f'- Features: {summary["dataset"]["features"]}',
        '',
        '## Baseline performance (test set)',
        '',
        '| Model | RMSE | MAE | Relative Error | R² |',
        '|---|---|---|---|---|',
    ]
    for model_name in ['persistence', 'linear', 'mlp']:
        m = summary['metrics'][model_name]
        report_lines.append(
            f'| {model_name.capitalize()} | {m["rmse"]:.4f} | {m["mae"]:.4f} | {m["relative_error"]:.4f} | {m["r2"]:.4f} |'
        )
    report_lines += [
        '',
        '## Feature ablation (linear regression)',
        '',
        '| Removed feature | RMSE | MAE | Relative Error | R² | ΔRMSE |',
        '|---|---|---|---|---|---|',
    ]
    base_rmse = summary['metrics']['linear']['rmse']
    for entry in summary['ablation']:
        m = entry['metrics']
        report_lines.append(
            f'| {entry["removed_feature"]} | {m["rmse"]:.4f} | {m["mae"]:.4f} | {m["relative_error"]:.4f} | {m["r2"]:.4f} | {m["rmse"] - base_rmse:.4f} |'
        )
    report_lines += [
        '',
        '## Key findings',
        '',
        f'- Persistence RMSE = {summary["metrics"]["persistence"]["rmse"]:.4f}',
        f'- Linear regression RMSE = {summary["metrics"]["linear"]["rmse"]:.4f}',
        f'- MLP RMSE = {summary["metrics"]["mlp"]["rmse"]:.4f}',
        '',
        '## Diagnostic outputs',
        '',
        f'- Plots saved to `{PLOT_DIR}`',
        f'- Summary JSON saved to `{summary_path}`',
    ]

    save_json({'report': '\n'.join(report_lines)}, data_dir / 'PHASE_2_REPORT_TEMP.json')
    with report_path.open('w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f'Saved report to {report_path}')


if __name__ == '__main__':
    main()
