"""
Phase 1: Pressure Surrogate Dataset Preprocessing

This script transforms raw reservoir simulation outputs into a machine-learning-ready dataset
for pressure prediction. It implements the complete preprocessing pipeline including:
  - Data loading and extraction
  - Training pair generation (X_t -> Y_t+1)
  - Normalization (z-score standardization)
  - Train/val/test splitting
  - Sanity checks and diagnostics
  - Visualization of pressure statistics

Educational Focus:
  - Each step is documented with clear explanations
  - Why choices are made is explained in comments
  - Expected shapes are printed at each stage
  - Validation checks prevent common data science mistakes
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass
from typing import Tuple, Dict, List

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

class DataLoader:
    """Load and validate raw .mat files."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.case_files = sorted(data_dir.glob("Case_*.mat"))
        
    def load_case(self, case_idx: int = 0) -> Dict:
        """
        Load a single case and extract required variables.
        
        Why .mat format?
        - MATLAB stores data in binary .mat format
        - scipy.io.loadmat reads these files
        - Returns a dictionary with field names as keys
        
        What we extract:
        - Static properties: Perm_initial, Poro_initial, X_coords, Z_coords
        - Dynamic state: P_matrix (pressure over space and time)
        - Controls: inj_BHP, prod_BHP, inj_H2, prod_H2, inj_status, prod_status, dt_days
        """
        case_file = self.case_files[case_idx]
        print(f"\n{'='*70}")
        print(f"Loading: {case_file.name}")
        print(f"{'='*70}")
        
        data_raw = sio.loadmat(str(case_file))
        
        # Remove MATLAB metadata (keys starting with '__')
        data = {k: v for k, v in data_raw.items() if not k.startswith('__')}
        
        # Extract variables needed for pressure surrogate
        required_vars = [
            'Perm_initial', 'Poro_initial', 'X_coords', 'Z_coords',
            'P_matrix', 'inj_BHP', 'prod_BHP', 'inj_H2', 'prod_H2',
            'inj_status', 'prod_status', 'dt_days'
        ]
        
        case_data = {}
        for var in required_vars:
            if var in data:
                case_data[var] = data[var].astype(np.float32)
            else:
                raise ValueError(f"Missing required variable: {var}")
        
        return case_data
    
    def load_all_cases(self) -> List[Dict]:
        """Load all 5 cases."""
        all_cases = []
        for i in range(len(self.case_files)):
            case = self.load_case(i)
            all_cases.append(case)
        return all_cases


# ============================================================================
# STEP 2: TRAINING PAIR GENERATION
# ============================================================================

@dataclass
class SampleStats:
    """Store statistics of generated samples."""
    n_spatial_cells: int
    n_timesteps: int
    n_time_pairs: int  # timesteps - 1
    total_samples: int
    input_features: int
    

class TrainingPairGenerator:
    """
    Convert time-series reservoir data into supervised learning samples.
    
    The Core Transformation:
    ========================
    Raw data: P_matrix[10116 cells, 146 timesteps]
    
    This is ONE time series with 146 snapshots of a 10116-cell grid.
    
    Goal: Create supervised pairs (X, Y) where:
      - X = reservoir state at time t
      - Y = pressure field at time t+1
    
    Result: 145 training samples (one less than 146 timesteps)
    
    Why this transformation?
    - ML models learn: given state X, predict output Y
    - Pairing consecutive timesteps creates labeled examples
    - Model learns to predict one step into the future
    - Can be rolled forward autoregressively for longer forecasts
    """
    
    def __init__(self):
        self.X_samples = []  # List of input samples
        self.Y_samples = []  # List of output samples
        self.case_boundaries = []  # Track which samples belong to which case
        
    def generate_pairs(self, case_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate all (X, Y) pairs from a single case.
        
        Expected Input Shapes:
        ----------------------
        Perm_initial: [10116, 1]          # Permeability per cell (static)
        Poro_initial: [10116, 1]          # Porosity per cell (static)
        P_matrix:     [10116, 146]        # Pressure over space & time
        inj_BHP:      [1, 146]            # Injection well pressure (time series)
        prod_BHP:     [1, 146]            # Production well pressure (time series)
        inj_H2:       [1, 146]            # Injection rate (time series)
        prod_H2:      [1, 146]            # Production rate (time series)
        dt_days:      [1, 146]            # Timestep duration (constant = 5 days)
        
        Output Shapes:
        ---------------
        X:            [n_samples, 10116, 9]     # Stack of input features
        Y:            [n_samples, 10116]        # Stack of pressure targets
        """
        
        # Extract variables
        perm = case_data['Perm_initial'].squeeze()  # [10116] or [10116, 1] -> [10116]
        poro = case_data['Poro_initial'].squeeze()  # [10116, 1] -> [10116]
        P = case_data['P_matrix']                   # [10116, 146]
        inj_bhp = case_data['inj_BHP'].squeeze()   # [1, 146] -> [146]
        prod_bhp = case_data['prod_BHP'].squeeze() # [1, 146] -> [146]
        inj_h2 = case_data['inj_H2'].squeeze()     # [1, 146] -> [146]
        prod_h2 = case_data['prod_H2'].squeeze()   # [1, 146] -> [146]
        dt = case_data['dt_days'].squeeze()        # [1, 146] -> [146]
        
        n_cells = perm.shape[0]
        n_timesteps = P.shape[1]
        
        print(f"  n_spatial_cells: {n_cells}")
        print(f"  n_timesteps: {n_timesteps}")
        print(f"  n_training_pairs: {n_timesteps - 1}")
        print(f"  input_features: 9 (Perm, Poro, P_t, 5 controls, dt)")
        
        # Generate pairs for timesteps 0 to n_timesteps-2
        # (each pair: (state at t, pressure at t+1))
        for t in range(n_timesteps - 1):
            # Input: reservoir state at time t
            # Stack all input features into shape [10116, 9]
            X_t = np.stack([
                perm,                      # Feature 1: static perm
                poro,                      # Feature 2: static poro
                P[:, t],                   # Feature 3: pressure at time t
                np.full(n_cells, inj_bhp[t]),  # Feature 4: injection BHP (broadcast)
                np.full(n_cells, prod_bhp[t]), # Feature 5: production BHP (broadcast)
                np.full(n_cells, inj_h2[t]),   # Feature 6: injection H2 (broadcast)
                np.full(n_cells, prod_h2[t]),  # Feature 7: production H2 (broadcast)
                np.full(n_cells, dt[t]),       # Feature 8: timestep duration (broadcast)
                np.full(n_cells, t),           # Feature 9: time index (normalized later)
            ], axis=1)  # Stack along feature dimension -> [10116, 9]
            
            # Target: pressure at time t+1
            Y_t = P[:, t + 1]  # Shape: [10116]
            
            self.X_samples.append(X_t)
            self.Y_samples.append(Y_t)
        
        # Convert lists to arrays
        X = np.array(self.X_samples)  # [n_samples, 10116, 9]
        Y = np.array(self.Y_samples)  # [n_samples, 10116]
        
        # Record this case's boundaries
        n_samples = len(self.X_samples)
        self.case_boundaries.append((len(self.X_samples), len(self.Y_samples)))
        
        return X, Y
    
    def get_all_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return all accumulated samples."""
        if len(self.X_samples) == 0:
            raise ValueError("No samples generated. Call generate_pairs() first.")
        
        X = np.array(self.X_samples)  # [total_samples, 10116, 9]
        Y = np.array(self.Y_samples)  # [total_samples, 10116]
        return X, Y


# ============================================================================
# STEP 3: NORMALIZATION
# ============================================================================

class Normalizer:
    """
    Normalize features using z-score standardization.
    
    Why normalize?
    ===============
    1. Faster training: Neural networks learn better with normalized inputs
    2. Numerical stability: Prevents overflow/underflow
    3. Fair feature scaling: Large-magnitude features don't dominate learning
    4. Transfer learning: Normalized models are more generalizable
    
    Z-Score Normalization (Standardization):
    =========================================
    x_norm = (x - mean) / std
    
    Result: x_norm has mean ≈ 0 and std ≈ 1
    
    Critical Rule - Data Leakage Prevention:
    ========================================
    - Compute mean and std from TRAINING SET ONLY
    - Apply same statistics to validation and test sets
    - Otherwise: test set statistics "leak" into training
    """
    
    def __init__(self):
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self.is_fitted = False
        
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        Compute normalization statistics from training data ONLY.
        
        Input shapes:
        - X_train: [n_train_samples, 10116, 9]
        - Y_train: [n_train_samples, 10116]
        """
        print(f"\n{'='*70}")
        print("FITTING NORMALIZATION STATISTICS")
        print(f"{'='*70}")
        
        # For X: compute mean/std across all samples and cells
        # Shape: [n_train_samples, 10116, 9] -> compute across samples & cells
        # Result: [9] (one mean/std per feature)
        
        # Reshape to [n_samples * n_cells, 9] for easier computation
        X_reshaped = X_train.reshape(-1, X_train.shape[-1])  # [n_samples*10116, 9]
        self.X_mean = np.mean(X_reshaped, axis=0)  # [9]
        self.X_std = np.std(X_reshaped, axis=0)    # [9]
        
        # For Y: compute mean/std across all samples and cells
        Y_reshaped = Y_train.reshape(-1)  # [n_samples*10116]
        self.Y_mean = np.mean(Y_reshaped)  # scalar
        self.Y_std = np.std(Y_reshaped)    # scalar
        
        self.is_fitted = True
        
        print(f"X_mean shape: {self.X_mean.shape}")
        print(f"X_std shape: {self.X_std.shape}")
        print(f"Y_mean: {self.Y_mean:.4f}, Y_std: {self.Y_std:.4f}")
        
        # Print feature-wise statistics
        feature_names = ['Perm', 'Poro', 'P(t)', 'inj_BHP', 'prod_BHP', 
                        'inj_H2', 'prod_H2', 'dt_days', 't_index']
        print("\nFeature-wise statistics (from training data):")
        print(f"{'Feature':<15} {'Mean':<12} {'Std':<12}")
        print("-" * 40)
        for i, name in enumerate(feature_names):
            print(f"{name:<15} {self.X_mean[i]:>11.4f} {self.X_std[i]:>11.4f}")
        
    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Normalize input features."""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        X_reshaped = X.reshape(-1, X.shape[-1])  # [n_samples*10116, 9]
        X_norm = (X_reshaped - self.X_mean) / (self.X_std + 1e-8)  # Avoid division by zero
        X_norm = X_norm.reshape(X.shape)  # Reshape back to [n_samples, 10116, 9]
        return X_norm
    
    def transform_Y(self, Y: np.ndarray) -> np.ndarray:
        """Normalize target pressure."""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        Y_reshaped = Y.reshape(-1)  # [n_samples*10116]
        Y_norm = (Y_reshaped - self.Y_mean) / (self.Y_std + 1e-8)
        Y_norm = Y_norm.reshape(Y.shape)  # [n_samples, 10116]
        return Y_norm
    
    def inverse_transform_Y(self, Y_norm: np.ndarray) -> np.ndarray:
        """Convert normalized pressure back to physical units."""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        Y_reshaped = Y_norm.reshape(-1)
        Y_original = Y_reshaped * self.Y_std + self.Y_mean
        Y_original = Y_original.reshape(Y_norm.shape)
        return Y_original
    
    def get_stats(self) -> Dict:
        """Return normalization statistics for logging."""
        return {
            'X_mean': self.X_mean.tolist() if self.X_mean is not None else None,
            'X_std': self.X_std.tolist() if self.X_std is not None else None,
            'Y_mean': float(self.Y_mean) if self.Y_mean is not None else None,
            'Y_std': float(self.Y_std) if self.Y_std is not None else None,
        }


# ============================================================================
# STEP 4: TRAIN/VAL/TEST SPLITTING
# ============================================================================

class DataSplitter:
    """
    Split dataset into training, validation, and test sets.
    
    Splitting Strategy:
    ===================
    We use a CASE-BASED split (not random sample-based):
    - Training: Cases 1-4 (all 145*4 = 580 samples)
    - Validation: Case 5, first half (73 samples)
    - Test: Case 5, second half (72 samples)
    
    Why case-based splitting?
    - Test set is truly independent (different simulation case)
    - Prevents overfitting to specific initial conditions
    - More realistic: train on some scenarios, test on new ones
    
    Why not random splitting?
    - Random split mixes samples from the same case
    - Could lead to data leakage (training on time t, testing on t-1)
    - Less rigorous for evaluating generalization
    """
    
    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
    
    def split(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """
        Split data based on case boundaries.
        
        Strategy:
        ---------
        With 5 cases of 145 samples each = 725 total samples:
        - Train: 580 samples (cases 1-4)
        - Val: 73 samples (case 5, first 73 samples)
        - Test: 72 samples (case 5, remaining samples)
        """
        n_samples = X.shape[0]
        samples_per_case = 145
        n_cases = 5
        
        print(f"\n{'='*70}")
        print("TRAIN/VAL/TEST SPLITTING")
        print(f"{'='*70}")
        print(f"Total samples: {n_samples}")
        print(f"Samples per case: {samples_per_case}")
        print(f"Number of cases: {n_cases}")
        
        # Calculate split points
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)
        n_test = n_samples - n_train - n_val
        
        print(f"\nSplit Strategy (Case-Based):")
        print(f"  Training:   samples 0-{n_train-1} ({n_train} samples, cases 1-4)")
        print(f"  Validation: samples {n_train}-{n_train+n_val-1} ({n_val} samples)")
        print(f"  Test:       samples {n_train+n_val}-{n_samples-1} ({n_test} samples)")
        
        X_train = X[:n_train]
        Y_train = Y[:n_train]
        
        X_val = X[n_train:n_train+n_val]
        Y_val = Y[n_train:n_train+n_val]
        
        X_test = X[n_train+n_val:]
        Y_test = Y[n_train+n_val:]
        
        print(f"\nOutput Shapes:")
        print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
        print(f"  X_val:   {X_val.shape}, Y_val:   {Y_val.shape}")
        print(f"  X_test:  {X_test.shape}, Y_test:  {Y_test.shape}")
        
        return {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_val': X_val,
            'Y_val': Y_val,
            'X_test': X_test,
            'Y_test': Y_test,
        }


# ============================================================================
# STEP 5: SANITY CHECKS
# ============================================================================

class SanityChecker:
    """
    Verify data integrity before training.
    
    Common Data Mistakes to Catch:
    ==============================
    1. Temporal misalignment: X and Y don't correspond to consecutive timesteps
    2. Future leakage: X contains information from Y's timestep
    3. NaN/Inf values: Missing or infinite data breaks training
    4. Split contamination: Same sample appears in train and test
    5. Pressure statistics: Values outside expected range indicate bugs
    6. Feature scaling: Some features dominate others (motivates normalization)
    """
    
    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
    
    def check_no_nans(self, X, Y, name: str = ""):
        """Check for missing values."""
        check_name = f"NaN/Inf Check {name}"
        
        if np.isnan(X).any():
            self.checks_failed.append(f"{check_name}: Found NaN in X")
            return False
        
        if np.isnan(Y).any():
            self.checks_failed.append(f"{check_name}: Found NaN in Y")
            return False
        
        if np.isinf(X).any():
            self.checks_failed.append(f"{check_name}: Found Inf in X")
            return False
        
        if np.isinf(Y).any():
            self.checks_failed.append(f"{check_name}: Found Inf in Y")
            return False
        
        self.checks_passed.append(f"{check_name}: ✓ No NaN/Inf values")
        return True
    
    def check_pressure_range(self, Y, name: str = ""):
        """Check pressure is within expected physical range."""
        check_name = f"Pressure Range Check {name}"
        
        p_min = np.min(Y)
        p_max = np.max(Y)
        p_mean = np.mean(Y)
        
        # Expected range from data analysis
        if p_min < 40 or p_max > 220:
            self.checks_failed.append(
                f"{check_name}: Pressure out of range [min={p_min:.2f}, max={p_max:.2f}]"
            )
            return False
        
        self.checks_passed.append(
            f"{check_name}: ✓ Pressure in expected range "
            f"[min={p_min:.2f}, max={p_max:.2f}, mean={p_mean:.2f}]"
        )
        return True
    
    def check_feature_scaling(self, X, name: str = ""):
        """Check that features have reasonable ranges."""
        check_name = f"Feature Scaling Check {name}"
        
        feature_names = ['Perm', 'Poro', 'P(t)', 'inj_BHP', 'prod_BHP', 
                        'inj_H2', 'prod_H2', 'dt_days', 't_index']
        
        # Reshape to [n_samples*n_cells, n_features]
        X_flat = X.reshape(-1, X.shape[-1])
        
        print(f"\n  {check_name}:")
        for i, name in enumerate(feature_names):
            f_min = np.min(X_flat[:, i])
            f_max = np.max(X_flat[:, i])
            f_range = f_max - f_min
            print(f"    {name:12s}: min={f_min:10.4f}, max={f_max:10.4f}, range={f_range:10.4f}")
        
        self.checks_passed.append(f"{check_name}: ✓ See ranges above")
        return True
    
    def report(self):
        """Print all checks."""
        print(f"\n{'='*70}")
        print("SANITY CHECKS REPORT")
        print(f"{'='*70}")
        
        print("\n✓ PASSED:")
        for check in self.checks_passed:
            print(f"  {check}")
        
        if self.checks_failed:
            print("\n✗ FAILED:")
            for check in self.checks_failed:
                print(f"  {check}")
            return False
        
        print("\n✓ ALL CHECKS PASSED")
        return True


# ============================================================================
# STEP 6: DIAGNOSTICS AND VISUALIZATION
# ============================================================================

class Visualizer:
    """Generate diagnostic plots."""
    
    @staticmethod
    def plot_pressure_evolution(case_data: Dict, case_idx: int):
        """Plot pressure at selected timesteps and cells."""
        P = case_data['P_matrix']
        n_cells, n_timesteps = P.shape
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Pressure Evolution - Case {case_idx+1}', fontsize=14, fontweight='bold')
        
        # Sample a few timesteps
        timesteps = [0, 50, 100, 145]
        
        for idx, (ax, t) in enumerate(zip(axes.flat, timesteps)):
            p_t = P[:, t]
            ax.hist(p_t, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Pressure (bar)')
            ax.set_ylabel('Number of cells')
            ax.set_title(f'Timestep {t} (Day {t*5})')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pressure_prediction_error(Y_test, Y_pred, case_name: str = ""):
        """Plot actual vs predicted pressure."""
        # Y_test, Y_pred: [n_samples, 10116]
        
        # Compute error per sample
        rmse_per_sample = np.sqrt(np.mean((Y_test - Y_pred)**2, axis=1))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Pressure Prediction Diagnostics {case_name}', fontsize=14, fontweight='bold')
        
        # Plot 1: RMSE per sample
        axes[0].plot(rmse_per_sample, marker='o', linestyle='-', markersize=4)
        axes[0].set_xlabel('Sample index')
        axes[0].set_ylabel('RMSE (bar)')
        axes[0].set_title('Prediction Error Over Samples')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Actual vs Predicted (scatter)
        Y_flat_actual = Y_test.reshape(-1)
        Y_flat_pred = Y_pred.reshape(-1)
        axes[1].scatter(Y_flat_actual, Y_flat_pred, alpha=0.1, s=1)
        axes[1].plot([Y_flat_actual.min(), Y_flat_actual.max()], 
                     [Y_flat_actual.min(), Y_flat_actual.max()], 
                     'r--', lw=2, label='Perfect prediction')
        axes[1].set_xlabel('Actual Pressure (bar)')
        axes[1].set_ylabel('Predicted Pressure (bar)')
        axes[1].set_title('Actual vs Predicted Pressure')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        error = Y_flat_pred - Y_flat_actual
        axes[2].hist(error, bins=100, edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Prediction Error (bar)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title(f'Error Distribution (Mean={error.mean():.4f}, Std={error.std():.4f})')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def main():
    """Execute complete preprocessing pipeline."""
    
    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    results_figures_dir = project_root / "results" / "figures"
    
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    results_figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PHASE 1: PRESSURE SURROGATE DATASET PREPROCESSING")
    print("="*70)
    
    # ========== STEP 1: LOAD DATA ==========
    print("\n[STEP 1] Loading reservoir simulation data...")
    loader = DataLoader(data_raw_dir)
    all_cases = loader.load_all_cases()
    
    # ========== STEP 2: GENERATE TRAINING PAIRS ==========
    print("\n[STEP 2] Generating training pairs (X_t -> Y_t+1)...")
    pair_generator = TrainingPairGenerator()
    
    for i, case in enumerate(all_cases):
        print(f"\n  Case {i+1}:")
        X_case, Y_case = pair_generator.generate_pairs(case)
    
    X_all, Y_all = pair_generator.get_all_samples()
    print(f"\n  Total shapes after stacking all cases:")
    print(f"    X_all: {X_all.shape}")
    print(f"    Y_all: {Y_all.shape}")
    
    # ========== STEP 3: SPLIT DATA ==========
    print("\n[STEP 3] Splitting into train/val/test...")
    splitter = DataSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    splits = splitter.split(X_all, Y_all)
    
    # ========== STEP 4: FIT NORMALIZATION ==========
    print("\n[STEP 4] Fitting normalization on training data...")
    normalizer = Normalizer()
    normalizer.fit(splits['X_train'], splits['Y_train'])
    
    # ========== STEP 5: APPLY NORMALIZATION ==========
    print("\n[STEP 5] Applying normalization to all splits...")
    splits['X_train_norm'] = normalizer.transform_X(splits['X_train'])
    splits['Y_train_norm'] = normalizer.transform_Y(splits['Y_train'])
    splits['X_val_norm'] = normalizer.transform_X(splits['X_val'])
    splits['Y_val_norm'] = normalizer.transform_Y(splits['Y_val'])
    splits['X_test_norm'] = normalizer.transform_X(splits['X_test'])
    splits['Y_test_norm'] = normalizer.transform_Y(splits['Y_test'])
    
    # ========== STEP 6: SANITY CHECKS ==========
    print("\n[STEP 6] Running sanity checks...")
    checker = SanityChecker()
    
    for split_name in ['train', 'val', 'test']:
        X = splits[f'X_{split_name}']
        Y = splits[f'Y_{split_name}']
        checker.check_no_nans(X, Y, f"({split_name})")
        checker.check_pressure_range(Y, f"({split_name})")
        checker.check_feature_scaling(X, f"({split_name})")
    
    checker.report()
    
    # ========== STEP 7: SAVE DATASET ==========
    print("\n[STEP 7] Saving dataset...")
    dataset = {
        'X_train': splits['X_train'],
        'Y_train': splits['Y_train'],
        'X_val': splits['X_val'],
        'Y_val': splits['Y_val'],
        'X_test': splits['X_test'],
        'Y_test': splits['Y_test'],
        'X_train_norm': splits['X_train_norm'],
        'Y_train_norm': splits['Y_train_norm'],
        'X_val_norm': splits['X_val_norm'],
        'Y_val_norm': splits['Y_val_norm'],
        'X_test_norm': splits['X_test_norm'],
        'Y_test_norm': splits['Y_test_norm'],
    }
    
    np.savez_compressed(
        str(data_processed_dir / 'pressure_surrogate_dataset.npz'),
        **dataset
    )
    
    # Save normalization statistics
    norm_stats = normalizer.get_stats()
    norm_stats['feature_names'] = ['Perm', 'Poro', 'P(t)', 'inj_BHP', 'prod_BHP', 
                                   'inj_H2', 'prod_H2', 'dt_days', 't_index']
    
    with open(str(data_processed_dir / 'normalization_stats.json'), 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    print(f"  ✓ Saved to: data/processed/pressure_surrogate_dataset.npz")
    print(f"  ✓ Saved to: data/processed/normalization_stats.json")
    
    # ========== STEP 8: GENERATE DIAGNOSTICS ==========
    print("\n[STEP 8] Generating diagnostic plots...")
    visualizer = Visualizer()
    
    # Plot pressure evolution from first case
    fig = visualizer.plot_pressure_evolution(all_cases[0], case_idx=0)
    fig.savefig(str(results_figures_dir / 'diagnostic_pressure_evolution.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to: results/figures/diagnostic_pressure_evolution.png")
    
    plt.close('all')
    
    # ========== STEP 9: SUMMARY STATISTICS ==========
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE - SUMMARY")
    print("="*70)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples available: {X_all.shape[0]}")
    print(f"  Spatial cells: {X_all.shape[1]}")
    print(f"  Input features: {X_all.shape[2]}")
    print(f"\nTrain/Val/Test Split:")
    print(f"  Training samples:   {splits['X_train'].shape[0]} (80%)")
    print(f"  Validation samples: {splits['X_val'].shape[0]} (10%)")
    print(f"  Test samples:       {splits['X_test'].shape[0]} (10%)")
    print(f"\nOutput files:")
    print(f"  - pressure_surrogate_dataset.npz (raw and normalized)")
    print(f"  - normalization_stats.json (statistics for reproducibility)")
    print(f"  - diagnostic_pressure_evolution.png (visualization)")
    
    print(f"\nNext Steps:")
    print(f"  1. Load dataset: X, Y = np.load('pressure_surrogate_dataset.npz')")
    print(f"  2. Design neural network architecture")
    print(f"  3. Train model on normalized data")
    print(f"  4. Evaluate on test set")
    print(f"  5. Denormalize predictions for physical interpretation")


if __name__ == "__main__":
    main()
