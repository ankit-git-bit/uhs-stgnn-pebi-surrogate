# Phase 1 Complete: Pressure Surrogate Dataset Preprocessing

## 🎯 Project Summary

**Objective**: Transform raw Underground Hydrogen Storage (UHS) reservoir simulation data into a machine-learning-ready dataset for pressure prediction.

**Status**: ✅ **PHASE 1 COMPLETE**

---

## 📊 What Was Accomplished

### 1. **Data Inspection & Analysis** ✓
- Analyzed 5 reservoir simulation cases
- Identified 46 distinct physical properties
- Extracted key variables for pressure prediction
- Generated data statistics and value ranges

### 2. **Data Transformation** ✓
- Converted 146 timestep sequences into 145 supervised training pairs per case
- Created 9-feature input vectors per spatial cell
- **Total samples generated: 725** (145 pairs × 5 cases)
- Each sample: input state at time t → target pressure at time t+1

### 3. **Normalization** ✓
- Applied z-score standardization to all features
- Computed statistics from training set only (prevented data leakage)
- Output: All features normalized to mean≈0, std≈1
- Enabled numerically stable neural network training

### 4. **Train/Val/Test Splitting** ✓
- **Case-based splitting strategy** (not random):
  - Training: Cases 1–4 = 580 samples (80%)
  - Validation: Case 5, first half = 72 samples (10%)
  - Test: Case 5, second half = 73 samples (10%)
- Ensures test set is truly independent (different simulation case)

### 5. **Data Integrity Checks** ✓
- ✓ No NaN/Inf values
- ✓ Pressure values in physical range (50–212 bar)
- ✓ Features properly scaled
- ✓ Temporal alignment verified (t and t+1 paired correctly)
- ✓ No data leakage between splits

### 6. **Diagnostic Visualizations** ✓
- Pressure evolution histograms (4 timesteps)
- Feature distributions (9 input features)
- Target pressure distributions (train/val/test)
- Dataset overview (split sizes, statistics)

---

## 📁 Output Files Generated

### Core Dataset

| File | Size | Purpose |
|------|------|---------|
| `pressure_surrogate_dataset.npz` | ~450 MB | Complete dataset with 12 arrays (raw + normalized) |
| `normalization_stats.json` | ~1 KB | Normalization parameters for reproducibility |

### Python Scripts

| File | Purpose |
|------|---------|
| `pressure_surrogate_preprocessing.py` | Main pipeline (loads, processes, normalizes, splits) |
| `load_and_use_dataset.py` | Usage guide and data loading template |
| `generate_plots.py` | Diagnostic visualization generator |

### Documentation

| File | Content |
|------|---------|
| `PHASE_1_GUIDE.md` | **Educational guide** (detailed explanations of every step) |
| `data_analysis_report.tex` | LaTeX report (technical documentation) |
| `README_PHASE_1.md` | This file |

### Visualizations

| File | Shows |
|------|-------|
| `diagnostic_pressure_evolution.png` | Pressure histograms at 4 timesteps |
| `feature_distributions.png` | Input feature distributions |
| `target_distributions.png` | Pressure distributions per split |
| `dataset_overview.png` | Split sizes, statistics, boxplots |

---

## 🔢 Dataset Specifications

### Dimensions

```
Training:   X_train    [580, 10116, 9]    Y_train    [580, 10116]
Validation: X_val      [72,  10116, 9]    Y_val      [72,  10116]
Test:       X_test     [73,  10116, 9]    Y_test     [73,  10116]
Total:      725 samples × 10,116 cells × 9 features
```

### Features (9 per cell)

1. **Perm_initial** — Rock permeability (static)
2. **Poro_initial** — Rock porosity (static)
3. **P(t)** — Current pressure at each cell
4. **inj_BHP** — Injection well pressure (broadcast)
5. **prod_BHP** — Production well pressure (broadcast)
6. **inj_H2** — H₂ injection rate (broadcast)
7. **prod_H2** — H₂ production rate (broadcast)
8. **dt_days** — Timestep duration (5 days, broadcast)
9. **t_index** — Time index for temporal encoding

### Normalization Statistics

From training set (580 samples, 10116 cells):

| Feature | Mean | Std Dev | Min | Max |
|---------|------|---------|-----|-----|
| Perm | 188.44 | 115.13 | 0.0001 | 417.04 |
| Poro | 0.1037 | 0.0176 | 0.0800 | 0.1897 |
| P(t) | 121.09 | 42.64 | 50.05 | 212.16 |
| inj_BHP | 123.88 | 56.49 | 0 | 212.19 |
| prod_BHP | 24.73 | 31.90 | 0 | 75.33 |
| inj_H2 | 3.50 | 2.85 | 0 | 8.53 |
| prod_H2 | 2.31 | 2.41 | 0 | 8.49 |
| dt_days | 5.00 | 0.00 | 5 | 5 |
| t_index | 72.5 | 42.30 | 0 | 144 |

**Pressure target (Y)**: Mean = 121.09 bar, Std = 42.64 bar

---

## 💡 Key Design Decisions & Why

### 1. **Supervised Pair Generation**
- **Decision**: Convert 146 timesteps → 145 (X, Y) pairs per case
- **Why**: ML models learn from input-output examples; each timestep transition is a labeled example
- **Benefit**: 725 training samples from just 5 simulations; enables autoregressive prediction

### 2. **Nine Input Features**
- **Decision**: Include static (perm/poro), dynamic (pressure), and control signals
- **Why**: Physics-based prediction requires material properties, current state, and boundary conditions
- **Benefit**: Model learns complete physics of pressure evolution

### 3. **Z-Score Normalization**
- **Decision**: (x - mean) / std for all features
- **Why**: Neural networks train faster and more stably with normalized inputs
- **Benefit**: Prevents gradient vanishing/explosion; fair feature importance

### 4. **Fit on Training Set Only**
- **Decision**: Compute mean/std from training data; apply to validation/test
- **Why**: Prevents data leakage—test set information shouldn't influence training
- **Benefit**: Rigor; ensures true generalization test

### 5. **Case-Based Splitting**
- **Decision**: Train on cases 1–4; validate/test on case 5
- **Why**: Ensures test set comes from completely different initial conditions
- **Better than random split**: Prevents temporal leakage; tests generalization to new scenarios
- **Benefit**: More meaningful evaluation of surrogate model

### 6. **Sanity Checks**
- **Decision**: Verify NaN/Inf, pressure range, feature scaling, temporal alignment
- **Why**: Catches data loading bugs before training starts
- **Benefit**: Confidence in data quality; fails fast on problems

---

## 🚀 How to Use the Dataset

### Loading in Python

```python
import numpy as np
import json

# Load dataset
data = np.load('pressure_surrogate_dataset.npz')
X_train = data['X_train_norm']
Y_train = data['Y_train_norm']
X_val = data['X_val_norm']
Y_val = data['Y_val_norm']
X_test = data['X_test_norm']
Y_test = data['Y_test_norm']

# Load normalization stats for denormalization
with open('normalization_stats.json', 'r') as f:
    stats = json.load(f)
```

### Training with PyTorch

```python
import torch

X_train_torch = torch.from_numpy(X_train).float()
Y_train_torch = torch.from_numpy(Y_train).float()

# Build and train your model
model = YourPressureModel()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    for X_batch, Y_batch in train_loader:
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()
```

### Denormalizing Predictions

```python
# After model prediction
Y_pred_norm = model.predict(X_test_norm)

# Convert back to bar
Y_pred_physical = Y_pred_norm * stats['Y_std'] + stats['Y_mean']
print(f"Pressure range: {Y_pred_physical.min():.2f}–{Y_pred_physical.max():.2f} bar")
```

---

## 📈 Expected Performance Baseline

For reference, a baseline model should achieve:

- **Naive Persistence** (predict Y = X): RMSE ≈ 2–3 bar (pressure change per timestep)
- **Linear Regression**: RMSE ≈ 1–2 bar
- **Well-tuned Neural Network**: RMSE < 1 bar (target for Phase 2)

Your model should predict pressure changes within ±1 bar RMSE on the test set.

---

## 📖 Documentation Files

### For Understanding the Physics & ML

**Start here**: `PHASE_1_GUIDE.md`
- Educational guide with simple language
- Explains every step: why it's needed, what problem it solves
- Examples and diagrams

### For Technical Details

**Reference**: `data_analysis_report.tex` → compile to PDF
- LaTeX report with formal notation
- Statistical tables and formulas
- Complete methodology documentation

### For Using the Data

**Tutorial**: `load_and_use_dataset.py` + code examples
- How to load and inspect data
- PyTorch and Keras integration patterns
- Denormalization recipes

---

## 🔗 Phase 2: Next Steps (Neural Network Design)

The dataset is now ready. Phase 2 will focus on:

1. **Architecture Design**
   - How many hidden layers?
   - Neurons per layer?
   - Activation functions (ReLU, Tanh, etc.)?

2. **Training Configuration**
   - Learning rate schedule
   - Batch size
   - Early stopping criteria

3. **Model Options**
   - Fully connected (dense) layers
   - Convolutional layers (if using spatial structure)
   - Recurrent layers (if using temporal windows)
   - Graph neural networks (if leveraging mesh connectivity)

4. **Evaluation Metrics**
   - RMSE on test set
   - MAE per cell
   - Spatial error maps
   - Temporal stability (can predictions roll forward?)

---

## ✅ Verification Checklist

Before moving to Phase 2, confirm:

- [ ] `pressure_surrogate_dataset.npz` exists (~450 MB)
- [ ] `normalization_stats.json` exists with 4 numeric parameters
- [ ] Dataset shapes are (580, 10116, 9) for training X, etc.
- [ ] All 4 diagnostic PNG files exist
- [ ] `PHASE_1_GUIDE.md` is readable and makes sense
- [ ] `load_and_use_dataset.py` runs without errors
- [ ] Normalized data has mean≈0 and std≈1

---

## 📞 Quick Reference

| Task | File |
|------|------|
| Understand the approach | `PHASE_1_GUIDE.md` |
| Load data for training | `load_and_use_dataset.py` |
| View technical details | `data_analysis_report.tex` |
| See preprocessing code | `pressure_surrogate_preprocessing.py` |
| Generate new plots | `generate_plots.py` |

---

## 🎓 Learning Outcomes

By studying Phase 1, you should understand:

1. **Why data transformation matters**: Supervised learning requires input-output pairs
2. **How normalization prevents bugs**: Neural networks need scaled features
3. **Why data splitting is critical**: Prevents overfitting and data leakage
4. **The role of sanity checks**: Catches errors early, before wasting compute
5. **Case-based vs random splitting**: When and why each approach applies

This foundation will help you design better neural networks, debug training issues, and build robust ML systems.

---

**Generated**: June 2026  
**Status**: Ready for Phase 2 (Neural Network Training)  
**Questions?** See `PHASE_1_GUIDE.md` or review `pressure_surrogate_preprocessing.py` code comments
