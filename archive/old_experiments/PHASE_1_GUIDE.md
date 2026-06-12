# Phase 1: Pressure Surrogate Dataset Preprocessing

## Educational Guide to Scientific Machine Learning for UHS

This document explains **step-by-step** how to convert reservoir simulation data into a machine-learning-ready dataset for pressure prediction.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Why This Approach?](#why-this-approach)
3. [Data Transformation Logic](#data-transformation-logic)
4. [The Nine Input Features](#the-nine-input-features)
5. [Creating Training Pairs](#creating-training-pairs)
6. [Expected Tensor Shapes](#expected-tensor-shapes)
7. [Calculating Sample Counts](#calculating-sample-counts)
8. [Normalization: Why and How](#normalization-why-and-how)
9. [Train/Val/Test Splitting](#trainvaltest-splitting)
10. [Sanity Checks](#sanity-checks)
11. [What the Code Produces](#what-the-code-produces)
12. [Next Steps](#next-steps)

---

## The Big Picture

**Goal**: Build a neural network that predicts how pressure evolves in a reservoir during hydrogen injection.

**Problem**: The physics-based simulator is slow. We want a fast, learned approximation.

**Solution**: Train a neural network on simulation data to learn the pattern:
```
Given:   pressure at time t, rock properties, injection rates
Predict: pressure at time t+1
```

**Scale**: 
- 5 simulation cases
- 10,116 grid cells per case
- 146 time steps per case
- **Total: 725 training samples** (one per timestep transition)

---

## Why This Approach?

### The Simulator is Complex

A physics-based reservoir simulator solves these PDEs at every cell and every timestep:

- **Flow Equation**: How pressure changes due to injection/production
- **Mass Conservation**: Total mass is conserved
- **Constitutive Relations**: Rock properties affect flow

This takes **hours or days** to run 2 years of simulation.

### A Learned Surrogate is Fast

Once trained, a neural network can:
- Predict pressure for 10,116 cells in **milliseconds**
- Run 2-year simulations in **seconds**
- Enable optimization, uncertainty quantification, and real-time forecasting

### Machine Learning Requires Supervision

Neural networks learn from examples. We need:
- **Input X**: "Here's the state at time t"
- **Output Y**: "The pressure changes to this value at time t+1"

The simulator provides both! Each timestep transition is a training example.

---

## Data Transformation Logic

### Step 1: Raw Data Structure

The simulator outputs a pressure matrix:

```
P_matrix: [10,116 cells × 146 timesteps]
```

This is **one long sequence** of pressure snapshots.

We also have static properties (same for all time):
```
Perm_initial: [10,116 cells × 1]
Poro_initial: [10,116 cells × 1]
```

And time-varying controls:
```
inj_BHP:  [1 well × 146 timesteps]
prod_BHP: [1 well × 146 timesteps]
inj_H2:   [1 well × 146 timesteps]
prod_H2:  [1 well × 146 timesteps]
```

### Step 2: The Transformation

We convert the sequence into **supervised pairs**:

**Sample 0**: 
- Input: State at t=0 (pressure + properties + controls at day 0)
- Target: Pressure at t=1 (day 5)

**Sample 1**:
- Input: State at t=1 (pressure + properties + controls at day 5)
- Target: Pressure at t=2 (day 10)

...and so on...

**Sample 144**:
- Input: State at t=144 (pressure + properties + controls at day 720)
- Target: Pressure at t=145 (day 725)

**Total per case**: 145 samples (from 146 timesteps)

### Step 3: Why This Works

Each pair is **independent**. A neural network can:
1. Learn from any sample in any order
2. Compute gradients without worrying about time dependencies
3. Be trained with standard backpropagation

By using all 5 cases: **145 samples/case × 5 cases = 725 total training examples**

---

## The Nine Input Features

For each sample, we stack 9 features per cell:

| # | Feature | Type | Shape | Physical Meaning |
|---|---------|------|-------|-----------------|
| 1 | Perm_initial | static | [10116] | Rock permeability (how easily fluid flows) |
| 2 | Poro_initial | static | [10116] | Rock porosity (void space fraction) |
| 3 | P(t) | dynamic | [10116] | Current pressure at each cell |
| 4 | inj_BHP | control | [10116] (broadcast) | Injection well pressure (same for all cells) |
| 5 | prod_BHP | control | [10116] (broadcast) | Production well pressure |
| 6 | inj_H2 | control | [10116] (broadcast) | Hydrogen injection rate |
| 7 | prod_H2 | control | [10116] (broadcast) | Hydrogen production rate |
| 8 | dt_days | metadata | [10116] (broadcast) | Timestep size (5 days) |
| 9 | t_index | metadata | [10116] (broadcast) | Current time index (0–145) |

### Why These Features?

- **Perm & Poro**: Rock controls how pressure changes (permeability equation)
- **P(t)**: Current pressure determines how it changes next
- **Well Pressures**: Drive fluid flow (boundary conditions)
- **Injection/Production Rates**: Specify how much H₂ enters/exits
- **Time Index**: Helps model learn time-dependent effects (e.g., pressure buildup)

### Broadcasting: What Does That Mean?

Control signals like `inj_BHP` are scalars (one value per timestep for the well).

But we need one value per cell (10,116 values).

**Broadcasting** means: repeat the scalar value 10,116 times.

```python
# Raw control (scalar)
inj_BHP_t = 150.0

# After broadcasting (array)
inj_BHP_t_broadcast = [150.0, 150.0, 150.0, ..., 150.0]  # 10116 values
```

This makes sense physically: the well pressure is the same everywhere (approximately).

---

## Creating Training Pairs

### Pseudocode

```python
for each case:
    load Perm, Poro, P_matrix, controls
    
    for timestep t from 0 to 144:  # 145 pairs per case
        # Assemble input (state at time t)
        X_t = stack [
            Perm,
            Poro,
            P[:, t],
            broadcast(inj_BHP[t]),
            broadcast(prod_BHP[t]),
            broadcast(inj_H2[t]),
            broadcast(prod_H2[t]),
            broadcast(dt[t]),
            broadcast(t)
        ]  # Result: [10116, 9]
        
        # Extract target (pressure at time t+1)
        Y_t = P[:, t+1]  # Result: [10116]
        
        # Store pair
        X_samples.append(X_t)
        Y_samples.append(Y_t)

# Stack all samples
X_all = np.array(X_samples)  # [725, 10116, 9]
Y_all = np.array(Y_samples)  # [725, 10116]
```

### Why Stack Features This Way?

By stacking features as [cells, features], we:
- Keep spatial information intact (cell index is meaningful)
- Make data compatible with convolutional neural networks (if we reshape)
- Enable per-cell predictions (each cell predicted independently)

---

## Expected Tensor Shapes

### Step 1: Load Raw .mat Files

```
Perm_initial:     [10116, 1]
Poro_initial:     [10116, 1]
P_matrix:         [10116, 146]
inj_BHP:          [1, 146]
prod_BHP:         [1, 146]
inj_H2:           [1, 146]
prod_H2:          [1, 146]
dt_days:          [1, 146]
```

### Step 2: Extract Single Timestep t

```
P_t:              [10116]
inj_BHP_t:        scalar
prod_BHP_t:       scalar
inj_H2_t:         scalar
prod_H2_t:        scalar
```

### Step 3: Stack Features Per Cell

```
X_t:              [10116, 9]
```

Each row is one cell. Each column is one feature.

### Step 4: Extract Target

```
Y_t:              [10116]
```

Pressure at each cell at time t+1.

### Step 5: Stack All Samples

```
X_all:            [725, 10116, 9]
Y_all:            [725, 10116]
```

Interpretation:
- **X_all[i]**: The i-th sample (state at some time t)
- **X_all[i, j, :]**: Cell j in sample i (9 features stacked)
- **Y_all[i, j]**: Target pressure for cell j in sample i

### Step 6: After Train/Val/Test Split

```
X_train:          [580, 10116, 9]  (80% of samples)
Y_train:          [580, 10116]

X_val:            [73, 10116, 9]   (10% of samples)
Y_val:            [73, 10116]

X_test:           [72, 10116, 9]   (10% of samples)
Y_test:           [72, 10116]
```

### Step 7: After Normalization

```
X_train_norm:     [580, 10116, 9]  (same shape, normalized values)
Y_train_norm:     [580, 10116]

... same for val and test
```

Normalization doesn't change shapes, only values.

---

## Calculating Sample Counts

### Per-Case Calculation

```
Timesteps per case:      146
Training pairs per case: 146 - 1 = 145
(We need timestep t AND t+1, so 146 timesteps → 145 pairs)
```

### Total Across All Cases

```
Total samples = 145 pairs/case × 5 cases = 725 samples
```

### Split Strategy

**Option 1: Random Splitting**
- Training: 580 random samples from all cases (80%)
- Validation: 73 random samples (10%)
- Test: 72 random samples (10%)

**Problem**: Samples from the same case might be in both train and test.

**Option 2: Case-Based Splitting (What We Use)**
- Training: All samples from cases 1–4 = 145 × 4 = 580 samples (80%)
- Validation: First half of case 5 samples = 73 samples (10%)
- Test: Second half of case 5 samples = 72 samples (10%)

**Advantage**: Test set is completely independent. Model must generalize to a new case.

---

## Normalization: Why and How

### The Problem Without Normalization

Imagine features with very different ranges:

```
Permeability:     0.0001 to 417 (range: 417)
Porosity:         0.08 to 0.19 (range: 0.11)
Pressure:         50 to 212 (range: 162)
Injection rate:   0 to 8.5 (range: 8.5)
```

Neural networks compute:

```
y = f(w1*perm + w2*poro + w3*pressure + w4*inj_rate)
```

If permeability values are 0.0001–417, the weight w1 must be tiny to avoid huge outputs.
Meanwhile, w2 for porosity must be huge.

This causes **unstable training**: gradients vanish or explode.

### The Solution: Z-Score Normalization

For each feature, compute:

```
feature_normalized = (feature - mean) / std_dev
```

**Effect**: 
- All features centered at 0
- All features scaled to width ≈ 1
- Neural network weights all in reasonable range [−1, +1]

### Example: Normalizing Pressure

**Raw values**:
```
Min:     50.05 bar
Max:     212.16 bar
Mean:    121.09 bar
Std Dev: 42.64 bar
```

**Formula**:
```
P_normalized = (P_raw - 121.09) / 42.64
```

**After normalization**:
```
Min:     (50.05 - 121.09) / 42.64 = -1.665
Max:     (212.16 - 121.09) / 42.64 = +2.134
Mean:    0 (by construction)
Std Dev: 1 (by construction)
```

### Critical Rule: Prevent Data Leakage

**WRONG**:
```python
# Compute statistics from ALL data
mean = X_all.mean()
std = X_all.std()

# Split into train/val/test
# ... train on normalized data

# Problem: val/test statistics leaked into training!
```

**CORRECT**:
```python
# Split FIRST
X_train, X_val, X_test = split(X_all)

# Compute statistics from TRAINING set ONLY
mean = X_train.mean()
std = X_train.std()

# Apply same mean/std to all sets
X_train_norm = (X_train - mean) / std
X_val_norm = (X_val - mean) / std    # Use training mean/std!
X_test_norm = (X_test - mean) / std  # Use training mean/std!
```

Why? Because the test set represents "future unseen data". 
The model should not know anything about test statistics during training.

---

## Train/Val/Test Splitting

### What Do These Sets Mean?

**Training Set (80%)**:
- Data the model learns from
- Weights are updated to minimize training loss
- Must be large and diverse

**Validation Set (10%)**:
- Data used to tune hyperparameters
- Examples: learning rate, batch size, layer width
- Early stopping: stop training if validation loss increases
- Model never directly trains on these examples

**Test Set (10%)**:
- Final evaluation dataset (opened only at the very end)
- Simulates "production" data the model has never seen
- Reports true generalization performance

### Why Not Just Train/Test?

With only 725 samples, 20% test set = 145 samples.
But validation during training needs independent examples.

Without validation set, you can't prevent **overfitting**:
- Model memorizes training data instead of learning general patterns
- Test performance is worse than it should be

### Our Split Strategy: Case-Based

```
Training:   Cases 1-4, all timesteps
            = 145 samples/case × 4 cases = 580 samples

Validation: Case 5, first 73 timesteps
            = 73 samples

Test:       Case 5, last 72 timesteps
            = 72 samples
```

**Rationale**:
- Training on cases 1–4 teaches model the physics in diverse scenarios
- Validation/test on case 5 (completely different initial conditions) tests true generalization
- Temporal split (first vs. second half of case 5) ensures no data leakage between val and test

### Temporal Split Prevents Leakage

If we randomly mixed timesteps from case 5:
```
Training timestep 100: P(t=100) → P(t+1=101)
Test timestep 101:     P(t=101) → P(t+1=102)
```

The model could "cheat":
- "I saw P(t=101) in training! I know what comes next!"

By splitting temporally (first half vs. second half):
```
Validation: Timesteps 0–72
Test:       Timesteps 73–144
```

The model never sees P(t>72) during training/validation.

---

## Sanity Checks

### Check 1: No NaN/Inf Values

**Purpose**: Ensure all numbers are valid.

**Why it matters**: 
- NaN (Not a Number) propagates through training
- Inf (Infinity) breaks neural networks
- Often indicates bugs in data loading or preprocessing

**What we check**:
```python
assert not np.isnan(X).any()
assert not np.isinf(X).any()
assert not np.isnan(Y).any()
assert not np.isinf(Y).any()
```

### Check 2: Pressure in Physical Range

**Purpose**: Verify pressure values are realistic.

**Expected range**: 50–212 bar (from data exploration)

**Why it matters**: 
- Pressure below 0 or above 1000 suggests errors
- Typo in loading or unit conversion could cause this

**What we check**:
```python
assert Y.min() > 40 and Y.max() < 220
```

### Check 3: Feature Scaling

**Purpose**: Verify no feature dominates others.

**Why it matters**:
- If permeability is 0.0001–417 but porosity is 0.08–0.19
- Neural network must scale weights very differently
- Training becomes numerically unstable

**What we check**:
- Print min/max for each feature
- Look for extreme differences in magnitude
- If problematic, normalization fixes this

### Check 4: Temporal Alignment

**Purpose**: Verify X[i] timestep = Y[i] timestep - 1.

**Why it matters**:
- If pairs are misaligned, model learns wrong patterns
- Like training a forecaster where inputs are from tomorrow!

**What we check**:
```python
# In pair generation, we explicitly create pairs
for t in range(n_timesteps - 1):
    X_t = state_at_t
    Y_t = pressure_at_t_plus_1
```

By construction, this is guaranteed.

### Check 5: No Data Leakage

**Purpose**: Ensure train/val/test sets don't overlap.

**Why it matters**:
- If the same spatial cell and timestep appears in both train and test
- Test performance would be artificially high
- We can't trust generalization claims

**What we check**:
```python
# Case-based split guarantees this
# Training uses cases 1-4
# Test uses only case 5
# No overlap possible
```

---

## What the Code Produces

### File 1: `pressure_surrogate_dataset.npz`

A compressed NumPy archive containing 12 arrays:

```
X_train           [580, 10116, 9]     Raw training inputs
Y_train           [580, 10116]        Raw training targets

X_train_norm      [580, 10116, 9]     Normalized training inputs
Y_train_norm      [580, 10116]        Normalized training targets

X_val             [73, 10116, 9]      Raw validation inputs
Y_val             [73, 10116]         Raw validation targets

X_val_norm        [73, 10116, 9]      Normalized validation inputs
Y_val_norm        [73, 10116]         Normalized validation targets

X_test            [72, 10116, 9]      Raw test inputs
Y_test            [72, 10116]         Raw test targets

X_test_norm       [72, 10116, 9]      Normalized test inputs
Y_test_norm       [72, 10116]         Normalized test targets
```

### File 2: `normalization_stats.json`

Stores the normalization parameters:

```json
{
  "X_mean": [188.44, 0.1037, 121.09, ...],
  "X_std": [115.13, 0.0176, 42.64, ...],
  "Y_mean": 121.09,
  "Y_std": 42.64,
  "feature_names": ["Perm", "Poro", "P(t)", "inj_BHP", ...]
}
```

**Why save these?**
- To denormalize predictions later (convert back to physical units)
- To apply same normalization to new data in production
- For reproducibility and auditing

### File 3: `diagnostic_pressure_evolution.png`

Four histograms showing pressure distribution at:
- Timestep 0 (day 0)
- Timestep 50 (day 250)
- Timestep 100 (day 500)
- Timestep 145 (day 730)

**What to look for**:
- Pressure increases over time (injection ongoing)
- Distribution shape consistent (no sudden artifacts)
- Range stays within 50–212 bar

---

## Next Steps

### Phase 2: Neural Network Design

Now that we have a clean dataset, design a neural network:

```
Input Layer:       [10116, 9]
Hidden Layers:     (to be designed)
Output Layer:      [10116]
```

**Questions to answer**:
1. How many hidden layers?
2. How many neurons per layer?
3. What activation functions?
4. What loss function (MSE, MAE)?
5. What optimizer (Adam, SGD)?

### Phase 3: Training and Evaluation

```python
model = build_model()
history = model.fit(
    X_train_norm, Y_train_norm,
    validation_data=(X_val_norm, Y_val_norm),
    epochs=100,
    batch_size=32
)
```

### Phase 4: Testing and Deployment

```python
Y_pred = model.predict(X_test_norm)
rmse = np.sqrt(np.mean((Y_test_norm - Y_pred)**2))

# Denormalize to physical units
Y_pred_physical = Y_pred * Y_std + Y_mean
Y_test_physical = Y_test * Y_std + Y_mean

# Evaluate on real pressure values
error_bar = np.abs(Y_pred_physical - Y_test_physical).mean()
print(f"Average prediction error: {error_bar:.2f} bar")
```

---

## Summary: Why This Matters

This preprocessing pipeline transforms **raw simulation data** into a format that:

1. **Enables Learning**: Supervised pairs let neural networks learn patterns
2. **Ensures Stability**: Normalization prevents numerical issues
3. **Prevents Cheating**: Careful splitting avoids data leakage
4. **Guarantees Generalization**: Case-based split tests on truly unseen scenarios
5. **Maintains Reproducibility**: Saved statistics enable auditing and production deployment

By following these principles, we build a **robust surrogate model** that:
- Runs 1000× faster than physics-based simulator
- Maintains physical accuracy (pressure in correct range)
- Generalizes to new scenarios (cases 1–4 → case 5)
- Supports optimization and uncertainty quantification

---

## References

- **Data Format**: MATLAB .mat files via `scipy.io`
- **Array Manipulation**: NumPy
- **Normalization**: Z-score (standardization)
- **Splitting**: Stratified (case-based) rather than random
- **Validation**: Sanity checks prevent common mistakes

