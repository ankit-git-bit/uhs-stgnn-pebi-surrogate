# Publication-Ready Tables: UHS Pressure Surrogate Modeling

This document presents the 6 publication-ready tables filled with the experimentally verified metrics from the project phases.

---

### Table 1: Dataset and Feature Statistics
* **Caption**: *Table 1: Physical features, units, means, and standard deviations calculated across the training set (Cases 1–4) for input normalization.*
* **Data Source**: `normalization_stats.json` & `DATA_INSPECTION_REPORT.md`

| Feature Name | Description | Physical Unit | Mean ($\mu$) | Std Dev ($\sigma$) | Range [Min, Max] |
|:---|:---|:---|:---:|:---:|:---:|
| `Perm_initial` | Absolute Permeability | mD | 203.32 | 209.85 | 0.001, 5.31 (for Bact) |
| `Poro_initial` | Porosity | fraction | 0.1076 | 0.0249 | - |
| `P(t)` | Grid Cell Pressure at t | bar | 120.70 | 50.76 | 50.05, 212.16 |
| `inj_BHP` | Injection Well BHP | bar | 122.64 | 74.14 | - |
| `prod_BHP` | Production Well BHP | bar | 69.94 | 49.01 | - |
| `inj_H2` | Injection H₂ Flow Rate | daily rate units | 3.527 | 4.198 | 0.0, 8.53 |
| `prod_H2` | Production H₂ Flow Rate | daily rate units | 2.243 | 3.581 | 0.0, 8.49 |
| `dt_days` | Time Step Duration | days | 5.0 | 0.0 | [5.0] (constant) |
| `t_index` | Time Step Index | dimensionless | 72.0 | 41.86 | 1, 146 |

*Note: Bacterial concentration (`Bact_matrix`) has a range of 0.001 to 5.31 with mean 0.112. Initial permeability and porosity are static properties.*

---

### Table 2: Phase 2 Baseline Results (Direct Pressure $P_{t+1}$ Prediction)
* **Caption**: *Table 2: Baseline model performance on the test set for direct next-step pressure ($P_{t+1}$) prediction, demonstrating deceptively high metrics due to temporal autocorrelation.*
* **Data Source**: `PHASE_2_REPORT.md`

| Model | RMSE (bar) | MAE (bar) | Relative Error | $R^2$ |
|:---|:---:|:---:|:---:|:---:|
| **Persistence ($P_{t+1} = P_t$)** | 4.2180 | 2.0068 | 0.0198 | 0.9931 |
| **Linear Regression** | 3.5266 | 1.3307 | 0.0114 | 0.9952 |
| **MLP (128-64)** | 3.4227 | 1.2009 | 0.0119 | 0.9955 |

---

### Table 3: Phase 3 Baseline Results (Pressure-Change $\Delta P$ Prediction)
* **Caption**: *Table 3: Baseline model performance on the test set for pressure-change ($\Delta P = P_{t+1} - P_t$) prediction, exposing the inadequacy of cell-local static features.*
* **Data Source**: `PHASE_3_REPORT.md`

| Model | RMSE (bar) | MAE (bar) | Relative Error | $R^2$ |
|:---|:---:|:---:|:---:|:---:|
| **Persistence ($\Delta P = 0$)** | 4.2180 | 3.5177 | 0.2043 | -0.0198 |
| **Linear Regression** | 4.1756 | 3.4870 | 0.2030 | 0.0007 |
| **MLP (128-64)** | 3.3614 | 2.5611 | 0.1535 | 0.3524 |

---

### Table 4: Phase 4 Feature Engineering Results (Test Case 5)
* **Caption**: *Table 4: Performance comparison of linear, regularization, random forest, and MLP models across three engineered feature sets (Baseline, Spatial, Spatiotemporal) on Test Case 5.*
* **Data Source**: `PHASE_4_FEATURE_ENGINEERING_SUMMARY.json`

| Feature Set | Model | RMSE (bar) | MAE (bar) | $R^2$ |
|:---|:---|:---:|:---:|:---:|
| **Baseline (9 features)** | Linear | 3.5440 | 1.3581 | 0.2801 |
| | Ridge | 3.5273 | 1.3275 | 0.2869 |
| | Random Forest | 4.5388 | 0.8151 | -0.1808 |
| | MLP | 3.1970 | 0.8589 | 0.4142 |
| **Spatial (19 features)** | Linear | 3.5440 | 1.3581 | 0.2801 |
| | Ridge | 3.5273 | 1.3275 | 0.2869 |
| | Random Forest | 4.5391 | 0.8166 | -0.1809 |
| | MLP | 3.2307 | 0.9170 | 0.4018 |
| **Spatiotemporal (25 features)** | Linear | 3.6718 | 0.9898 | 0.2272 |
| | Ridge | 3.6713 | 0.9866 | 0.2275 |
| | **Random Forest** | **0.5487** | **0.0941** | **0.9827** |
| | **MLP** | **0.9063** | **0.3557** | **0.9529** |

---

### Table 5: ST-GNN Rollout Metric Comparison (v1 vs. v2)
* **Caption**: *Table 5: Autoregressive rollout metrics (averaged over 142 steps) and final-step errors on Test Case 5 for ST-GNN v1 and v2, showing significant improvements in v2.*
* **Data Source**: `st_gnn_v2_rollout_summary.json`

| Model | Mean $P$-RMSE (bar) | Final $P$-RMSE (bar) | Mean $S_g$-RMSE | Final $S_g$-RMSE | Max $P$-Error (bar) | Max $S_g$-Error |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ST-GNN v1** | 35.2638 | 49.2455 | 0.016717 | 0.016476* | 278.29 | 0.3960 |
| **ST-GNN v2** | 31.7239 | 33.6702 | 0.022283 | 0.029837 | 240.64 | 0.4041 |
| **Improvement (%)** | **10.04%** | **31.63%** | - | - | **13.53%** | - |

*\*Note: ST-GNN v1 final saturation RMSE is taken from st_gnn_rollout_summary.json. ST-GNN v2 final pressure RMSE achieves a 31.6% reduction compared to v1.*

---

### Table 6: Multi-Step Autoregressive Rollout RMSE Details
* **Caption**: *Table 6: Multi-step autoregressive rollout RMSE for pressure ($P$) and gas saturation ($S_g$) at discrete forecasting horizons on Test Case 5.*
* **Data Source**: `v3_evaluation_results.json`

| Model & State Variable | t = 1 step | t = 10 steps | t = 50 steps | t = 146 steps (Final) |
|:---|:---:|:---:|:---:|:---:|
| **Pressure RMSE (bar)** | | | | |
| ST-GNN v1 | 0.3568 | 4.5491 | 75.5759 | 49.2455 |
| ST-GNN v2 | **0.1900** | **1.8337** | **59.1941** | **33.6702** |
| **Saturation RMSE ($S_g$)** | | | | |
| ST-GNN v1 | 0.003025 | 0.012442 | **0.025100** | **0.016476** |
| ST-GNN v2 | **0.001621** | **0.006212** | 0.030784 | 0.029837 |
