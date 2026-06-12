# Reproducibility Validation Report

This report documents the validation of the Spatio-Temporal Graph Neural Networks (ST-GNN) surrogate framework for forecasting reservoir pressure and gas saturation. All checks were executed in a safe, non-destructive validation mode.

---

## 1. Environment & Setup Context

* **Operating System**: Windows
* **Python Interpreter**: Python 3.13 (in virtual environment `.venv`)
* **PyTorch Version**: PyTorch 2.12.0 (CPU execution)
* **PyTorch Geometric**: Not required (custom message passing stencils are implemented using native PyTorch sparse and dense tensor operations).

---

## 2. Validation Procedure (Safe Mode)

To ensure that running validation did not overwrite or modify any of the published paper assets, the following protocol was executed:
1. **Backup**: Copied the entire `results/` directory to `results_backup/`.
2. **Inference Run**: Ran the multi-step autoregressive rollout evaluation script:
   ```bash
   .venv\Scripts\python src/evaluation/evaluate_rollout_v3.py
   ```
3. **LaTeX Run**: Ran the modular LaTeX project validator:
   ```bash
   .venv\Scripts\python src/utils/validate_latex.py
   ```
4. **Restore**: Deleted the validation outputs and restored the original `results/` folder from `results_backup/`.

---

## 3. Reproduced GNN Rollout Metrics

The multi-step autoregressive rollout (142 forecasting intervals) on held-out **Case 5** yielded the following metrics:

### Pressure Rollout RMSE (psi)
| Steps | ST-GNN v1 (Base) | ST-GNN v2 (+LN +Residuals) | ST-GNN v3 (+Weighted Loss) |
| :--- | :---: | :---: | :---: |
| **1-step** | `0.3568` | `0.1900` | `0.1008` |
| **10-step** | `4.5491` | `1.8337` | `2.2427` |
| **50-step** | `75.5759` | `59.1941` | `72.4473` |
| **146-step** | `49.2455` | `33.6702` | `31.6625` |
| **Mean RMSE** | **`35.2638`** | **`31.7239`** | **`27.1501`** |
| **Max Error** | `278.2911` | `240.6428` | `187.6818` |

### Gas Saturation Rollout RMSE
| Steps | ST-GNN v1 (Base) | ST-GNN v2 (+LN +Residuals) | ST-GNN v3 (+Weighted Loss) |
| :--- | :---: | :---: | :---: |
| **1-step** | `0.003025` | `0.001621` | `0.000922` |
| **10-step** | `0.012442` | `0.006212` | `0.003486` |
| **50-step** | `0.025100` | `0.030784` | `0.028324` |
| **146-step** | `0.016476` | `0.029837` | `0.029365` |
| **Mean RMSE** | **`0.016717`** | **`0.022283`** | **`0.018108`** |
| **Max Error** | `0.395988` | `0.404093` | `0.425782` |

These results match the metrics reported in the manuscript exactly, confirming that the pre-trained checkpoints are fully reproducible.

---

## 4. LaTeX Project Integrity

The validator script scanned the monolithic and modular manuscript directories:
* **Unique Labels Found**: 38
* **References Checked**: 32 (all resolved successfully, 0 broken)
* **Citations Checked**: 22 (all resolved in `references.bib`, 0 missing)
* **Figures Checked**: 12 (all paths mapped correctly to files, 0 missing)
* **Input Files Checked**: 18 (all modular section files resolved, 0 missing)

**Result**: LaTeX project validation completed with 0 warnings or errors.
