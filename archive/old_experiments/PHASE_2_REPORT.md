# Phase 2 Baseline Report

## Summary

This report compares persistence, linear regression, and a small MLP using the native cell representation.

## Dataset

- Train samples: 520
- Validation samples: 60
- Test samples: 145
- Cells per sample: 10116
- Features: 9

## Baseline performance (test set)

| Model | RMSE | MAE | Relative Error | R² |
|---|---|---|---|---|
| Persistence | 4.2180 | 2.0068 | 0.0198 | 0.9931 |
| Linear | 3.5266 | 1.3307 | 0.0114 | 0.9952 |
| Mlp | 3.4227 | 1.2009 | 0.0119 | 0.9955 |

## Feature ablation (linear regression)

| Removed feature | RMSE | MAE | Relative Error | R² | ΔRMSE |
|---|---|---|---|---|---|
| Perm_initial | 3.5331 | 1.3322 | 0.0113 | 0.9952 | 0.0065 |
| Poro_initial | 3.5268 | 1.3299 | 0.0113 | 0.9952 | 0.0002 |
| P(t) | 28.7422 | 23.2292 | 0.2197 | 0.6807 | 25.2157 |
| inj_BHP | 3.5532 | 1.3859 | 0.0116 | 0.9951 | 0.0267 |
| prod_BHP | 3.5448 | 1.4042 | 0.0119 | 0.9951 | 0.0182 |
| inj_H2 | 3.5271 | 1.3298 | 0.0113 | 0.9952 | 0.0006 |
| prod_H2 | 3.6582 | 1.6817 | 0.0158 | 0.9948 | 0.1316 |
| inj_status | 3.5272 | 1.3298 | 0.0114 | 0.9952 | 0.0006 |
| prod_status | 3.5342 | 1.3197 | 0.0111 | 0.9952 | 0.0077 |

## Key findings

- Persistence RMSE = 4.2180
- Linear regression RMSE = 3.5266
- MLP RMSE = 3.4227

## Diagnostic outputs

- Plots saved to `phase2_plots`
- Summary JSON saved to `PHASE_2_BASELINE_SUMMARY.json`