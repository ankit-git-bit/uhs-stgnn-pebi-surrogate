# Phase 1.5 Validation Report

## Summary

This validation study analyzes the Phase 1 dataset representation, feature relevance, and spatial topology for the UHS pressure surrogate.

Key findings:
- `dt_days` is constant at `5` for all timesteps and carries no predictive signal.
- `inj_status` and `prod_status` are binary control flags that encode operating phases more directly than `t_index`.
- The mesh is irregular: 10,116 cells produced by 403 unique `X` coordinates and 346 unique `Z` coordinates.
- A structured grid projection is possible, but coarse grids introduce substantial interpolation error.

---

## Dataset and Sample Structure

- Training dataset shape: `X_train (580, 10116, 9)`, `Y_train (580, 10116)`
- Validation/test splits preserved the original Phase 1 pipeline shapes.
- Input feature stack:
  1. `Perm_initial`
  2. `Poro_initial`
  3. `P(t)`
  4. `inj_BHP`
  5. `prod_BHP`
  6. `inj_H2`
  7. `prod_H2`
  8. `dt_days`
  9. `t_index`

---

## Control Schedule and Time Encoding

From raw case data:
- `dt_days` unique values: `[5]`
- `inj_status` values: `[0, 1]`
- `prod_status` values: `[0, 1]`
- `inj_status` is active for 60 of 146 steps
- `prod_status` is active for 60 of 146 steps
- `both_on` = `0`
- `both_off` = `26`

Schedule transitions in Case 0001:
- `inj_status` changes at timesteps: `[30, 73, 103]`
- `prod_status` changes at timesteps: `[43, 73, 116]`

Interpretation:
- `dt_days` is redundant for this dataset and should be dropped unless variable timestep sizes are introduced.
- `t_index` encodes time, but not explicit operating phase; it may indirectly carry phase information only because the schedule is deterministic.
- Explicit flags `inj_status` and `prod_status` are recommended for robust phase-aware modeling.

---

## Feature Relevance

### Correlation with next-step pressure (`Y`)

- `P(t)`: very high correlation (`≈ 0.997`)
- `prod_BHP`: moderate positive correlation (`≈ 0.309`)
- `inj_H2`: moderate positive correlation (`≈ 0.269`)
- `prod_H2`: moderate negative correlation (`≈ -0.259`)
- `t_index`: moderate negative correlation (`≈ -0.650`)
- `Perm_initial` / `Poro_initial`: weak negative correlations (`≈ -0.17` / `-0.13`)
- `dt_days`: undefined correlation because it is constant

### Correlation with pressure change `ΔP = P(t+1) - P(t)`

- `inj_H2`: positive correlation (`≈ 0.437`)
- `prod_H2`: negative correlation (`≈ -0.447`)
- `prod_BHP`: positive correlation (`≈ 0.043`)
- `inj_BHP`: negative correlation (`≈ -0.069`)
- `t_index`: weak negative correlation (`≈ -0.076`)
- `Perm_initial` / `Poro_initial`: almost zero correlation

### Mutual information (sampled)

With `Y`:
- `P(t)`: highest signal
- `inj_BHP`, `prod_BHP`, `t_index`: significant
- `prod_H2` and `inj_H2`: moderate
- `dt_days`: minimal

With `ΔP`:
- `P(t)`: still high, but less dominant
- `inj_BHP`, `prod_BHP`, `prod_H2`, `t_index`: meaningful
- `inj_H2`: moderate
- `dt_days`: effectively zero

### Ablation (linear Ridge model on `Y`)

- Base RMSE: `3.543`
- Removing `P(t)` increases RMSE dramatically to `30.078`
- Removing `prod_H2` increases RMSE by `0.157`
- Removing `inj_H2` increases RMSE by `0.042`
- Removing `prod_BHP` increases RMSE by `0.017`
- Removing `t_index` increases RMSE by `0.017`
- Removing `Perm_initial`, `inj_BHP`, or `Poro_initial` has negligible effect
- Removing `dt_days` has no effect

Implication:
- `P(t)` is essential for next-step pressure prediction and dominates the prediction problem.
- Dynamic controls (`prod_H2`, `inj_H2`, `prod_BHP`) are the most important non-state predictors of pressure evolution.
- Static properties (`Perm_initial`, `Poro_initial`) provide weaker but nonzero supporting information.

---

## Spatial Topology Analysis

From raw mesh coordinates:
- `n_cells = 10,116`
- `n_unique_X = 403`
- `n_unique_Z = 346`
- `X` range: `12.5` to `4987.5`
- `Z` range: `0.25` to `99.75`
- Average number of `Z` positions per `X`: `≈ 25`
- Average number of `X` positions per `Z`: `≈ 29`

Mesh irregularity:
- Many `X` values appear only once
- Many `Z` values appear only once
- This is not a dense rectangular grid
- Direct 2D CNN input on a regular grid would require interpolation or masking

---

## Structured-Grid Projection Error

Interpolating a single pressure field through regular grids and reconstructing back to original cell coordinates produces:

- `64×32` grid: RMSE `0.65` to `11.39`
- `128×64` grid: RMSE `0.63` to `11.43`
- `256×128` grid: RMSE `0.46` to `8.59`

Observations:
- Low-gradient snapshots can be approximated well by coarse grids.
- Stronger pressure gradients produce large errors, even at `256×128`.
- Because the raw mesh has 10,116 unique cells, 2,048 grid points (`64×32`) compress the field dramatically.

Recommendation:
- A structured grid representation is feasible only with careful interpolation and error-aware modeling.
- For Phase 1.5, prefer point-cloud, graph, or mesh-aware models over naive 2D dense-grid CNNs.

---

## Recommendations

1. Remove `dt_days` from the Phase 1 model input unless variable timestep lengths are introduced.
2. Add explicit `inj_status` and `prod_status` flags to the feature set.
3. Keep `P(t)` as a core predictor because it carries most of the next-step pressure signal.
4. Preserve dynamic controls (`inj_H2`, `prod_H2`, `prod_BHP`), especially for predicting pressure change.
5. Treat `Perm_initial` and `Poro_initial` as secondary spatial features rather than primary predictors.
6. Avoid treating the raw mesh as a regular dense grid without interpolation error analysis.

---

## Next Steps

- Update the Phase 1 preprocessing pipeline to include explicit status flags and drop constant timestep metadata.
- If a regular grid model is desired, implement a mask-aware or interpolation-aware representation and validate on multiple snapshots.
- Continue Phase 1.5 by testing model architectures that can use irregular spatial structure, such as graph neural networks or point-wise MLPs with spatial coordinates.
