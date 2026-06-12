# Figures Manifest: UHS Pressure Surrogate Modeling

This document lists all 11 required figures for the modular paper project, defining their filenames, captions, paper section references, and expected source data.

---

### Fig. 1: Dataset Overview
* **Filename**: `figures/fig1_dataset_overview.png`
* **Caption**: *Figure 1: Overview of the Underground Hydrogen Storage (UHS) dataset, showing typical cyclic well operating schedules (injection/production BHP and flow rates) and the resulting average reservoir pressure over a 730-day (2-year) timeline.*
* **Section Reference**: Section 3 (`dataset.tex`)
* **Expected Source Data**: `Case_0001_wj.mat` BHP and flow rate time series.

### Fig. 2: Mesh Topology
* **Filename**: `figures/fig2_mesh_topology.png`
* **Caption**: *Figure 2: Representation of the native irregular PEBI (Perpendicular Bisection) / Voronoi spatial grid, displaying the non-uniform cell centroid distribution, cell interfaces, and the locations of the injection and production wells.*
* **Section Reference**: Section 3 (`dataset.tex`)
* **Expected Source Data**: `mesh_graph.npz` cell centroids coordinates ($X$, $Z$) and edge connectivity list.

### Fig. 3: Interpolation Error Study
* **Filename**: `figures/fig3_interpolation_error.png`
* **Caption**: *Figure 3: Spatial interpolation and reconstruction error of a single reservoir pressure snapshot projected onto regular grids of varying resolution ($64 \times 32$, $128 \times 64$, and $256 \times 128$) and mapped back to native coordinates.*
* **Section Reference**: Section 7 (`graph_surrogate.tex`)
* **Expected Source Data**: `PHASE_1_5_VALIDATION_SUMMARY.json` reconstruction RMSE series.

### Fig. 4: Baseline Model Comparison
* **Filename**: `figures/fig4_baseline_comparison.png`
* **Caption**: *Figure 4: Test set evaluation metrics (RMSE and $R^2$) for the baseline direct pressure prediction ($P_{t+1}$) compared to the pressure-change prediction ($\Delta P$) formulation across Persistence, Linear Regression, and MLP models.*
* **Section Reference**: Section 4 (`baseline_models.tex`) / Section 5 (`pressure_change.tex`)
* **Expected Source Data**: `PHASE_2_BASELINE_SUMMARY.json` and `PHASE_3_PRESSURE_CHANGE_SUMMARY.json` test set results.

### Fig. 5: Feature Importance Ranking
* **Filename**: `figures/fig5_feature_importance.png`
* **Caption**: *Figure 5: Random Forest feature importance rankings for the baseline, spatial, and spatiotemporal feature sets. The addition of temporal lag features ($\Delta P_{t-1}$) dominates the predictive signal (64.65% importance).*
* **Section Reference**: Section 6 (`feature_engineering.tex`)
* **Expected Source Data**: `PHASE_4_FEATURE_ENGINEERING_SUMMARY.json` Random Forest Gini importances.

### Fig. 6: Pressure Evolution Examples
* **Filename**: `figures/fig6_pressure_evolution.png`
* **Caption**: *Figure 6: Spatial pressure distribution snapshots at selected timesteps (during injection, production, and shut-in phases) showing the dynamic expansion of pressure buildup around the injector and drawdown cones around the producer.*
* **Section Reference**: Section 3 (`dataset.tex`)
* **Expected Source Data**: `Case_0001_wj.mat` spatial-temporal pressure field snapshots.

### Fig. 7: ST-GNN Architecture Diagram
* **Filename**: `figures/fig7_stgnn_architecture.png`
* **Caption**: *Figure 7: Schematic of the Spatio-Temporal Graph Neural Network (ST-GNN) architecture, showcasing the transmissibility-weighted graph convolution layers, residual connections, LayerNorm, and the Gated Recurrent Unit (GRU) for temporal state propagation.*
* **Section Reference**: Section 8 (`stgnn.tex`)
* **Expected Source Data**: Conceptual diagram block layout (Visio / Inkscape schematic).

### Fig. 8: Training Loss Curves
* **Filename**: `figures/fig8_training_loss.png`
* **Caption**: *Figure 8: Training and validation loss curves (weighted MSE) over epochs for the ST-GNN v1 (12 epochs) and v2 (30 epochs) models, demonstrating convergence trends and the best checkpoint identification at the final epochs.*
* **Section Reference**: Section 9 (`rollout_results.tex`)
* **Expected Source Data**: `st_gnn_v2_training_history.json` validation and training loss history logs.

### Fig. 9: Rollout RMSE versus Timestep
* **Filename**: `figures/fig9_rollout_rmse.png`
* **Caption**: *Figure 9: Autoregressive rollout RMSE over 142 timesteps on independent Test Case 5 for ST-GNN v1 and v2, highlighting the growth of pressure and saturation errors and the stabilizing effect of scheduled sampling in v2.*
* **Section Reference**: Section 9 (`rollout_results.tex`)
* **Expected Source Data**: `v3_evaluation_results.json` per-timestep pressure and saturation RMSE lists.

### Fig. 10: Pressure Tracking Plots
* **Filename**: `figures/fig10_pressure_tracking.png`
* **Caption**: *Figure 10: Pressure tracking comparisons between numerical reservoir simulator ground truth and ST-GNN predictions at specific grid locations (near the injector, near the producer, and in the far-field) over the 142-step rollout.*
* **Section Reference**: Section 9 (`rollout_results.tex`)
* **Expected Source Data**: `Case_0005_wj.mat` simulator ground truth and GNN rollout predictions.

### Fig. 11: v1 versus v2 Comparison
* **Filename**: `figures/fig11_v1_v2_comparison.png`
* **Caption**: *Figure 11: Comparative summary of multi-step rollout performance (at steps 1, 10, 50, and 146) for ST-GNN v1 and v2, highlighting the 31.6% reduction in final-step pressure RMSE achieved by v2's structural improvements.*
* **Section Reference**: Section 9 (`rollout_results.tex`)
* **Expected Source Data**: `v3_evaluation_results.json` discrete-step pressure and saturation RMSE values.
