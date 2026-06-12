# Version Freeze Metadata

This document freezes the final state of the manuscript, dataset, model metrics, and document counts for submission.

---

## 1. Submission Metadata

* **Date**: June 12, 2026
* **Paper Title**: Spatio-Temporal Graph Neural Networks for Pressure Forecasting in Underground Hydrogen Storage Reservoirs on Irregular PEBI Meshes
* **Dataset Version**: KTU UHS Irregular PEBI Mesh Dataset (10,116 active cells, 5 cases, 146 steps each, 5-day intervals)
* **Author List**: KTU Internship Project Team
* **Affiliation**: Department of Scientific Computing & Machine Learning

---

## 2. Frozen Scientific Metrics

### One-Step-Ahead Forecasting (Test Case 5, Teacher-Forcing)
* **Random Forest (Spatiotemporal, 25 features)**:
  * $R^2 = 0.9827$
  * RMSE = 0.5487 bar
  * MAE = 0.0941 bar
* **MLP (Spatiotemporal, 25 features)**:
  * $R^2 = 0.9529$
  * RMSE = 0.9063 bar
  * MAE = 0.3557 bar
* **Baseline MLP (Local, 9 features)**:
  * $R^2 = 0.4142$
  * RMSE = 3.1970 bar
* **Baseline Linear Regression (Local, 9 features)**:
  * $R^2 = 0.2801$
  * RMSE = 3.5440 bar

### Autoregressive Rollout Forecasting (Test Case 5, 142 Steps, Fully Autoregressive)
* **ST-GNN v2 (128 features, residual connections, LayerNorm, scheduled sampling)**:
  * Mean Pressure RMSE = 31.7239 bar (10.04% improvement over v1)
  * Final-Step Pressure RMSE = 33.6702 bar (31.63% improvement over v1)
  * Max Pressure Error = 240.64 bar
  * Mean Saturation RMSE = 0.022283
  * Final-Step Saturation RMSE = 0.029837
  * Max Saturation Error = 0.4041
* **ST-GNN v1 (64 features, teacher forcing baseline)**:
  * Mean Pressure RMSE = 35.2638 bar
  * Final-Step Pressure RMSE = 49.2455 bar
  * Max Pressure Error = 278.29 bar
  * Mean Saturation RMSE = 0.016717
  * Final-Step Saturation RMSE = 0.016476
  * Max Saturation Error = 0.3960
* **ST-GNN v3 (64 features, loss sweep winner)**:
  * Mean Pressure RMSE = 25.4943 bar
  * Final-Step Pressure RMSE = 38.6918 bar
  * Mean Saturation RMSE = 0.021778
  * Final-Step Saturation RMSE = 0.029570

---

## 3. Document Manifest Counts

* **Figure Count**: 12 figures
  - Fig 1: Dataset Overview (fig6_dataset_overview.png)
  - Fig 2: Mesh Topology (fig7_mesh_topology.png)
  - Fig 3: Pressure Evolution (fig11_pressure_evolution.png)
  - Fig 4: Baseline Comparison (fig9_baseline_comparison.png)
  - Fig 5: Feature Importance (fig10_feature_importance.png)
  - Fig 6: Interpolation Error Study (fig8_interpolation_error.png)
  - Fig 7: ST-GNN Architecture Diagram (fig12_stgnn_architecture.png)
  - Fig 8: Training Loss Curves (fig1_training_curves.png)
  - Fig 9: Rollout RMSE (fig2_rollout_rmse.png)
  - Fig 10: Pressure Tracking (fig3_pressure_tracking.png)
  - Fig 11: Saturation Tracking (fig4_saturation_tracking.png)
  - Fig 12: Bar Comparison (fig5_bar_comparison.png)
* **Table Count**: 5 tables (all inlined in sections for compile-readiness)
  - Table 1: Dataset stats (dataset_stats)
  - Table 2: Baseline results (baseline_results)
  - Table 3: Feature results (feature_results)
  - Table 4: ST-GNN comparison (stgnn_comparison)
  - Table 5: Rollout metrics (rollout_metrics)
* **Bibliography Count**: 13 entries in references.bib
* **Appendix Sections**: 3 sections (Nomenclature, Hyperparameters, Reproducibility)
