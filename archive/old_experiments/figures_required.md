# Required Figures Blueprint: UHS Pressure Surrogate Modeling

This document lists the 11 required figures for the conference paper and technical project report, detailing their captions, content description, and role in the narrative.

---

### Figure 1: Dataset Overview
* **Caption**: *Figure 1: Overview of the Underground Hydrogen Storage (UHS) dataset, showing typical cyclic well operating schedules (injection/production BHP and flow rates) and the resulting average reservoir pressure over a 730-day (2-year) timeline.*
* **Content Description**: Line plots of well rates (injection of $H_2$, production of gas/water) and bottomhole pressures (BHP) for Case 0001, demonstrating the cyclic nature (60 injection steps, 60 production steps, 26 shut-in steps) of the storage operations.
* **Role in Paper**: Section 3 (Dataset and Reservoir Description) — establishes the operational schedule transitions (e.g., injection changes at steps 30, 73, 103; production changes at steps 43, 73, 116).

### Figure 2: Mesh Topology
* **Caption**: *Figure 2: Representation of the native irregular PEBI (Perpendicular Bisection) / Voronoi spatial grid, displaying the non-uniform cell centroid distribution, cell interfaces, and the locations of the injection and production wells.*
* **Content Description**: A 2D scatter plot showing the irregular layout of the 10,116 cells (varying $X$-coordinates from 12.5 to 4987.5 m, Z-coordinates from 0.25 to 99.75 m) with spatial lines representing Voronoi edges and highlighting injector/producer centroids.
* **Role in Paper**: Section 3 (Dataset and Reservoir Description) / Section 7 (Graph-Based Surrogate Modeling) — highlights the irregular cell sizes and why structured grid mapping is physically inappropriate.

### Figure 3: Interpolation Error Study
* **Caption**: *Figure 3: Spatial interpolation and reconstruction error of a single reservoir pressure snapshot projected onto regular grids of varying resolution ($64 \times 32$, $128 \times 64$, and $256 \times 128$) and mapped back to native coordinates.*
* **Content Description**: Heatmaps of reconstruction RMSE, highlighting that even at $256 \times 128$ resolution, truncation errors remain high (RMSE up to 8.59 bar) near the wellbores where the pressure gradients are steepest.
* **Role in Paper**: Section 3 (Grid Projection & Mesh Validation) / Section 7 — justifies the rejection of regular convolutional networks (CNNs, ConvLSTMs, FNOs) in favor of native irregular graph neural networks.

### Figure 4: Baseline Model Comparison
* **Caption**: *Figure 4: Test set evaluation metrics (RMSE and $R^2$) for the baseline direct pressure prediction ($P_{t+1}$) compared to the pressure-change prediction ($\Delta P$) formulation across Persistence, Linear Regression, and MLP models.*
* **Content Description**: Bar charts demonstrating that while direct pressure prediction yields deceptively high $R^2 > 0.99$, the linear model completely fails ($R^2 \approx 0.001$) when predicting pressure changes ($\Delta P$), exposing the true difficulty of the task.
* **Role in Paper**: Section 4 & Section 5 — highlights the deceptive target pitfall.

### Figure 5: Feature Importance Ranking
* **Caption**: *Figure 5: Random Forest feature importance rankings for the baseline, spatial, and spatiotemporal feature sets. The addition of temporal lag features ($\Delta P_{t-1}$) dominates the predictive signal (64.65% importance).*
* **Content Description**: Horizontal bar charts displaying feature Gini importance. It illustrates how the "Spatial" features originally registered 0% importance due to a 2D grid factorization bug on the irregular PEBI mesh, whereas spatiotemporal features are dominated by temporal lag and time elapsed.
* **Role in Paper**: Section 6 (Feature Engineering and Scientific Insights) — highlights the physical importance of temporal inertia and details the spatial grid factorization bug.

### Figure 6: Pressure Evolution Examples
* **Caption**: *Figure 6: Spatial pressure distribution snapshots at selected timesteps (during injection, production, and shut-in phases) showing the dynamic expansion of pressure buildup around the injector and drawdown cones around the producer.*
* **Content Description**: Heatmaps of native grid cell pressures at t=30 (end of first injection), t=73 (phase change), and t=146 (final step).
* **Role in Paper**: Section 3 & Section 10 — visualizes the reservoir pressure fields that the models are emulating.

### Figure 7: ST-GNN Architecture Diagram
* **Caption**: *Figure 7: Schematic of the Spatio-Temporal Graph Neural Network (ST-GNN) architecture, showcasing the transmissibility-weighted graph convolution layers, residual connections, LayerNorm, and the Gated Recurrent Unit (GRU) for temporal state propagation.*
* **Content Description**: Flow diagram showing node/edge features, message passing (scaled by inter-cell transmissibility $T_{ij}$), residual skip paths, LayerNorm block, GRU cells, and linear output head for predicting $[\Delta P_i(t), \Delta S_{g,i}(t)]^T$.
* **Role in Paper**: Section 8 (ST-GNN Architecture) — details the physical message passing mechanics.

### Figure 8: Training Loss Curves
* **Caption**: *Figure 8: Training and validation loss curves (weighted MSE) over epochs for the ST-GNN v1 (12 epochs) and v2 (30 epochs) models, demonstrating convergence trends and the best checkpoint identification at the final epochs.*
* **Content Description**: Line plots showing training and validation losses. v2 converges to a final validation loss of 0.248 at Epoch 30.
* **Role in Paper**: Section 9 (Rollout Forecasting Results) — documents model training stability.

### Figure 9: Rollout RMSE versus Timestep
* **Caption**: *Figure 9: Autoregressive rollout RMSE over 142 timesteps on independent Test Case 5 for ST-GNN v1 and v2, highlighting the growth of pressure and saturation errors and the stabilizing effect of scheduled sampling in v2.*
* **Content Description**: Multi-panel line plots of Pressure RMSE (bar) and Saturation RMSE ($S_g$) as a function of the rollout timestep (1 to 142). v2 shows lower pressure error growth (mean: 31.72 bar, final: 33.67 bar) compared to v1 (mean: 35.26 bar, final: 49.25 bar).
* **Role in Paper**: Section 9 (Rollout Forecasting Results) — demonstrates long-term rollout stability improvements.

### Figure 10: Pressure Tracking Plots
* **Caption**: *Figure 10: Pressure tracking comparisons between numerical reservoir simulator ground truth and ST-GNN predictions at specific grid locations (near the injector, near the producer, and in the far-field) over the 142-step rollout.*
* **Content Description**: Multi-panel line plots of pressure (bar) vs. time steps, demonstrating how the model tracks the cyclic pressure drawdowns and buildups, highlighting the accumulation of drift in the late steps.
* **Role in Paper**: Section 9 (Rollout Forecasting Results) — local temporal validation of the surrogate.

### Figure 11: v1 versus v2 Comparison
* **Caption**: *Figure 11: Comparative summary of multi-step rollout performance (at steps 1, 10, 50, and 146) for ST-GNN v1 and v2, highlighting the 31.6% reduction in final-step pressure RMSE achieved by v2's structural improvements.*
* **Content Description**: Grouped bar charts comparing the pressure RMSE (bar) and saturation RMSE ($S_g$) at discrete rollout steps.
* **Role in Paper**: Section 9 (Rollout Forecasting Results) — summarizes the impact of residual connections, LayerNorm, scheduled sampling, and hidden dimension expansion.
