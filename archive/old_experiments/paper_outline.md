# Paper Outline: Spatio-Temporal Graph Neural Networks for Pressure Forecasting in Underground Hydrogen Storage Reservoirs on Irregular PEBI Meshes

## Abstract
* **Context**: Underground Hydrogen Storage (UHS) as a seasonal energy storage option.
* **Problem**: Numerical simulators are slow; standard machine learning surrogates require regular rectangular grids, which introduces spatial projection/interpolation errors on PEBI/Voronoi meshes.
* **Insight 1 (Deceptive Target)**: Direct pressure prediction ($P_{t+1}$) is dominated by temporal autocorrelation, allowing trivial copy models to achieve artificially high metrics ($R^2 > 0.99$) while learning no physics.
* **Insight 2 (Pressure-Change)**: Reformulating the target to pressure change ($\Delta P = P_{t+1} - P_t$) exposes the true difficulty of the task.
* **Insight 3 (Temporal Inertia)**: Spatiotemporal feature engineering (adding $\Delta P_{t-1}$) allows local models to capture transient decay dynamics ($R^2 = 0.9827$, RMSE = 0.5487 bar).
* **Insight 4 (Grid Factorization Bug & GNN)**: Explaining how naive spatial features failed due to mesh irregularity and introducing a physical transmissibility-weighted Spatio-Temporal GNN (ST-GNN) that operates on the native irregular grid.
* **Results**: Rollout metrics show ST-GNN v2 outperforms v1, achieving a 31.6% reduction in final-step pressure RMSE.

## 1. Introduction
* Context of green hydrogen and seasonal storage in subsurface formations.
* Limitations of traditional numerical simulation (ECLIPSE, INTERSECT, tNavigator).
* The rise of Scientific Machine Learning (SciML) and surrogates.
* The challenge of irregular spatial grids (PEBI / Voronoi) in reservoir modeling.
* Roadmap and contributions of this study.

## 2. Background and Governing Physics
* **Governing PDEs**: Two-phase mass conservation equation for water ($w$) and gas/hydrogen ($g$).
* **Simplification**: Transient diffusion equation for compressible single-phase fluid systems.
* **Discretization**: Finite Volume Method (FVM) formulation on irregular Voronoi/PEBI grids.
* **Physical Connectors**: Inter-cell transmissibility ($T_{ij}$) dependent on shared area ($A_{ij}$), centroid distances ($d_i, d_j$), absolute permeability tensor ($\mathbf{k}$), and fluid viscosity ($\mu$).

## 3. Dataset and Reservoir Description
* **Simulation Setup**: 5 synthetic cyclic storage cases on a 2D irregular PEBI mesh.
* **Grid Properties**: 10,116 cells, irregular coordinates (403 unique $X$, 346 unique $Z$).
* **Time Scale**: 146 steps (5-day intervals, 730 days total).
* **Dynamic Range**: Well BHP, water/gas saturation, liquid/vapor hydrogen mole fractions, bacterial concentration.
* **Z-score Standardization**: Normalization based strictly on training cases (Cases 1–4) to prevent data leakage.

## 4. Baseline Surrogate Models
* **Formulation**: Predicting $P(t+1)$ directly from cell-local state and controls.
* **Baseline Models**: Persistence ($P_{t+1} = P_t$), Linear Regression, Multi-Layer Perceptron (MLP).
* **Deceptive Metrics**: High $R^2 \approx 0.995$ and low RMSE $\approx 3.42$ bar.
* **Critical Critique**: Why this represents a temporal copying mechanism rather than transient flow modeling.

## 5. Pressure-Change Formulation
* **Formulation**: Redefining the target as $\Delta P = P(t+1) - P(t)$.
* **Impact**: Exposing the inadequacy of local features ($R^2 \approx 0.001$ for linear regression).
* **Ablation Studies**: Explaining how the absence of single dominant local features proves that pressure change is driven by spatial gradients and temporal histories.

## 6. Feature Engineering and Scientific Insights
* **Feature Sets**: Baseline (9 features), Spatial (19 features), Spatiotemporal (25 features).
* **Quantitative Results**: Random Forest ($R^2 = 0.9827$, RMSE = 0.5487 bar) and MLP ($R^2 = 0.9529$, RMSE = 0.9063 bar) under spatiotemporal features.
* **Physical Insight**: Temporal inertia and local pressure decay.
* **The Grid Factorization Bug**: How attempting to factor an irregular 10,116-cell PEBI mesh into a structured 2D grid mathematically caused spatial gradients to collapse to zero, emphasizing the need for graph representations.

## 7. Graph-Based Surrogate Modeling
* **Justification**: Structured grid projections (CNNs, ConvLSTMs, FNOs) introduce unacceptable spatial truncation errors (RMSE up to 11.4 bar).
* **Native Representation**: Reservoir mesh modeled as a directed graph $G = (V, E)$.
* **Node Features**: Static geology (permeability, porosity), dynamic state (pressures, saturations), and controls.
* **Edge Features**: Geometrical area, distances, and transmissibility ($T_{ij}$).

## 8. ST-GNN Architecture
* **Message Passing**: Transmissibility-weighted graph convolutions mirroring the FVM formulation.
* **Layer Stacking**: stacked graph convolutional layers with residual connections, LayerNorm, and LeakyReLU activations.
* **Recurrent State**: Gated Recurrent Unit (GRU) to track transient memory.
* **Scheduled Sampling Curriculum**: Training with rollout probability ($p_{\text{rollout}} = 0 \to 0.5$) to stabilize autoregressive forecasting.
* **Loss-Weight Formulation**: Weighted MSE loss balancing pressure and saturation: $\mathcal{L} = \alpha_P \mathcal{L}_P + \alpha_{S_g} \mathcal{L}_{S_g}$.

## 9. Rollout Forecasting Results
* **Multi-Step Rollout**: Evaluating model performance over 142 steps on Case 5 (independent test case).
* **ST-GNN v1 vs. v2**: Showing how the architectural improvements in v2 (residual paths, LayerNorm, scheduled sampling, larger dimension) reduce final-step pressure RMSE from 49.25 bar to 33.67 bar (31.6% reduction) and mean rollout RMSE from 35.26 bar to 31.72 bar (10.0% reduction).
* **PI-GNN Comparison**: Benchmarking against the physics-informed baseline, which achieved a mean rollout pressure RMSE of 59.37 bar.

## 10. Discussion
* **Direct vs. Delta Target**: The physical necessity of predicting changes.
* **Grid Irregularity**: Why standard CNNs fail and GNNs are the natural solution.
* **Temporal Inertia**: Why previous pressure changes act as local indicators of transient decay.
* **Rollout Drift**: Autoregressive error accumulation in saturation and pressure.

## 11. Limitations
* Dataset scale (5 simulation cases) limits generalization to unseen geologic fields and well configurations.
* Autoregressive rollout drift over long timelines (>100 steps) remains a major challenge.
* High computational overhead during PyG data preprocessing on large-scale grids.

## 12. Conclusions
* Summary of major findings: the importance of native graph connectivity, the spatiotemporal inertia, and the curriculum training.
* Future work: scaling the dataset to 50–100 cases, geostatistical field variation, and coupling with Bayesian optimization for well control.
