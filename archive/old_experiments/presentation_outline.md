# Conference Presentation Slide Outlines

This document contains slide-by-slide outlines for three presentation formats: a 1-slide executive summary, a 5-slide quick pitch, and a 10-slide standard conference presentation.

---

## A. 1-Slide Executive Summary

### Title: Spatio-Temporal Graph Neural Networks for UHS Pressure Surrogates on PEBI Meshes
* **Motivation**: UHS cyclic schedules induce highly transient flow fields. Traditional simulators are too slow for real-time optimization ($O(10^2$--$10^4)$ evaluations needed), while regular CNNs cannot handle native geological PEBI meshes.
* **The Deceptive Target**: Absolute pressure prediction ($P_{t+1}$) is a misleading baseline due to temporal autocorrelation ($R^2 > 0.99$ for zero-parameter models). The surrogate must target pressure-change ($\Delta P$).
* **Key Findings**:
  * **Temporal Inertia**: Single-step temporal lag ($\Delta P_{t-1}$) explains 64.65% of predictive importance in local models.
  * **Silent Preprocessing Failure**: Projecting PEBI meshes onto Cartesian grids introduces severe wellbore truncation errors (up to 11.42 bar RMSE) or causes silent spatial gradient failures.
* **ST-GNN Solution**: Maps cell transmissibilities ($T_{ij}$) to graph edge weights to mirror Finite Volume updates. 
* **Rollout Results**: ST-GNN v2 (LayerNorm, residual skips, scheduled sampling) reduces 142-step final pressure rollout RMSE by 31.63% (from 49.25 bar to 33.67 bar).
* **Limitations**: Small training scale (5 cases), no mass conservation constraints, and significant mid-cycle rollout drift (59.19 bar).
* **Suggested Figures**: Combine the **ST-GNN Architecture Diagram** (Fig. 12) and the **Rollout RMSE Curve** (Fig. 2) side-by-side.

---

## B. 5-Slide Quick Presentation

### Slide 1: The Problem (Geological Grid Irregularity & Compute Costs)
* **Compute Bottleneck**: High-fidelity simulators solve multi-phase flow PDEs but require minutes to hours. Schedule optimization demands thousands of runs.
* **Mesh Mismatch**: Industrial reservoir grids use irregular PEBI/Voronoi meshes to conform to faults and wells. Standard CNNs/FNOs require structured Cartesian arrays.
* **Projection Error**: Spatial interpolation to regular grids averages out steepest pressure gradients near wellbores, yielding up to 11.42 bar reconstruction RMSE.
* **Suggested Figure**: **Mesh Centroid and Well Topology** (Fig. 7) alongside the **Interpolation Error Study** (Fig. 8).

### Slide 2: The Dataset and The Autocorrelation Trap
* **Dataset**: 5 cases of 2-year cyclic injection-withdrawal cycles on a 10,116-cell PEBI mesh (146 snapshots at 5-day intervals).
* **Evaluation Pitfall**: Framing the task as next-step pressure prediction ($P_{t+1}$) yields deceptive metrics ($R^2 > 0.99$ for Persistence and OLS) due to temporal autocorrelation.
* **The Solution**: Surrogates must target the pressure increment $\Delta P_i(t) = P_i(t+1) - P_i(t)$. Persistence $R^2$ drops to $-0.0198$, exposing the true task difficulty.
* **Suggested Figure**: **UHS Simulation Schedule Profiles** (Fig. 6) and **Baseline RMSE Comparison Bar Chart** (Fig. 9).

### Slide 3: Feature Engineering and The Grid Factorization Bug
* **Temporal Inertia**: Adding temporal history breaks baseline limits. The lag feature $\Delta P_{t-1}$ dominates local models (64.65% Gini importance), representing pressure diffusion rates.
* **The Grid Factorization Bug**: Attempting to calculate spatial gradients via 2D structured reshaping on irregular PEBI meshes causes a silent code failure, forcing gradients to zero.
* **Graph Necessity**: Modelling spatial coupling requires operating on the native grid adjacency graph using cell transmissibilities.
* **Suggested Figure**: **Random Forest Gini Feature Importance rankings** (Fig. 10).

### Slide 4: ST-GNN Architecture & Rollout Results
* **Native Message Passing**: Stacked 4 GCN layers aggregation weighted by row-normalized physical transmissibility ($T_{ij}$).
* **Rollout Stability**: Stacking GCN with residual skip connections and LayerNorm stabilizes training. Ramping scheduled sampling during training mitigates autoregressive drift.
* **Inference Speed**: The GNN performs a 142-step rollout in milliseconds, enabling rapid schedule screening.
* **Suggested Figure**: **ST-GNN Architecture Diagram** (Fig. 12) and **Rollout RMSE Growth Curves** (Fig. 2).

### Slide 5: Conclusions & Key Limitations
* **Key Conclusions**: Surrogates must target $\Delta P$, utilize native graph representations with transmissibility edge weights, and incorporate temporal lag history.
* **Remaining Weaknesses (Brutally Honest)**:
  * Emulation is local: Only tested on 5 cases (single geological model template, identical well locations).
  * Rollout drift: Mid-cycle pressure error reaches 59.19 bar (49% of operating mean).
  * Saturation drift: Saturation RMSE reaches 0.404; no physical conservation is enforced.
* **Suggested Figure**: **Pressure Tracking at Select Grid Locations** (Fig. 3).

---

## C. 10-Slide Conference Presentation

### Slide 1: Motivation & UHS Challenges
* **Context**: Underground Hydrogen Storage (UHS) provides massive capacity to balance renewable energy grids.
* **Physics**: Hydrogen has low viscosity and density, promoting gravity segregation, viscous fingering, and highly transient pressure/saturation fields under cyclic operating schedules.
* **Compute Bottleneck**: Forward simulations take minutes to hours. Ensemble workflows (history matching, optimization) require thousands of evaluations, necessitating fast surrogates.
* **Suggested Figure**: **Dynamic Spatial Pressure Snapshots** (Fig. 11).

### Slide 2: UHS Background & PEBI Meshes
* **Discretization**: Reservoirs use Perpendicular Bisection (PEBI) / Voronoi grids to conform to geological boundaries and well paths.
* **SciML Grid Mismatch**: standard CNNs, ConvLSTMs, and FNOs require structured Cartesian arrays.
* **The Interpolation Cost**: Projecting irregular cells onto structured grids averages out steep wellbore gradients, yielding reconstruction RMSEs from 11.42 bar ($64 \times 32$) down to 8.59 bar ($256 \times 128$).
* **Suggested Figure**: **Interpolation Error Study** (Fig. 8).

### Slide 3: Dataset Characteristics
* **Details**: Five 2-year simulation cases on a 10,116-cell PEBI mesh (146 snapshots, 5-day reporting intervals).
* **Operational Cycle**: Injection (steps 1-60), shut-in (steps 61-86), and production (steps 87-146).
* **Split**: Cases 1-4 for training, Case 5 held out as independent test set.
* **Suggested Figure**: **UHS Simulation Schedule Profiles** (Fig. 6).

### Slide 4: Baseline Models & The Autocorrelation Trap
* **Formulation**: Direct next-step absolute pressure target: $P_i(t+1) = f(X_{\text{local}, i}(t))$.
* **Deceptive Results**: All models achieve $R^2 > 0.99$ (OLS: 0.9952, MLP: 0.9955).
* **The Trap**: The parameter-free Persistence model ($P_{t+1} = P_t$) reaches $R^2 = 0.9931$ and RMSE = 4.22 bar. High metrics are an artifact of temporal autocorrelation, not physical learning.
* **Suggested Figure**: **Baseline RMSE Comparison Bar Chart** (Fig. 9).

### Slide 5: Pressure-Change Reformulation
* **New Target**: Predicting the one-step pressure increment $\Delta P_i(t) = P_i(t+1) - P_i(t)$.
* **Skill Assessment**: Persistence $R^2$ drops to $-0.0198$. OLS linear regression fails completely ($R^2 = 0.0007$). MLP reaches only $R^2 = 0.3524$ (RMSE = 3.36 bar).
* **Local Feature Insufficiency**: Ablation study shows removing local features (permeability, porosity, BHP) yields RMSE change within $\pm 0.0007$ bar. Local features lack spatial gradients and history.
* **Suggested Table**: **Baseline Model Performance Comparison Table** (Table 2).

### Slide 6: Feature Engineering & The Grid Factorization Bug
* **Temporal Inertia**: Adding temporal lag features yields a breakthrough. Random Forest achieves $R^2 = 0.9827$ and RMSE = 0.5487 bar.
* **Lag Dominance**: The lag $\Delta P_{t-1}$ explains 64.65% Gini importance, acting as a local proxy for diffusion rate.
* **Silent Failure**: Spatial features in standard preprocessing failed because factorization of the 10,116-cell PEBI grid into 2D defaulted to a 1D array, forcing gradients to zero.
* **Suggested Figure**: **Random Forest Gini Feature Importance rankings** (Fig. 10).

### Slide 7: Graph Construction on Native PEBI Meshes
* **Graph Definition**: Directed graph $G = (V, E)$. Node $v_i$ represents cell $i$; edge $e_{ij}$ represents face connectivity.
* **Features**: Node features $\mathbf{h}_i \in \mathbb{R}^{20}$ (static, state, 3 pressure lags, dynamic controls).
* **Edge Mapping**: Edge features $\mathbf{e}_{ij} \in \mathbb{R}^3$ (distance $d_{ij}$, area $A_{ij}$, and two-point transmissibility $T_{ij}$). Edges capture flow pathways directly analogous to FVM.
* **Suggested Figure**: **Mesh Centroid and Well Topology** (Fig. 7).

### Slide 8: Spatio-Temporal GNN (ST-GNN)
* **Darcy Graph Convolution**: Aggregates neighbor messages using row-normalized transmissibility weights $\tilde{T}_{ij}$.
* **Structural Stabilization**: Stacks four convolutions with residual skip connections and LayerNorm.
* **Temporal Gating**: Node states are updated across timesteps using a GRU block, followed by an MLP decoder.
* **Curriculum Learning**: Scheduled sampling (probability $p$ scaled $0.0 \to 0.5$) trains the model on its own predictions to stabilize rollouts.
* **Suggested Figure**: **ST-GNN Architecture Diagram** (Fig. 12).

### Slide 9: Rollout Results & Error Progression
* **Horizons**: 142-step fully autoregressive rollout on Test Case 5.
* **GNN Progression**: ST-GNN v2 (128 hidden features, residual skip, LayerNorm, scheduled sampling) yields a 31.63% reduction in final pressure RMSE (33.67 bar) vs v1 (49.25 bar).
* **Drift Bottleneck**: Pressure error spikes to 59.19 bar at step 50 (shut-in transition) before settling. Saturation RMSE grows to 0.404.
* **Suggested Figure**: **Autoregressive Rollout RMSE Curves** (Fig. 2) and **Pressure Tracking Plots** (Fig. 3).

### Slide 10: Conclusions & Future Outlook
* **Conclusions**: Surrogates must target pressure changes and operate on native graphs with transmissibility edge weights. Temporal lags are critical.
* **Outlook**: Future research must focus on (i) expanding dataset scale to 50+ cases for geology and well diversity, (ii) incorporating physics-informed neural network (PINN) loss terms to enforce mass conservation, and (iii) benchmarking against conformal operator learning methods (FNO, DeepONet).
* **Suggested Figure**: **Multi-step Rollout Comparison Bar Charts** (Fig. 5).
