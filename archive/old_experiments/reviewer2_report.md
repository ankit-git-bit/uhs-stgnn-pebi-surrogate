# Peer Review Report (Reviewer #2)

**Manuscript Title**: Spatio-Temporal Graph Neural Networks for Pressure Forecasting in Underground Hydrogen Storage Reservoirs on Irregular PEBI Meshes  
**Target Venues**: SPE Reservoir Simulation Conference / ECMOR / EAGE / Computational Geosciences / NeurIPS SciML Workshop  

---

## A. Summary of the Paper
This paper presents a Graph Neural Network (GNN) framework to emulate fluid pressure and gas saturation dynamics in Underground Hydrogen Storage (UHS) reservoirs discretized on irregular Perpendicular Bisection (PEBI) meshes. The authors identify a key evaluation pitfall in absolute pressure forecasting ($P_{t+1}$), demonstrating that temporal autocorrelation allows trivial copy models to achieve $R^2 > 0.99$ without learning reservoir physics. They reformulate the target as a one-step pressure change ($\Delta P$). They show that temporal features (particularly the lag $\Delta P_{t-1}$) dominate predictions due to temporal inertia. The paper proposes a Spatio-Temporal GNN (ST-GNN) that maps cell transmissibilities ($T_{ij}$) to graph edge weights. Under a 142-step autoregressive rollout, the ST-GNN v2 model (utilizing residual skip connections, LayerNorm, and scheduled sampling) achieves a 31.63% reduction in final-step pressure RMSE (33.67 bar) compared to the v1 baseline (49.25 bar).

---

## B. Major Comments

### 1. Severe Dataset Deficit and Generalizability Claims
The training dataset is composed of only **4 simulation cases** (with Case 5 held out for testing) under a **single geological model template**. From a machine learning perspective, training a deep neural network (with 128 hidden features and 4 layers) on 4 samples is highly insufficient. The surrogate is not evaluated on:
* Unseen permeability realizations (drawn from a different variogram range or anisotropy).
* Unseen well configurations or placement coordinates (well centroids are identical in all cases).
* Alternative geological features (no faults, fractures, or facies changes).

The model behaves as a **cyclic trajectory emulator for one specific geological grid** rather than a generalized physical solver. The authors must tone down any claims of general geological surrogate capabilities. In its current form, this is a local overfitting study.

### 2. Large Rollout Drift and Lack of Practical Utility
The authors report a peak pressure RMSE of **59.19 bar at step 50** for the ST-GNN v2 model. Given that the mean reservoir pressure is 120.70 bar, an error of 59.19 bar represents a **49.0% deviation** from the mean state. 
* While the error partially recovers to 33.67 bar at step 146, a peak error of 49% in the mid-rollout makes this model completely useless for real-time operations, caprock integrity monitoring, or hydrogen containment safety analysis.
* The "partial recovery" at the end of the rollout is likely a statistical artifact of the model regressing toward the historical mean pressure of the production phase, rather than a physical recovery of flow dynamics.

### 3. Physical Inconsistency and Saturation Drift
The model co-emulates pressure change and gas saturation change ($\Delta S_g$) but does not enforce any physical conservation laws (such as mass balance).
* The cumulative gas saturation error reaches an RMSE of **0.404** (Table 4, max $S_g$-Error for v2). Saturation is bounded between 0.0 and 1.0, so an RMSE of 0.404 represents massive physical drift.
* The algebraic closure $S_w = 1 - S_g$ is used, but this does not prevent local hydrogen mass accumulation or depletion. Purely data-driven MSE loss is insufficient for reservoir engineering emulators. The authors must explain how they plan to incorporate mass-conservation constraints.

### 4. Lack of Standard Operator Learning Benchmarks
The authors justify not benchmarking against the Fourier Neural Operator (FNO) or DeepONet by stating that PEBI meshes require non-trivial parameterization.
* While PEBI grids are unstructured, there are standard ways to interpolate these grids onto Cartesian coordinates to run FNO, or to use Graph Neural Operator (GNO) architectures. 
* By failing to run any comparative benchmarks against operator-learning methods, the paper lacks a standard baseline. The comparison is restricted to basic OLS, Ridge, RF, and MLP models, which are weak baselines for a deep learning paper.

---

## C. Minor Comments
1. **Normalization Table**: In Table 1, the variable `dt_days` is shown to have a mean of 5.0 and standard deviation of 0.0. A constant feature with zero variance should be completely removed from the model rather than normalized, as it provides no information. The authors state it is excluded, but listing it in Table 1 as a normalized variable is confusing.
2. **RF vs GNN Metric Discrepancy**: The Random Forest achieves $R^2 = 0.9827$ (RMSE = 0.5487 bar) which is far superior to the ST-GNN v2 rollout results. Although the authors explain this in the discussion as a teacher-forcing vs. rollout difference, they should report the Random Forest performance in rollout mode as well to make a fair comparison.
3. **Typography**: Ensure all dynamic variable names (e.g. `inj_BHP`, `prod_BHP`) are formatted consistently in math mode or text mode.
4. **Figure Labels**: Fig. 7 and Fig. 11 in the monolithic draft are referenced out of order. Ensure that figure numbering matches the order of appearance in the text.

---

## D. Scientific Strengths
1. **Exposure of the Autocorrelation Trap**: The identification of the direct pressure prediction pitfall ($P_{t+1}$) is a significant contribution. Demonstrating that a zero-parameter Persistence model achieves $R^2 = 0.9931$ exposes a major flaw in how many published reservoir surrogates report accuracy.
2. **Transmissibility-Weighted Graph Convolution**: Mapping the physical two-point transmissibility ($T_{ij}$) directly to GNN edge weights is mathematically elegant and establishes a direct analogy to the Finite Volume Method (FVM) stencil.
3. **Structured Grid Bug Identification**: The analysis of the grid factorization failure provides an excellent diagnostic of how structured-grid assumptions silently corrupt spatial gradients on irregular grids.

---

## E. Scientific Weaknesses
1. **Absence of Generalization Testing**: No validation is performed on unseen geology or well locations.
2. **High Rollout Error**: The 59.19 bar rollout pressure error represents a major accuracy bottleneck.
3. **Lack of Conservation Constraints**: The model predicts saturation and pressure changes without enforcing mass conservation, violating basic reservoir physics.
4. **Weak Baselines**: No comparison against modern neural operators (FNO, DeepONet, GNO).

---

## F. Novelty Assessment
The novelty is **Moderate**. Using GNNs for reservoir emulation is not new. However, the combination of:
1. Exposing the autocorrelation floor.
2. Formulating a transmissibility-weighted spatial convolution mirroring the FVM stencil.
3. Performing a systematic spatiotemporal feature ablation study on irregular PEBI meshes.
provides sufficient novelty for a conference-level paper.

---

## G. Reproducibility Assessment
**High**. The authors provide detailed training configurations, hyperparameter settings, normalization statistics, and list the exact libraries used (PyTorch 2.0, PyTorch Geometric 2.3). The dataset structure is well-documented.

---

## H. Acceptance Recommendation
**Recommendation**: **Borderline / Weak Reject**  
* **Justification**: While the paper exposes a critical methodological trap (autocorrelation floor) and presents an elegant transmissibility-weighted GNN formulation, the extremely small dataset scale (4 training cases) and high rollout pressure drift (59.19 bar, 49% error) significantly limit its scientific and practical impact. If the authors can add at least 20 additional cases with varying well locations and permeability structures to demonstrate generalization, this paper would be a strong candidate for acceptance. In its current form, it is borderline.
