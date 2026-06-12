# Project Scientific Overview: ST-GNN for UHS Surrogate Modeling

## 1. Background & Importance of UHS

Underground Hydrogen Storage (UHS) in depleted oil/gas reservoirs or saline aquifers represents a vital technology for balancing seasonal fluctuations in green energy systems. Because hydrogen features low density and viscosity compared to resident fluids (water, methane), its injection and production cycles induce:
* Gravitational segregation (hydrogen rising rapidly).
* Viscous fingering instabilities.
* Highly transient pressure and saturation fluctuations near wellbores.

Emulating these coupled flow transients requires high-fidelity numerical simulators (e.g. ECLIPSE, INTERSECT, tNavigator). However, Bayes-matching, closed-loop schedule optimizations, and uncertainty quantification studies demand thousands of forward simulations, which is computationally prohibitive.

---

## 2. Main Scientific Contributions

This repository provides the code for a data-driven surrogate model that accelerates pressure and saturation forecasting by emulating fluid flows directly on irregular grids:

1. **The Autocorrelation Trap Analysis**: Exposes why absolute pressure prediction ($P_{t+1}$) is a deceptive target. In smoothly evolving porous media, trivial, physics-free persistence models ($\hat{P}_{t+1} = P_t$) score $R^2 > 0.99$. This project demonstrates that predicting pressure change ($\Delta P_t$) is a mathematically rigorous target.
2. **Feature Engineering Insights**: Identifies that transient pressure propagation has high temporal inertia, where the single-step temporal pressure lag ($\Delta P_{t-1}$) holds $64.65\%$ of Gini feature importance.
3. **Native PEBI Graph Representation**: Irregular Perpendicular Bisection (PEBI) meshes conform naturally to wells and complex boundaries. This project models PEBI geometry as a graph $G = (V, E)$ where centroids map to nodes and transmissibilities $T_{ij}$ map to edge weights, framing GNN message passing as a learned generalization of the Finite Volume flux stencil.
4. **ST-GNN Rollout Stabilization**: Proposes a Spatio-Temporal GNN integrating Darcy-weighted graph convolutions with a Gated Recurrent Unit (GRU). Stabilizes a 142-step autoregressive rollout via residual skip connections, Layer Normalization, and a scheduled sampling curriculum, reducing GNN rollout pressure errors by $23\%$.
