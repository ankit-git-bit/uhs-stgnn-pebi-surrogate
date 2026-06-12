# Manuscript Corrections and Peer-Review Log

This log documents the scientific, technical, stylistic, and LaTeX corrections made to the manuscript **"Spatio-Temporal Graph Neural Networks for Pressure Forecasting in Underground Hydrogen Storage Reservoirs on Irregular PEBI Meshes"**.

---

## 1. Technical Inconsistencies Fixed

We performed a deep audit of the data files and source code to resolve numerical discrepancies:
* **Feature Count Standardization**: In the initial draft, the baseline, spatial, and spatiotemporal feature sets were inconsistently reported as having 8, 18, and 24 features in the markdown draft, but 9, 19, and 25 features in the LaTeX tables. By inspecting `PHASE_4_FEATURE_ENGINEERING_SUMMARY.json`, we confirmed the true counts are **9, 19, and 25 features** (the baseline includes two dynamic well status flags, `inj_status` and `prod_status`). The text in `feature_engineering.tex` and `final_paper.tex` has been standardized to these exact numbers.
* **Exclusion of Constant Timestep**: The constant timestep variable (`dt_days` = 5.0) has zero variance. We clarified in the text that it is excluded from model feature scaling (since it contributes zero variance) but is documented in the raw dataset characteristics.
* **Autoregressive Rollout Steps**: We confirmed the exact length of the test rollout is **142 steps** (from step 4 to 146, using the first 3 steps to populate the pressure history lags). All references to rollout length are standardized to 142 steps.
* **Standardized Performance Metrics**: All baseline, feature engineering, and ST-GNN v1/v2/v3 performance numbers were audited and cross-checked against `PHASE_2_BASELINE_SUMMARY.json`, `st_gnn_v2_rollout_summary.json`, and `st_gnn_v3_study_summary.json` to ensure complete consistency between text, tables, and figures.

---

## 2. List of Corrections Made

### Scientific & Stylistic Rewrite
* **AI-Language Deletion**: Eliminated highly repetitive and robotic transition phrases such as *"This demonstrates"*, *"This indicates"*, *"This shows"*, and *"This finding suggests"*. Replaced them with natural, diverse scientific phrasing: *"A plausible explanation is..."*, *"The observed trend may arise from..."*, *"Quantitatively,..."*, *"The results reveal..."*, *"The comparison highlights..."*, and *"The data suggest..."*.
* **Evidence-Driven Numerical Discussions**: Converted generic discussions into quantitative, evidence-driven arguments. For example:
  * In Section 4 (Baseline), explicitly compared the Persistence model RMSE (4.22 bar) to OLS (3.53 bar) and MLP (3.42 bar) to expose the autocorrelation floor.
  * In Section 6 (Feature Engineering), added specific Gini importance percentages (64.65% for $\Delta P_{t-1}$ and 24.58% for elapsed time) to explain the physical concept of temporal inertia.
  * In Section 9 (Rollout Results), detailed the stepwise RMSE progression of ST-GNN v2 (0.19 bar at step 1, 1.83 bar at step 10, 59.19 bar at step 50, and 33.67 bar at step 146).

### Section Expansions & Additions
* **Abstract Rewrite**: Reauthored the abstract (228 words) to be strictly numerical and concise, mentioning the 10,116-cell PEBI mesh, 5 cases, Random Forest $R^2 = 0.9827$ and RMSE = 0.5487 bar, ST-GNN rollout v2 vs v1 improvements (31.6% reduction in final-step RMSE), and primary limitations.
* **One-Step RF vs. Rollout GNN Clarification**: Added a dedicated subsection in the Discussion (**"One-Step Prediction versus Autoregressive Rollout Dynamics"**) explaining why the Random Forest one-step performance ($R^2 = 0.9827$, RMSE = 0.5487 bar) is not directly comparable to the GNN rollout results. Highlighted that RF uses teacher-forced one-step prediction, whereas the GNN performs fully autoregressive 142-step forecasting, making it vulnerable to error accumulation.
* **Operator Learning Discussion**: Added a dedicated subsection in the Discussion (**"Relevance of Operator Learning Frameworks"**) discussing FNO, DeepONet, and neural operators. Explained that PEBI/Voronoi meshes require non-trivial conformal parameterization before FNO can be applied, and explicitly stated that no relative performance claims against operator learning methods are drawn from this study.
* **Practical Implications Section**: Created a new Section 11 (**"Practical Implications for UHS Surrogate Modeling"**) before Conclusions, outlining what the surrogate can currently do (rapid screening studies) and cannot do (safety-critical operational control), referencing the 59.19 bar peak rollout error.
* **Limitations Strengthening**: Expanded the Limitations section into an honest list of **9 specific points** covering training scale, geology, mesh topology, benchmarks, physical conservation, and rollout drift.

### LaTeX & Reference Formatting
* **Standardized Citations**: Expanded `references.bib` to include 10 new references spanning Mayur Pal's papers, MPFA, FNO, PINN, and UHS reviews. Ensured every citation key is correctly placed in the text and resolved by the bibliography compiler.
* **Standardized Cross-Referencing**: Standardized all references to equations (`Eq.~\ref{}`), figures (`Fig.~\ref{}`), and tables (`Table~\ref{}`) to comply with publication standards.
* **Figure Placement and Labeling**: Verified that all figure environments use standard `[htbp]` placement and carry appropriate `\label{fig:...}` tags. Added the actual `\includegraphics` commands pointing to the figure files in `figures/` rather than empty vertical spaces.

---

## 3. List of Missing Figures and Recommended Captions

During the audit, we identified that the **ST-GNN Architecture Diagram** (referenced in the blueprint as Figure 7) was missing from the `paper_project` files. We successfully copied `stgnn_architecture.png` from `uhs_extracted` to `paper_project/figures/fig12_stgnn_architecture.png` and embedded it in Section 8. 

Below is the complete manifest of the 12 figures now integrated into the manuscript:

1. **Fig.~\ref{fig:dataset_overview} (figures/fig6_dataset_overview.png)**:
   * *Caption*: `Spatial absolute permeability realizations ($\text{Perm}$, mD) across the active 10,116-cell PEBI mesh for the five simulation cases, showing approximately one order of magnitude of geological heterogeneity.`
2. **Fig.~\ref{fig:mesh_topology} (figures/fig7_mesh_topology.png)**:
   * *Caption*: `Centroid coordinate distribution of the native 10,116-cell irregular PEBI/Voronoi mesh, highlighting the locations of the injector (red triangles) and producer (blue triangles) well centroids.`
3. **Fig.~\ref{fig:pressure_evolution} (figures/fig11_pressure_evolution.png)**:
   * *Caption*: `Dynamic spatial pressure snapshots (bar) at representative timesteps during injection, shut-in, and production phases of Case 1, highlighting localized transient pressure gradients near the wellbores.`
4. **Fig.~\ref{fig:baseline_comparison} (figures/fig9_baseline_comparison.png)**:
   * *Caption*: `Forecasting RMSE (bar) comparison on Test Case 5 for the direct pressure ($P_{t+1}$) versus the pressure-change ($\Delta P$) targets across the three baseline models.`
5. **Fig.~\ref{fig:feature_importance} (figures/fig10_feature_importance.png)**:
   * *Caption*: `Gini feature importance ranking from the Random Forest model trained on the spatiotemporal feature set, showing the dominance of temporal lag ($\Delta P_{t-1}$) and time elapsed.`
6. **Fig.~\ref{fig:interpolation_error} (figures/fig8_interpolation_error.png)**:
   * *Caption*: `Pressure snapshot reconstruction RMSE (bar) comparing structured Cartesian grids ($64 \times 32$, $128 \times 64$, and $256 \times 128$) against the native 10,116-cell PEBI mesh, showing concentrated errors near wellbores.`
7. **Fig.~\ref{fig:stgnn_architecture} (figures/fig12_stgnn_architecture.png)**:
   * *Caption*: `Architecture of the Spatio-Temporal Graph Neural Network (ST-GNN), showing the integration of node/edge inputs, stacked transmissibility-weighted graph convolutions with residual skip connections and LayerNorm, and the GRU block for recurrent temporal update.`
8. **Fig.~\ref{fig:training_curves} (figures/fig1_training_curves.png)**:
   * *Caption*: `Training and validation loss curves (weighted MSE) over epochs for the ST-GNN configurations, showing stable convergence profiles.`
9. **Fig.~\ref{fig:rollout_rmse} (figures/fig2_rollout_rmse.png)**:
   * *Caption*: `Autoregressive rollout RMSE over the 142-step forecasting horizon on Test Case 5. Left: Pressure RMSE (bar). Right: Gas saturation ($S_g$) RMSE.`
10. **Fig.~\ref{fig:pressure_tracking} (figures/fig3_pressure_tracking.png)**:
    * *Caption*: `Pressure tracking comparisons between numerical reservoir simulator ground truth (solid lines) and ST-GNN v2 predictions (dashed lines) at select grid locations over the rollout.`
11. **Fig.~\ref{fig:saturation_tracking} (figures/fig4_saturation_tracking.png)**:
    * *Caption*: `Gas saturation ($S_g$) tracking comparisons between numerical reservoir simulator ground truth (solid lines) and ST-GNN v2 predictions (dashed lines) at select grid locations over the rollout.`
12. **Fig.~\ref{fig:bar_comparison} (figures/fig5_bar_comparison.png)**:
    * *Caption*: `Multi-step rollout comparison bar charts showing pressure and saturation errors at discrete rollout steps (t = 1, 10, 50, and 146) for ST-GNN v1, v2, and v3.`

---

## 4. Reviewer-Style Comments on Remaining Weaknesses

While the revised paper is technically solid and suitable for submission, potential peer reviewers at venues like SPE RSC or Computational Geosciences may raise the following concerns:

1. **Severe Dataset Limitations (Generalizability)**:
   * *Comment*: "The authors train their model on only 4 cases and test on 1 case, representing a single geological template. While the results demonstrate temporal interpolation capabilities, there is no evidence that the model can generalize to unseen geological facies, fault networks, or even different well placement coordinates. A dataset of at least 50–100 cases is necessary to establish generalizability."
2. **Rollout Drift and Operational Utility**:
   * *Comment*: "A peak pressure RMSE of 59.19 bar (representing 49% of the mean operating pressure) occurs at step 50. While the model partially recovers to 33.67 bar at the final step, such a large error in the mid-rollout renders the emulator unusable for real-time safety controls, caprock integrity monitoring, or operational gas-recovery optimization."
3. **Mass Conservation Violations**:
   * *Comment*: "The paper co-emulates pressure and saturation using a weighted MSE loss but does not enforce mass balance. Over a 142-step rollout, the cumulative gas saturation error reaches an RMSE of 0.404. Without hard physical constraints or mass-conservative neural network layers, the predictions are physically inconsistent and can result in mass accumulation or depletion."
4. **Lack of Comparative Benchmarks against Operator Learning**:
   * *Comment*: "The authors note that operator learning frameworks like FNO and DeepONet are not implemented due to mesh parameterization difficulties. While this is understandable for PEBI meshes, the lack of comparison against these state-of-the-art SciML models limits the paper's contribution to geometric deep learning. At a minimum, a baseline comparison on a regular grid slice or simplified geometry should be provided to benchmark GNNs against FNOs."
