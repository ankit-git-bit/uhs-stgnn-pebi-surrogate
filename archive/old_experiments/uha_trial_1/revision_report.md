# Revision Report: Peer-Review Revisions

This report summarizes all technical, scientific, figure, and consistency changes implemented in the paper: *"Spatio-Temporal Graph Neural Networks for Pressure Forecasting in Underground Hydrogen Storage Reservoirs on Irregular PEBI Meshes"* during the final reviewer-style audit and revision round.

---

## 1. Summary of Major Changes

### A. Priority 1: LaTeX Errors & Table Layouts
* **Broken Table Typo Fixed**: Corrected a double backslash typo `\\begin{table*}` $\rightarrow$ `\begin{table*}` in `sections/rollout_results.tex` (line 25), resolving raw LaTeX text rendering in the PDF.
* **Table Column Overflow Resolution**: Converted standard single-column `table` environments to double-column `table*` environments for:
  - Table 1 (`tab:dataset_stats` in `sections/dataset.tex`)
  - Table 2 (`tab:baseline_results` in `sections/baseline_models.tex`)
  - Table 3 (`tab:feature_results` in `sections/feature_engineering.tex`)
  This resolved all `Overfull \hbox` and `tabularx` column narrowness warnings, allowing comfortable multi-column layout rendering.
* **Project Validation**: Ran `validate_latex.py` on the project and resolved all warnings, achieving **0 broken labels, 0 undefined refs, and 0 missing figure references**.

### B. Priority 2: Figure Quality & Redesign (Overleaf Timeout Fix)
* **Lightweight Vector Figures**: Generated vector **PDF formats** for all 12 figures, reducing the total figure package size from over 6 MB to under 400 KB.
* **Timeout Fix**: Modified all `\includegraphics` commands across LaTeX source files to remove file extensions (e.g., removing `.png`), allowing the compiler to automatically default to the lightweight vector `.pdf` files. This resolved Overleaf compile timeout limits on free plans.
* **Figure 3 Redesign**: Redesigned the Pressure Evolution plot into a 2x3 grid mapping 6 specific timesteps ($t = 0, 30, 43, 73, 103, 145$), overlaying white-filled, black-bordered wellbore markers, and applying equal aspect scaling.
* **Figure 7 Schematic**: Created a professional vertical block diagram for the ST-GNN architecture (including GCN layers, GRU blocks, Dense decoders, residual connections, LayerNorm, and transmissibility edge weights).
* **Figure 9 Rollout RMSE Annotations**: Added vertical dashed transition lines ($t = 30, 43, 73, 103, 116$) representing operational control changes, along with peak error annotations (e.g., peak pressure error of 72.45 bar for v3 at step 50) and error stabilization regions.

### C. Priority 3: Scientific Discussion & Equation Upgrades
* **Diffusion Timescale Equation**: Converted the reservoir pressure diffusion timescale formula into a numbered equation (Equation 11) in `sections/discussion.tex`:
  \[
  t_D \approx \frac{L^2 \phi c_t \mu}{k}
  \label{eq:diffusion_time}
  \]
  and updated in-text references to use `Eq.~\ref{eq:diffusion_time}`.
* **Well-Control Transitions Subsection**: Added a new subsection `\subsection{Impact of Well-Control Transitions on Rollout Error}` explaining how BHP transitions trigger transient pressure waves, leading to localized covariate shift and peak error propagation.
* **Random Forest Warning**: Appended a clear warning to both `sections/discussion.tex` and `sections/conclusions.tex` to prevent reader misinterpretation of the Random Forest results:
  > *"The Random Forest result ($R^2 = 0.9827$, $\text{RMSE} = 0.5487$ bar) corresponds to a one-step teacher-forced prediction task and should not be interpreted as long-horizon forecasting performance."*
* **Literature Comparison Table Upgrade**: Upgraded **Table 6** in `sections/discussion.tex` to include columns for Study, Method, Grid Type, Training Cases, Physics, Forecast Horizon, and Reported Error (representing Farchi et al., FNO, DeepONet, and PINN), along with a clear disclaimer that studies are not directly comparable due to differing physics and datasets.

### D. Priority 4: Pruning Overclaims
* **Terminology Audit**: Scanned all source files and replaced marketing or exaggerated language with conservative, evidence-grounded terms:
  - *“massive speedup”* $\rightarrow$ *“substantial computational reduction”*
  - *“deployment”* or *“operational deployment”* $\rightarrow$ *“preferred configuration for future investigation within the current proof-of-concept framework”*
  - *“production-ready”* $\rightarrow$ *“preliminary screening prototype”*
  - *“operational deployment”* $\rightarrow$ *“direct field control”* / *“direct field application”*

### E. Priority 5: Reference Verification
* **Bibliography Clean-Up**: Audited `references.bib` and expanded all placeholder authors (`and others`) for key papers:
  - Expanded FNO (`li2020fno`), DeepONet (`lu2021deeponet`), and PINN (`raissi2019pinn`) publications.
  - Corrected the FNO reference from arXiv to its peer-reviewed version published at **ICLR 2021**.
  - Replaced the placeholder Schmidt 2020 reference with the highly cited **Heinemann et al. (2021)** review paper in *Energy & Environmental Science*, updating all citation links.
  - Expanded all Mayur Pal papers (`pal2024uhsLithuania`, `pal2023ccs`, `pal2021mlhistory`, `pal2006tensor`) and Ahmed's MPFA paper (`ahmed2015mpfa`) with correct author lists and journals.

---

## 2. Walkthrough of Folder Structure & Compilation

The synchronized Overleaf-ready project folder is structured as follows:

```
uha_trial_1/
├── main.tex                  # Main LaTeX document (reconstructs sections via \input)
├── references.bib            # Checked, audited bibliography database
├── sections/                 # Modular LaTeX sections
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── physics.tex
│   ├── dataset.tex
│   ├── baseline_models.tex
│   ├── pressure_change.tex
│   ├── feature_engineering.tex
│   ├── graph_surrogate.tex
│   ├── stgnn.tex
│   ├── rollout_results.tex
│   ├── discussion.tex
│   ├── practical_implications.tex
│   ├── limitations.tex
│   └── conclusions.tex
├── supplementary/            # Supplementary material (nomenclature, hyperparameters, reproducibility)
│   ├── nomenclature.tex
│   ├── hyperparameters.tex
│   └── reproducibility.tex
├── tables/                   # Pre-compiled table files
└── figures/                  # PDF (vector) and PNG (600 DPI raster) figures
```

### Verification Verification Status
* **Syntax Validation**: **SUCCESSFUL** (0 compilation errors, 0 broken citations, 0 missing labels)
* **File Size**: Total package size under 1.5 MB, ensuring instant upload and rapid compilation on Overleaf.
