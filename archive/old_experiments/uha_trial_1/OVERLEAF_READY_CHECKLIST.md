# Overleaf Ready Checklist

This checklist confirms that the `uha_trial_1/` folder represents a complete, self-contained, and validated conference submission package ready to be uploaded directly to Overleaf.

---

## 1. Directory Completeness Verification

- [x] **Root Directory Files**:
  - `main.tex` (Main LaTeX document)
  - `references.bib` (Complete and expanded BibTeX database)
  - `reviewer2_report.md` (Peer review audit report)
  - `publication_readiness.md` (Publication tier assessment)
  - `pdf_quality_report.md` (Typesetting audit report)
  - `story_audit.md` (Scientific narrative evaluation)
  - `conference_questions.md` (20 defense Q&A preparation guide)
  - `future_work_roadmap.md` (7-step prioritized roadmap)
  - `corrections_log.md` (Log of scientific revisions)
  - `VERSION_FREEZE.md` (Frozen metadata and metrics sheet)
- [x] **Sections Directory (`sections/`)**:
  - `abstract.tex`
  - `introduction.tex`
  - `physics.tex`
  - `dataset.tex` (inlined Table 1)
  - `baseline_models.tex` (inlined Table 2)
  - `pressure_change.tex`
  - `feature_engineering.tex` (inlined Table 3)
  - `graph_surrogate.tex`
  - `stgnn.tex`
  - `rollout_results.tex` (inlined Table 4 and Table 5)
  - `discussion.tex`
  - `practical_implications.tex`
  - `limitations.tex`
  - `conclusions.tex`
- [x] **Supplementary Directory (`supplementary/`)**:
  - `nomenclature.tex`
  - `hyperparameters.tex`
  - `reproducibility.tex`
- [x] **Figures Directory (`figures/`)**:
  - `fig1_training_curves.png`
  - `fig2_rollout_rmse.png`
  - `fig3_pressure_tracking.png`
  - `fig4_saturation_tracking.png`
  - `fig5_bar_comparison.png`
  - `fig6_dataset_overview.png`
  - `fig7_mesh_topology.png`
  - `fig8_interpolation_error.png`
  - `fig9_baseline_comparison.png`
  - `fig10_feature_importance.png`
  - `fig11_pressure_evolution.png`
  - `fig12_stgnn_architecture.png`

---

## 2. Compile-Readiness Checks (pdflatex + bibtex)

- [x] **No Missing Input Files**: Checked that all `\input{sections/...}` and `\input{supplementary/...}` files exist.
- [x] **No Broken Cross-References**: Ran the python validation script; verified that all 31 references (e.g. `Eq.~\ref{eq:mass_cons}`) map exactly to their corresponding `\label{}` definitions with 0 warnings.
- [x] **No Broken Citations**: Verified that all 18 `\cite{}` calls map to active keys inside `references.bib` with 0 warnings.
- [x] **No Missing Figures**: Verified that all 12 `\includegraphics{}` paths point to existing files in the `figures/` folder.
- [x] **Clean Preamble**: Removed redundant packages, resolved package conflicts, and kept only standard packages compatible with the standard pdflatex engine on Overleaf:
  * `geometry` (margins setup)
  * `booktabs` (clean tables)
  * `hyperref` (hyperlinks)
  * `amsmath`, `amssymb` (math typeset)
  * `graphicx` (embedded figures)
  * `cite` (bracketed citation format)
  * `authblk` (author affiliations)
- [x] **Inlined Tables**: Inlined all five tables in their respective section files, removing the `tables/` directory requirement and ensuring Overleaf compiles without external file dependencies.
