# UHS GNN Surrogate Paper - Overleaf Submission Package

This folder contains the complete, submission-ready LaTeX project for the paper:
**"Spatio-Temporal Graph Neural Networks for Underground Hydrogen Storage Forecasting on Irregular PEBI Meshes"**

This package is configured to compile directly on Overleaf with zero manual edits, and includes lightweight vector PDFs for all figures to prevent compilation timeouts.

## Overleaf Upload Instructions

1. **Create a ZIP Archive**: ZIP all the files and folders inside this directory (i.e., `sections/`, `figures/`, `supplementary/`, `tables/`, `main.tex`, `references.bib`, and this `README.md`). Do not ZIP the parent folder itself, just the contents.
2. **Upload to Overleaf**: On your Overleaf dashboard, click **New Project** -> **Upload Project** and select the `.zip` file you created.
3. **Configure Compiler Settings**:
   - **Compiler**: `pdfLaTeX`
   - **TeX Live Version**: `2023` or later (recommended)
   - **Main Document**: `main.tex`
4. **Compile**: Click **Recompile**. The document will compile immediately.

## Directory Structure

```text
uha_trial_1/
│
├── main.tex                  # Main LaTeX entry point (modular)
├── references.bib            # Complete bibliography database
├── README.md                 # This instruction file
│
├── sections/                 # LaTeX sections (modular)
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
│
├── supplementary/            # LaTeX appendices (modular)
│   ├── nomenclature.tex
│   ├── hyperparameters.tex
│   └── reproducibility.tex
│
├── tables/                   # Pre-compiled LaTeX table inputs
│   ├── baseline_results.tex
│   ├── dataset_stats.tex
│   ├── feature_results.tex
│   ├── rollout_metrics.tex
│   └── stgnn_comparison.tex
│
├── figures/                  # Figures folder containing both PDF (vector) and PNG (raster)
│   ├── fig1_dataset_overview.{pdf,png}
│   ├── fig2_mesh_topology.{pdf,png}
│   ├── fig3_pressure_evolution.{pdf,png}
│   ├── fig4_baseline_comparison.{pdf,png}
│   ├── fig5_feature_importance.{pdf,png}
│   ├── fig6_interpolation_error.{pdf,png}
│   ├── fig7_stgnn_architecture.{pdf,png}
│   ├── fig8_training_curves.{pdf,png}
│   ├── fig9_rollout_rmse.{pdf,png}
│   ├── fig10_pressure_tracking.{pdf,png}
│   ├── fig11_saturation_tracking.{pdf,png}
│   └── fig12_bar_comparison.{pdf,png}
│
└── [MD Reports]              # Scientific reviews and publication readiness reports
    ├── reviewer2_report.md
    ├── publication_readiness.md
    ├── pdf_quality_report.md
    ├── story_audit.md
    ├── conference_questions.md
    ├── future_work_roadmap.md
    └── corrections_log.md
```

## Compilation Notes

- **Vector Graphics Integration**: All `\includegraphics` calls omit file extensions (e.g., `{figures/fig1_dataset_overview}`). This configuration instructs the LaTeX engine to prioritize loading the lightweight vector `.pdf` files, resolving any compile timeout errors (which frequently occur when loading high-resolution 600 DPI `.png` raster files).
- **Author Anonymization**: The author block in `main.tex` is configured for **Anonymous Review** as required by conference guidelines. Commented-out templates for the camera-ready author block are provided in `main.tex` for easy reactivation post-review.
- **Bibliography**: Checked and verified for completeness. All entries in `references.bib` resolve correctly without undefined citation warnings.
