# Figure Quality Report

This document reports the quality improvements and resolution audit for the figures in the UHS reservoir surrogate paper. In accordance with reviewer guidelines, all figures have been audited and updated to professional publication-quality resolutions (600 DPI) and exported as vector PDF graphics to optimize typesetting and resolve compilation timeout limits.

| Figure | Description | Raster Resolution (PNG) | Vector Format (PDF) | Regenerated? (Y/N) | Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Figure 1** | `fig1_dataset_overview` | **600 DPI** (5937 $\times$ 4726 pixels) | **Yes** (32 KB vector) | **Yes** | Regenerated via `generate_missing_plots.py` at high DPI. |
| **Figure 2** | `fig2_mesh_topology` | **600 DPI** (7137 $\times$ 680 pixels) | **Yes** (31 KB vector) | **Yes** | Regenerated via `generate_missing_plots.py` at high DPI. |
| **Figure 3** | `fig3_pressure_evolution` | **600 DPI** (6435 $\times$ 6924 pixels) | **Yes** (117 KB vector) | **Yes** | Regenerated via `generate_missing_plots.py` at high DPI. |
| **Figure 4** | `fig4_baseline_comparison` | **600 DPI** (5937 $\times$ 2926 pixels) | **Yes** (24 KB vector) | **Yes** | Regenerated via `generate_missing_plots.py` at high DPI. |
| **Figure 5** | `fig5_feature_importance` | **600 DPI** (4740 $\times$ 3525 pixels) | **Yes** (25 KB vector) | **Yes** | Regenerated via `generate_missing_plots.py` at high DPI. |
| **Figure 6** | `fig6_interpolation_error` | **600 DPI** (4737 $\times$ 2926 pixels) | **Yes** (22 KB vector) | **Yes** | Regenerated via `generate_missing_plots.py` at high DPI. |
| **Figure 7** | `fig7_stgnn_architecture` | **600 DPI** (5940 $\times$ 3540 pixels) | **Yes** (28 KB vector) | **Yes** | Exported/generated at high DPI. |
| **Figure 8** | `fig8_training_curves` | **600 DPI** (5919 $\times$ 2925 pixels) | **Yes** (22 KB vector) | **Yes** | Regenerated via `evaluate_rollout_v3.py` at high DPI. |
| **Figure 9** | `fig9_rollout_rmse` | **600 DPI** (7141 $\times$ 4730 pixels) | **Yes** (31 KB vector) | **Yes** | Regenerated via `evaluate_rollout_v3.py` at high DPI. |
| **Figure 10** | `fig10_pressure_tracking` | **600 DPI** (7129 $\times$ 2906 pixels) | **Yes** (32 KB vector) | **Yes** | Regenerated via `evaluate_rollout_v3.py` with ST-GNN v2 data and steelblue theme. |
| **Figure 11** | `fig11_saturation_tracking` | **600 DPI** (7129 $\times$ 2906 pixels) | **Yes** (29 KB vector) | **Yes** | Regenerated via `evaluate_rollout_v3.py` with ST-GNN v2 data and steelblue theme. |
| **Figure 12** | `fig12_bar_comparison` | **600 DPI** (8340 $\times$ 3555 pixels) | **Yes** (23 KB vector) | **Yes** | Regenerated via `evaluate_rollout_v3.py` at high DPI. |

### Overleaf Compile Timeout Resolution
High-resolution 600 DPI raster images (PNG format) are heavy (totaling over 6 MB) and require substantial CPU processing and decompression overhead to render in pdfLaTeX, which triggers the Overleaf compile timeout limit on the free plan. 
To resolve this, we exported all 12 figures in native vector **PDF format**. The vector PDF files:
1. Maintain infinite geometric resolution for printing.
2. Are extremely lightweight (totaling under 400 KB across all figures).
3. Do not require decompression or rasterization during compilation, resulting in near-instantaneous compilation speeds.
The LaTeX documents have been updated to strip the `.png` extension from `\includegraphics` commands, ensuring the LaTeX compiler automatically selects the vector `.pdf` version first.

### Verification Methodology
The resolution and DPI details were verified programmatically using the Python Pillow (`PIL`) library (via the script `scratch/check_resolutions.py`). All images are stored in both lossless PNG format and vector PDF format to guarantee sharp, print-ready typesetting.
