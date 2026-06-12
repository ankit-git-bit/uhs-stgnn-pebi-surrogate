# Step-by-Step Reproducibility Guide

Follow these instructions to reproduce the baselines, GNN rollout metrics, figures, and tables in the paper using the preserved pre-trained checkpoints.

---

## 1. Environment Setup

Configure the environment using either Conda or Pip. No PyTorch Geometric or external CUDA compilation is needed.

### Conda setup
```bash
conda env create -f environment.yml
conda activate sciml_models
```

### Pip setup
```bash
pip install -r requirements.txt
```

---

## 2. Dataset Setup

This repository excludes large `.mat` case files to remain lightweight. To run the evaluations:
1. Download the raw dataset files (`Case_0001_wj.mat` through `Case_0005_wj.mat`) using the download instructions in [data/README.md](../data/README.md).
2. Place the downloaded case files inside the `data/raw/` directory.

---

## 3. Reproduce All Outputs Automatically

To execute the entire verification pipeline from start to finish:

### On Unix/Linux/macOS
```bash
chmod +x reproduce_results.sh
./reproduce_results.sh
```

### On Windows (PowerShell)
```powershell
Set-ExecutionPolicy Bypass -Scope Process
.\reproduce_results.ps1
```

---

## 4. Run Steps Separately

If you prefer to execute the pipeline manually:

### Step 1: Extract Graph Connectivity
Constructs Delaunay edge lists, Voronoi centroid area factors, and transmissibility weights:
```bash
python src/graph_construction/graph_construction.py
```
*Output: `data/processed/mesh_graph.npz`*

### Step 2: Extract Baseline & Feature Engineering Metrics
Trains Baseline, Spatial, and Spatiotemporal OLS/Ridge/RF/MLP models on the raw cases and evaluates them, saving the feature importance ranking:
```bash
python src/feature_engineering/phase4_spatial_temporal_features.py
```
*Outputs:*
* `results/metrics/PHASE_4_FEATURE_ENGINEERING_SUMMARY.json`
* `results/figures/model_performance_comparison.png`
* `results/figures/feature_importance_comparison.png`

### Step 3: Run ST-GNN Rollout Evaluations
Loads existing checkpoints (`checkpoints/stgnn_v1/st_gnn_checkpoint.pt` etc.) and runs 142-step autoregressive rollouts on the held-out test Case 5. It prints the multi-step RMSE tables and saves Figures 8-12:
```bash
python src/evaluation/evaluate_rollout_v3.py
```
*Outputs:*
* `results/metrics/v3_evaluation_results.json`
* Rollout figures saved to `results/figures/` and `paper_project/figures/` (mapped to paper naming: `fig8_training_curves` to `fig12_bar_comparison`).

### Step 4: Generate Supplementary Plots
Generates the remaining plots (Figures 1 to 7) including dataset overview, PEBI centroids, pressure snapshots, interpolation grids, and GNN schematic:
```bash
python src/utils/generate_missing_plots.py
python src/utils/generate_fig7_stgnn.py
```
*Outputs: Figures 1-7 in both `.png` and `.pdf` formats saved to `results/figures/` and `paper_project/figures/`.*

### Step 5: Assemble and Validate LaTeX
Combines modular section files into a monolithic manuscript and validates links/cites/graphics:
```bash
python src/utils/reconstruct_final_paper.py
python src/utils/validate_latex.py
```
*Outputs:*
* Unified manuscript `final_paper.tex` at the project root.
* Compilation checks printed to console (yielding validation SUCCESSFUL).
