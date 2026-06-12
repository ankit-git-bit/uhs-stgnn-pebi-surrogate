# Spatio-Temporal Graph Neural Networks for Underground Hydrogen Storage Forecasting on Irregular PEBI Meshes

This repository contains the scientific codebase, trained models, figures, tables, and manuscript files for the paper:  
**"Spatio-Temporal Graph Neural Networks for Underground Hydrogen Storage Forecasting on Irregular PEBI Meshes"**

---

## 1. Project Overview

Underground Hydrogen Storage (UHS) in depleted reservoirs is a critical technology for grid-scale green energy capacity. Emulating transient fluid flow dynamics is computationally expensive using classical Finite Volume Method (FVM) simulators. 

This project provides a native Graph Neural Network surrogate framework that operates directly on irregular PEBI (Perpendicular Bisection) Voronoi meshes:
* **The Autocorrelation Trap**: Exposes why absolute pressure prediction ($P_{t+1}$) is a deceptive target ($R^2 > 0.99$ for zero-physics models) and formulates the pressure-change target ($\Delta P_t$).
* **Temporal Inertia Analysis**: Systematically engineers spatiotemporal stacks, showing that a one-step temporal pressure lag ($\Delta P_{t-1}$) holds $64.65\%$ of predictive Gini importance.
* **Native ST-GNN**: Proposes a Spatio-Temporal GNN implementing Darcy-weighted message passing scaled by physical transmissibility ($T_{ij}$).
* **Autoregressive Rollout Stability**: Implements Layer Normalization, residual skip connections, and a scheduled sampling curriculum to stabilize a 142-step autoregressive rollout, yielding a $23\%$ reduction in mean pressure RMSE over the baseline.

---

## 2. Directory Structure

```
project_root/
├── README.md                           # This reproducibility guide
├── requirements.txt                    # Pip package specifications
├── environment.yml                     # Conda environment specs
├── reproduce_results.sh                # Main automated reproduction bash script
├── reproduce_results.ps1               # Main automated reproduction PowerShell script
│
├── data/
│   ├── raw/                            # Case_0001_wj.mat to Case_0005_wj.mat (raw files)
│   └── processed/                      # pressure_surrogate_dataset.npz, normalization_stats.json, mesh_graph.npz
│
├── src/
│   ├── preprocessing/
│   │   └── pressure_surrogate_preprocessing.py # Preprocesses raw .mat cases into training pairs
│   ├── feature_engineering/
│   │   └── phase4_spatial_temporal_features.py # Feature stack ablation & baseline OLS/RF training
│   ├── graph_construction/
│   │   └── graph_construction.py       # Delaunay/Voronoi mesh graph extraction
│   ├── models/
│   │   ├── stgnn_v1/
│   │   │   └── st_gnn_model.py         # ST-GNN v1 model architecture
│   │   ├── stgnn_v2/
│   │   │   └── st_gnn_v2_model.py      # ST-GNN v2 model architecture (LN + residuals)
│   │   └── stgnn_v3/
│   │       └── st_gnn_v3_model.py      # ST-GNN v3 model architecture (shares v2 architecture)
│   ├── training/
│   │   ├── train_st_gnn_v1.py          # Training script for ST-GNN v1 (12 epochs)
│   │   ├── train_st_gnn_v2.py          # Training script for ST-GNN v2 (30 epochs, scheduled sampling)
│   │   └── train_st_gnn_v3.py          # Training script for ST-GNN v3 (60 epochs, loss weight sweep)
│   ├── evaluation/
│   │   ├── evaluate_rollout_v1.py      # Rollout evaluation for ST-GNN v1
│   │   ├── evaluate_rollout_v2.py      # Rollout evaluation for ST-GNN v2
│   │   └── evaluate_rollout_v3.py      # Comparative rollout evaluation (v1 vs v2 vs v3) and figures 8-12
│   └── utils/
│       ├── generate_missing_plots.py   # Generates figures 1-6
│       ├── generate_fig7_stgnn.py      # Generates figure 7
│       ├── reconstruct_final_paper.py  # Assembles monolithic LaTeX file
│       └── validate_latex.py           # LaTeX source validator
│
├── results/
│   ├── tables/                         # Latex tables (.tex)
│   ├── figures/                        # Generated figures 1-12 (png and pdf)
│   └── metrics/                        # Rollout & baseline JSON metrics
│
├── checkpoints/
│   ├── stgnn_v1/
│   │   └── st_gnn_checkpoint.pt        # Trained weights for ST-GNN v1
│   ├── stgnn_v2/
│   │   └── st_gnn_v2_checkpoint.pt     # Trained weights for ST-GNN v2
│   └── stgnn_v3/
│       └── st_gnn_v3_checkpoint.pt     # Trained weights for ST-GNN v3
│
├── archive/                            # Legacy milestone reports and unused experiments (PIGNN)
└── SUPERVISOR_RELEASE/                 # Clean, self-contained supervisor package (code, models, outputs)
```

---

## 3. Installation

The codebase requires standard Python packages. There is **no dependency on PyTorch Geometric**, eliminating complex compilation issues.

### Option A: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate sciml_models
```

### Option B: Using Pip
```bash
pip install -r requirements.txt
```

---

## 4. Dataset Requirements

Raw simulation cases are stored as MATLAB binary files (`.mat`) in `data/raw/`:
* `Case_0001_wj.mat` to `Case_0004_wj.mat`: Used for model training and validation.
* `Case_0005_wj.mat`: Held out for fully autoregressive rollout testing.

Each case represents a 2-year cycle (146 reporting intervals at 5-day resolution) of hydrogen injection, shut-in, and production on a 10,116-cell PEBI grid.

---

## 5. Quick Start: Reproduce All Results

To execute the entire pipeline from scratch, run the automated script. It will run graph construction, preprocessing, feature training, rollout evaluation, figure plotting, paper assembly, and verification checks.

### On Unix/Linux/macOS:
```bash
chmod +x reproduce_results.sh
./reproduce_results.sh
```

### On Windows (PowerShell):
```powershell
Set-ExecutionPolicy Bypass -Scope Process
.\reproduce_results.ps1
```

---

## 6. Step-by-Step Execution Commands

If you prefer to run steps individually:

### 1. Build Mesh Graph
Computes Delaunay triangulation and Voronoi face areas to calculate inter-cell transmissibility $T_{ij}$ used as GNN edge weights:
```bash
python src/graph_construction/graph_construction.py
```
*Output: `data/processed/mesh_graph.npz`*

### 2. Preprocess Dataset
Loads raw MATLAB variables, generates input training pairs, fits standard scale parameters, and splits data:
```bash
python src/preprocessing/pressure_surrogate_preprocessing.py
```
*Outputs: `data/processed/pressure_surrogate_dataset.npz`, `data/processed/normalization_stats.json`*

### 3. Feature Engineering & Baselines
Trains Linear OLS, Ridge, Random Forest, and MLP models across Baseline, Spatial, and Spatiotemporal stacks:
```bash
python src/feature_engineering/phase4_spatial_temporal_features.py
```
*Outputs: `results/metrics/PHASE_4_FEATURE_ENGINEERING_SUMMARY.json`, `results/figures/model_performance_comparison.png`, `results/figures/feature_importance_comparison.png`*

### 4. GNN Rollout Evaluation & Figure Generation
Runs a 142-step autoregressive rollout on independent Test Case 5 using GNN checkpoints, compiles metrics, and plots training curves, rollout RMSE, tracking series, and comparisons (Figures 8-12):
```bash
python src/evaluation/evaluate_rollout_v3.py
```
*Outputs: `results/metrics/v3_evaluation_results.json`, GNN figures saved to `results/figures/` and `paper_project/figures/`*

### 5. Generate Supplementary Paper Figures
Generates the remaining manuscript plots (Figures 1-7), including dataset schedules, Voronoi mesh grid centroids, spatial pressure distributions, interpolation errors, and the GNN flow schematic:
```bash
python src/utils/generate_missing_plots.py
python src/utils/generate_fig7_stgnn.py
```
*Outputs: Figures 1-7 in `.png` and `.pdf` saved to `results/figures/` and `paper_project/figures/`*

### 6. Assemble LaTeX Document
Merges modular LaTeX section files into the monolithic publication-ready document:
```bash
python src/utils/reconstruct_final_paper.py
```
*Output: `final_paper.tex`*

### 7. Validate LaTeX Integrity
Runs cross-reference, figure file path, input tag, and citation database validation:
```bash
python src/utils/validate_latex.py
```

---

## 7. Figure and Table Mapping

### Manuscript Figures
* **Figure 1**: Dataset operating schedules (`results/figures/fig1_dataset_overview`)
* **Figure 2**: Voronoi mesh centroids (`results/figures/fig2_mesh_topology`)
* **Figure 3**: Spatial pressure snapshots (`results/figures/fig3_pressure_evolution`)
* **Figure 4**: Baseline target comparison OLS/MLP/RF (`results/figures/fig4_baseline_comparison`)
* **Figure 5**: Random Forest Gini feature importance (`results/figures/fig5_feature_importance`)
* **Figure 6**: regular grid projection reconstruction error (`results/figures/fig6_interpolation_error`)
* **Figure 7**: ST-GNN Architecture Flow Diagram (`results/figures/fig7_stgnn_architecture`)
* **Figure 8**: GNN training curves v2 vs v3 (`results/figures/fig8_training_curves`)
* **Figure 9**: Autoregressive rollout RMSE over time (`results/figures/fig9_rollout_rmse`)
* **Figure 10**: Local pressure tracking vs simulator (`results/figures/fig10_pressure_tracking`)
* **Figure 11**: Local gas saturation tracking vs simulator (`results/figures/fig11_saturation_tracking`)
* **Figure 12**: Multi-step rollout bar comparison GNN v1/v2/v3 (`results/figures/fig12_bar_comparison`)

### Manuscript Tables (LaTeX format in `results/tables/`)
* **Table 1**: Dataset standardization parameters (`dataset_stats.tex`)
* **Table 2**: OLS/MLP baseline errors for direct/increment targets (`baseline_results.tex`)
* **Table 3**: Feature set ablation scores (`feature_results.tex`)
* **Table 4**: GNN multi-step rollout RMSE comparison (`stgnn_comparison.tex`)
* **Table 5**: GNN rollout pressure & saturation metrics (`rollout_metrics.tex`)
