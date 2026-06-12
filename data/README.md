# UHS Voronoi Mesh Dataset & Download Guide

This directory manages the raw simulator cases and preprocessed graph arrays for Underground Hydrogen Storage (UHS) forecasting on irregular PEBI (Perpendicular Bisection) Voronoi meshes.

---

## 1. Dataset Overview

The dataset consists of **5 independent simulation cases** representing injection, shut-in, and production operational schedules on an irregular 3D PEBI mesh consisting of **10,116 cells**. 

* **Simulation Duration**: 2-year cycles (730 days).
* **Reporting Intervals**: 5-day intervals (yielding 146 steps per case).
* **Variables**: Centroid coordinates, cell pore volume, cell permeability, cell connections (transmissibility), dynamic cell-wise pressure, dynamic cell-wise gas saturation, well operational BHPs, and injection/production status flags.

### File Specifications & Sizes

| File Name | Size (MB) | Purpose | Split |
| :--- | :--- | :--- | :--- |
| **`Case_0001_wj.mat`** | 59.4 MB | Model training and history-matching | Train |
| **`Case_0002_wj.mat`** | 59.9 MB | Model training and history-matching | Train |
| **`Case_0003_wj.mat`** | 58.4 MB | Model training and history-matching | Train |
| **`Case_0004_wj.mat`** | 59.8 MB | Model training and history-matching | Train |
| **`Case_0005_wj.mat`** | 60.0 MB | Epoch validation & multi-step rollout test | Val / Test |
| **`pressure_surrogate_dataset.npz`** | 229.7 MB | Preprocessed training inputs and targets | Ignored (Regenerated) |
| **`mesh_graph.npz`** | 2.0 MB | Extracted Delaunay edges and transmissibility weights | Tracked in Git |

---

## 2. Storage Requirements

* **Minimal Clone Size**: **~5.5 MB** (with preprocessed `mesh_graph.npz` and checkpoints).
* **With Raw Datasets**: **~305 MB**.
* **With Preprocessed Training Matrices**: **~535 MB**.

---

## 3. Download Instructions

To reproduce the evaluations:
1. Contact the manuscript authors or visit the Zenodo/Mendeley Data repository link at: `https://doi.org/10.5281/zenodo.xxxxxxx` *(placeholder link to be updated upon paper publication)*.
2. Download the five MATLAB case files (`Case_0001_wj.mat` through `Case_0005_wj.mat`).
3. Place them directly into the `data/raw/` folder in your local clone of the repository.

---

## 4. Expected Directory Structure

After downloading and running the preprocessing pipelines, your `data/` directory layout must match the following:

```
data/
├── README.md                      # This guide
│
├── raw/                           # [EXCLUDED FROM GIT]
│   ├── Case_0001_wj.mat           # Raw simulator output Case 1
│   ├── Case_0002_wj.mat           # Raw simulator output Case 2
│   ├── Case_0003_wj.mat           # Raw simulator output Case 3
│   ├── Case_0004_wj.mat           # Raw simulator output Case 4
│   └── Case_0005_wj.mat           # Raw simulator output Case 5
│
└── processed/
    ├── mesh_graph.npz             # [TRACKED] Irregular PEBI mesh graph structure
    ├── normalization_stats.json   # Mean and standard deviation scaling factors
    ├── st_gnn_stats.json          # Metrics from GNN training
    ├── st_gnn_v2_stats.json       # Metrics from GNN v2 training
    ├── st_gnn_v3_stats.json       # Metrics from GNN v3 training
    └── pressure_surrogate_dataset.npz # [EXCLUDED FROM GIT] Generated training data
```

No code modifications are required. All preprocessing, baseline training, and GNN rollout evaluation scripts dynamically look for cases inside `data/raw/` and outputs inside `data/processed/`.
