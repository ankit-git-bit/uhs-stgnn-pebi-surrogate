# Repository Architecture & File Directory Map

This document explains the organization of files and directories across the project.

---

## 1. Directory Tree Layout

```
project_root/
├── README.md                           # Main 60-second landing page
├── LICENSE                             # MIT License
├── CHANGELOG.md                        # Log of refactoring and repository cleanup events
├── RELEASE_NOTES.md                    # Specifications, performance results, and limitations
├── CITATION.cff                        # Citation metadata for Ankit Kumar et al.
├── requirements.txt                    # Pip environment specifications
├── environment.yml                     # Conda environment specifications
│
├── .github/                            # Community templates
│   ├── ISSUE_TEMPLATE.md               # Bug reports and feature requests
│   ├── PULL_REQUEST_TEMPLATE.md        # Contributions tracker
│   └── CONTRIBUTING.md                 # Academic contribution rules
│
├── data/
│   ├── README.md                       # Download guide and storage requirements
│   ├── raw/                            # Excluded from git; contains Case_*.mat files
│   └── processed/                      # Contains mesh_graph.npz (tracked) and scaler stats
│
├── src/                                # Reorganized, modularized pipeline
│   ├── preprocessing/                  # Data preparation
│   │   └── pressure_surrogate_preprocessing.py
│   ├── feature_engineering/            # RF features and baseline models
│   │   └── phase4_spatial_temporal_features.py
│   ├── graph_construction/             # Delaunay/Voronoi graph construction
│   │   └── graph_construction.py
│   ├── models/                         # GNN model definitions
│   │   ├── stgnn_v1/st_gnn_model.py
│   │   ├── stgnn_v2/st_gnn_v2_model.py
│   │   └── stgnn_v3/st_gnn_v3_model.py (shares v2 layout)
│   ├── training/                       # Historical training files
│   │   ├── train_st_gnn_v1.py
│   │   ├── train_st_gnn_v2.py
│   │   └── train_st_gnn_v3.py
│   ├── evaluation/                     # Test rollout evaluations
│   │   ├── evaluate_rollout_v1.py
│   │   ├── evaluate_rollout_v2.py
│   │   └── evaluate_rollout_v3.py      # Main comparative evaluation & figures 8-12
│   └── utils/                          # Plotters and validators
│       ├── generate_missing_plots.py   # Generates figures 1-6
│       ├── generate_fig7_stgnn.py      # Generates figure 7 (architecture diagram)
│       ├── reconstruct_final_paper.py  # Merges sections to final_paper.tex
│       └── validate_latex.py           # LaTeX document validator
│
├── results/                            # Main paper deliverables (preserved)
│   ├── tables/                         # Latex table files (.tex)
│   ├── figures/                        # Preserved publication figures (png & pdf)
│   └── metrics/                        # Raw GNN and baseline evaluation summaries
│
├── checkpoints/                        # Preserved GNN weights (v1, v2, v3)
│
├── docs/                               # Reorganized documentation folder
│   ├── project_overview.md             # Scientific details and contributions
│   ├── repository_structure.md         # This folder guide
│   ├── reproducibility.md              # Steps to reproduce baselines and GNN rollouts
│   ├── evaluation_protocol.md          # Split protocol and preservation disclaimer
│   ├── large_files_report.md           # Size audit and LFS configuration
│   ├── repository_quality_review.md    # Persona scores and strengths
│   └── github_release_checklist.md     # Verification checklists
│
└── archive/                            # Deprecated/archived milestone files
```

---

## 2. Script Descriptions

* **src/graph_construction/graph_construction.py**: Evaluates PEBI node positions, runs Delaunay triangulation and Voronoi tessellation to compute cell-to-cell connections and transmissibilities, saving `mesh_graph.npz`.
* **src/preprocessing/pressure_surrogate_preprocessing.py**: Loads raw simulation case data, structures training matrices ($X_t \rightarrow Y_{t+1}$), scales properties, and saves `pressure_surrogate_dataset.npz` and normalization parameters.
* **src/feature_engineering/phase4_spatial_temporal_features.py**: Explores baseline vs spatial vs spatiotemporal features. Trains OLS, Ridge, Random Forest, and MLP models to output baseline comparison and feature importance metrics.
* **src/evaluation/evaluate_rollout_v3.py**: Autoregressively rolls out predictions over 142 timesteps on Case 5 using pre-trained GNN checkpoints, printing comparisons and saving Figures 8-12.
* **src/utils/generate_missing_plots.py**: Re-creates Figures 1-6 including operational BHPs, Voronoi coordinates, pressure snapshot evolutions, interpolation grids, baseline bar charts, and Random Forest feature rankings.
* **src/utils/generate_fig7_stgnn.py**: Generates the vector PDF flow schematic of GNN blocks.
* **src/utils/reconstruct_final_paper.py**: Concatenates separate modular `.tex` section documents under `paper_project/` into a single monolithic document.
* **src/utils/validate_latex.py**: Recursively parses latex inputs, checking for missing citation keys, broken references, or unresolvable image files.
