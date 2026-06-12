# Changelog

All notable changes to this project during the cleanup and GitHub packaging phase are documented below.

---

## [1.0.0] - 2026-06-12

### Added
* **GitHub Community Files**: Added issue template, pull request template, and contribution guidelines in `.github/`.
* **Standardized Documentation**:
  * `docs/project_overview.md`: Summary of scientific contributions, UHS context, and GNN innovations.
  * `docs/repository_structure.md`: Visual tree directory map detailing modular script functionality.
  * `docs/reproducibility.md`: Step-by-step instructions for reproducing baseline features and GNN model rollouts.
  * `docs/evaluation_protocol.md`: Outlines training, validation, and testing splits, potential evaluation leakage, and alternative split solutions.
  * `docs/large_files_report.md`: Identifies all repository assets > 25MB and provides a guide for Git LFS setup.
  * `data/README.md`: Identifies raw and processed data shapes, sizes, and provides download/installation guidelines.
* **Metadata & Licensing**:
  * `CITATION.cff`: Citation schema mapping authors **Ankit Kumar, Shankar Lal, and Mayur Pal**.
  * `LICENSE`: MIT open-source license.
  * `REPRODUCIBILITY_REPORT.md`: Details safe-mode validation results and LaTeX validation parameters.

### Changed
* **Repository Architecture Re-Modularized**:
  * Restructured project root into modular directories: `src/`, `data/`, `checkpoints/`, `results/`, `docs/`, `archive/`.
  * Moved preprocessing scripts to `src/preprocessing/`, baseline feature training to `src/feature_engineering/`, graph connectivity builder to `src/graph_construction/`, model architectures to `src/models/`, and evaluation scripts to `src/evaluation/`.
  * Relocated all deprecated scripts and intermediate reports to `archive/`.
* **Path & Import Stabilization**:
  * Patched all executable python scripts to dynamically resolve project root (`project_root = Path(__file__).resolve().parents[N]`) and append it to `sys.path` to eliminate absolute-path execution errors.
* **Git Tree Optimization**:
  * Configured `.gitignore` to exclude large simulator `.mat` files and the 229MB preprocessed dataset `pressure_surrogate_dataset.npz` to keep the repository clone fast.
  * Preserved the lightweight `mesh_graph.npz` (2.0 MB) in Git since it is required for inference.
