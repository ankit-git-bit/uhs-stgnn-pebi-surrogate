# Release Notes - v1.0.0

This release prepares the Spatio-Temporal Graph Neural Networks (ST-GNN) surrogate framework for public repository release and supplementary conference material. It preserves the exact trained weights and scientific results of the accompanying manuscript.

---

## 📦 What's Included

1. **Pre-trained checkpoints**:
   * `checkpoints/stgnn_v1/st_gnn_checkpoint.pt`: Base ST-GNN.
   * `checkpoints/stgnn_v2/st_gnn_v2_checkpoint.pt`: ST-GNN with Layer Normalization and Residual Skips.
   * `checkpoints/stgnn_v3/st_gnn_v3_checkpoint.pt`: ST-GNN with Layer Normalization, Residual Skips, and Weighted Loss.
2. **Preprocessing & Training Pipelines**: Fully modular Python scripts to extract Voronoi connection structures, scale dynamic features, and train baseline estimators.
3. **Automated Reproduction Pipelines**: `reproduce_results.sh` and `reproduce_results.ps1` for one-click verification of baseline modeling, GNN rollouts, figure generation, and LaTeX compilation checks.
4. **LaTeX Manuscript Source**: Complete modular LaTeX files and unified `final_paper.tex` with figures 1–12 and tables 1–5.

---

## 📊 Performance Summary (Rollout pressure RMSE)
On held-out Case 5:
* **ST-GNN v1 (Base)**: `35.26 psi`
* **ST-GNN v2 (+LN +Residuals)**: `31.72 psi`
* **ST-GNN v3 (+LN +Residuals +Weighted Loss)**: **`27.15 psi`** (a 23.01% error reduction over v1)

---

## ⚠️ Known Limitations & Evaluation Split Details

To guarantee 100% numerical replication of the published results, this release preserves the original train/test split protocol:
* **Training Set**: Cases 1–4.
* **Validation Set**: Case 5 (used for early-stopping checkpoint selection).
* **Test Set**: Case 5 (used for 142-step autoregressive rollout evaluation).

*Caution*: Because Case 5 guides validation-based model selection, this introduces a minor evaluation leakage channel. The results represent optimistic estimates of rollout performance. We strongly recommend that future work using this codebase adopts the alternative splits detailed in [docs/evaluation_protocol.md](docs/evaluation_protocol.md) (such as holding out Case 4 for validation and Case 5 strictly for testing).
