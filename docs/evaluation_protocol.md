# Evaluation Protocol & Preservation Disclaimer

This document describes the training, validation, and testing protocol implemented in the project, details the known data splitting limitations, and outlines the scientific preservation guidelines.

---

## 1. Current Train / Validation / Test Strategy

The current version of the project uses the following split protocol across the five independent simulation cases:
* **Training Set (Cases 1–4)**: Spatiotemporal node features and targets are extracted from Cases 1, 2, 3, and 4 (comprising 584 reporting steps, 10,116 cells per step) to train the baseline ML models and the ST-GNN models.
* **Validation Set (Case 5)**: Case 5 is loaded as the validation dataset during GNN training. The epoch validation loss on Case 5 is used to trigger model checkpoint saving (selecting the model parameters that minimize validation loss).
* **Test Set (Case 5)**: Case 5 is held out and evaluated in a fully autoregressive, multi-step rollout forecasting mode (142 steps) to compute the final scientific metrics (RMSE and $R^2$) reported in the paper.

---

## 2. Potential Evaluation Leakage & Limitations

Using Case 5 as both the validation set (for checkpoint saving/model selection) and the test set (for rollout evaluation) creates a potential **evaluation leakage** channel:
* While GNN parameters are updated using gradients computed solely on the training cases (Cases 1-4), the decision of *which epoch's parameters to save as final* is guided by performance on Case 5.
* This means the model selection is conditioned on the test case, which can lead to optimistic rollout performance estimates.
* In a rigorous open-source maintainer review, this represents a known limitation that should be documented for transparency.

### Recommended Alternative Splits for Future Work
To resolve this leakage in future revisions, researchers may implement:
* **Option A (Clean Split)**:
  * **Train**: Cases 1, 2, 3
  * **Val**: Case 4
  * **Test**: Case 5
* **Option B (Train-Validation Splitting)**:
  * **Train**: Cases 1, 2, 3, 4 (with a random 10% subset of training timesteps held out for validation)
  * **Test**: Case 5 (fully unseen)

---

## 3. Preservation of Published Results

> [!IMPORTANT]
> ### Scientific State Integrity Statement
> * **Preservation of Assets**: This repository preserves the exact checkpoints (`checkpoints/`), metrics (`results/metrics/`), figures (`results/figures/`), and LaTeX tables (`results/tables/`) reported in the published manuscript.
> * **Intentional Protocol Preservation**: The original evaluation protocol (Train: Cases 1-4, Val: Case 5, Test: Case 5) has been preserved intentionally to guarantee 100% numerical reproduction of the published paper.
> * **No Retraining**: No model retraining or model parameter modifications have been performed during this repository cleanup and restructuring.
> * **No Results Altered**: No scientific results, tables, metrics, or generated plots have been modified or edited.
> * **Transparency**: Potential evaluation split limitations are documented here for scientific transparency. Future investigations can leverage the restructured codebase to study alternative splits without affecting the baseline results of this paper.
