# Contributing to Spatio-Temporal GNN for UHS Forecasting

Thank you for your interest in contributing to this Scientific ML research project! We welcome collaborations that improve surrogate modeling, code efficiency, documentation, or physical constraint integration.

---

## Code of Conduct

By participating in this project, you agree to maintain a professional, collaborative, and constructive environment.

---

## How to Contribute

### 1. Reporting Bugs & Feature Requests
* Search the existing issues before opening a new one.
* Use the provided [Issue Template](.github/ISSUE_TEMPLATE.md) to report bugs with reproducible steps, environment details, and error logs.

### 2. Submitting Pull Requests
To submit a code contribution:
1. **Fork the Repository**: Create a personal copy of the repository.
2. **Create a Feature Branch**: Work on a separate branch (e.g., `feature/improve-gnn-efficiency`).
3. **Commit Changes**: Use clear, descriptive commit messages.
4. **Maintain Scientific Integrity**: Do not modify default hyperparameters or datasets in a way that alters the published benchmark results, unless explicitly requested.
5. **Run Verification**:
   - Ensure the automated verification script (`reproduce_results.sh` / `reproduce_results.ps1`) executes successfully without errors.
   - Run the LaTeX validator script to check manuscript integrity:
     ```bash
     python src/utils/validate_latex.py
     ```
6. **Submit PR**: Open a Pull Request targeting the main branch, filling out the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md) fully.

---

## Code Style & Formatting

* **Python Guidelines**:
  * Follow PEP 8 guidelines.
  * Use clear variable names that correspond to physical quantities (e.g., `transmissibility` instead of `t_var`, `pressure_change` instead of `dp`).
  * Document all functions and classes, noting the mathematical formulations where applicable.
* **LaTeX Guidelines**:
  * Avoid raw hardcoded numbers or hardcoded paths that break build scripts.
  * Keep modular sections structured under `paper_project/`.
