# Large File Audit & Git LFS Configuration Report

This report details the files in the repository that exceed 25MB, 50MB, and 100MB thresholds, along with recommendations and configurations for managing large assets using Git LFS (Large File Storage).

---

## 1. Large File Audit Findings

The project files have been audited and grouped by size thresholds:

### Files Exceeding 100MB
* **`data/processed/pressure_surrogate_dataset.npz`** (~229.7 MB)
  * *Description*: Preprocessed spatiotemporal node features and targets fitted with normalization scalers for training and evaluation.
  * *Handling*: Excluded from standard Git tracking via `.gitignore` to avoid inflating the repository size. Can be fully regenerated from raw data or downloaded.

### Files Exceeding 50MB (and under 100MB)
* **`data/raw/Case_0001_wj.mat`** (~59.4 MB)
* **`data/raw/Case_0002_wj.mat`** (~59.9 MB)
* **`data/raw/Case_0003_wj.mat`** (~58.4 MB)
* **`data/raw/Case_0004_wj.mat`** (~59.8 MB)
* **`data/raw/Case_0005_wj.mat`** (~60.0 MB)
  * *Description*: Raw simulator output case files containing pressure, gas saturation, grid connectivity, and well BHP/flow-rate histories.
  * *Handling*: Excluded from standard Git tracking via `.gitignore`. Download guide is provided in `data/README.md`.

### Files Exceeding 25MB (and under 50MB)
* *No files in this range.* All other model checkpoints (~720 KB), figures (<1.2 MB), and source files are lightweight.

---

## 2. Git LFS Configuration Guide

If your institution or conference repository supports Git LFS and you wish to track the raw datasets or large preprocessed matrices in the repository rather than git-ignoring them, follow these configuration steps:

### Step 1: Install Git LFS
Download and install the Git LFS extension from [git-lfs.com](https://git-lfs.github.com/). Initialize LFS in your local environment:
```bash
git lfs install
```

### Step 2: Track Large File Extensions
Configure Git LFS to track `.mat` files and the large preprocessed `.npz` file:
```bash
# Track all MATLAB raw case files
git lfs track "*.mat"

# Track the large preprocessed dataset specifically
git lfs track "data/processed/pressure_surrogate_dataset.npz"
```
*Note: This generates or updates a `.gitattributes` file in the root folder, which must be committed to the repository.*

### Step 3: Remove Exclusions from `.gitignore`
If you transition to Git LFS, remove the following lines from `.gitignore` so Git LFS can track them:
```diff
- *.mat
- data/raw/*.mat
- data/processed/pressure_surrogate_dataset.npz
```

### Step 4: Commit and Push
Commit `.gitattributes` along with your large files. Git LFS will automatically intercept the push and upload the files to the LFS server:
```bash
git add .gitattributes
git add data/raw/*.mat
git add data/processed/pressure_surrogate_dataset.npz
git commit -m "Configure Git LFS and add large datasets"
git push origin main
```
