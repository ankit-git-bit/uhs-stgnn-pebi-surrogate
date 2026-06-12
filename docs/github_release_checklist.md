# GitHub Release Checklist

Use this checklist to verify that all repository cleanup, documentation, and security steps are complete before uploading to GitHub or submitting as conference supplementary material.

---

## 📋 Release Readiness Checklist

- [x] **README.md Landing Page**: Concise project summary, highlights, Mermaid architecture flowchart, key results tables, and quick-start instructions are complete.
- [x] **MIT License**: Added the `LICENSE` file crediting authors Ankit Kumar, Shankar Lal, and Mayur Pal.
- [x] **Changelog**: Logged all repository cleanups, modular restructurings, and script fixes in `CHANGELOG.md`.
- [x] **Release Notes**: Summarized deliverables, performance results, and split limitations in `RELEASE_NOTES.md`.
- [x] **Citation File**: Created a valid `CITATION.cff` for authors to receive proper citation credit.
- [x] **Robust Git Ignore**: Checked `.gitignore` to verify that large MATLAB simulation cases (`*.mat`) and the large preprocessed matrix (`pressure_surrogate_dataset.npz`) are excluded.
- [x] **Lightweight Assets Tracked**: Confirmed that `data/processed/mesh_graph.npz` (2.0 MB) is tracked in Git to allow zero-setup rollout evaluation.
- [x] **Large File Report**: Created `docs/large_files_report.md` detailing size thresholds and Git LFS configurations.
- [x] **Data Download Guide**: Created `data/README.md` specifying data schemas, sizes, and layout instructions.
- [x] **Credentials & Secrets Check**: Scanned codebase to ensure no personal passwords or API keys are present.
- [x] **Documentation Library**: Reorganized `docs/` with files for evaluation protocols, project overview, and repo map.
- [x] **Safe-Mode Reproducibility**: Completed safe-mode validation runs (backing up results, executing rollouts and LaTeX, restoring results) and logged success in `REPRODUCIBILITY_REPORT.md`.
- [x] **Code Polish**: Cleaned python compile caches (`__pycache__`) and ensured all scripts use dynamic path resolution.
