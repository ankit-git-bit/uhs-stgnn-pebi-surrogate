# Reviewer Revision Tasks

- [x] **PHASE 1: Fix LaTeX Errors**
  - [x] Fix the double backslash `\\begin{table*}` typo in `sections/rollout_results.tex`.
  - [x] Perform a reference and citation validation check using `validate_latex.py`.

- [x] **PHASE 2: Redesign Figures**
  - [x] Update `generate_missing_plots.py` to redesign Figure 3 (Pressure Evolution) into a 2x3 layout with equal axis scaling, distinct wellbore markers, and updated captions.
  - [x] Create `scratch/generate_fig7_stgnn.py` to draw the professional ST-GNN architecture diagram and save as PNG/PDF.
  - [x] Update `evaluate_rollout_v3.py` to add vertical transition lines and annotations to Figure 9 (Rollout RMSE).
  - [x] Run the scripts to generate the new figures and copy them into `paper_project/figures/`.

- [x] **PHASE 3: Scientific Discussion Upgrades**
  - [x] Add the subsection `\subsection{Impact of Well-Control Transitions on Rollout Error}` to `sections/discussion.tex`.
  - [x] Expand Section 10.6 with physical interpretations of pressure diffusion, linking it to hydraulic diffusivity and the temporal lag feature.
  - [x] Convert the diffusion-timescale formula in Section 10.6 to a numbered equation `t_D` and reference it properly.
  - [x] Add the Random Forest teacher-forcing warning to both `sections/discussion.tex` and `sections/conclusions.tex`.
  - [x] Upgrade the literature comparison table (**Table 6**) in `sections/discussion.tex` to include all requested columns and the comparability disclaimer.

- [x] **PHASE 4: Remove Overclaims**
  - [x] Scan and replace AI-style overclaims (e.g. "massive speedup", "deployment") in all LaTeX files.

- [x] **PHASE 5: Reference Verification**
  - [x] Audit `references.bib` to verify authors, journals, years, volumes, pages, and DOIs for all SciML and UHS papers.

- [x] **PHASE 6: Assembly & Final Verification**
  - [x] Reconstruct `final_paper.tex` using the updated modular files.
  - [x] Synchronize all updated files, figures, tables, and reports to `uhs_trial_1/` and `uha_trial_1/`.
  - [x] Create `uhs_trial_1/revision_report.md` detailing all performed changes.
  - [x] Run `validate_latex.py` on `uhs_trial_1` and verify successful compilation.

# Restructuring & Packaging Checklist (Preservation Mode)

- [x] Phase 1: Repository Audit & Legacy Cleanup (move files to `archive/`)
- [x] Phase 2: Document Evaluation Protocol & Preservation Disclaimer (`docs/evaluation_protocol.md`)
- [x] Phase 3: Professional Repository Structure Setup
  - [x] Reorganize folders (`data/`, `src/`, `results/`, `checkpoints/`, `docs/`, `archive/`)
  - [x] Create `docs/project_overview.md`
  - [x] Create `docs/repository_structure.md`
  - [x] Create `docs/reproducibility.md`
  - [x] Create `CITATION.cff`
  - [x] Create academic community templates under `.github/`
- [x] Phase 4: Rewrite README.md for a 60-Second Landing Page
- [x] Phase 5: GitHub Safety Check & robust `.gitignore` setup
- [x] Phase 6: Large File Management & Auditing (`docs/large_files_report.md` & `data/README.md`)
- [x] Phase 7: Safe Reproducibility Validation (backup results, run inference, check LaTeX, restore backup)
  - [x] Backup `results/`
  - [x] Run GNN rollout check
  - [x] Run LaTeX validator
  - [x] Restore `results/`
  - [x] Generate `REPRODUCIBILITY_REPORT.md`
- [x] Phase 8: Public Release Preparation (`LICENSE`, `CHANGELOG.md`, `RELEASE_NOTES.md`)
- [x] Phase 9: GitHub Polish & Quality Review (`docs/repository_quality_review.md` & `docs/github_release_checklist.md`)
- [x] Phase 10: Final Audit Summary (`FINAL_REPOSITORY_AUDIT.md`)
