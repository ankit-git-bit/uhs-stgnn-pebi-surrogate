# Reproduce all results for the GNN UHS Surrogate model paper from start to finish in PowerShell.

$ErrorActionPreference = "Stop"

# Detect python executable
if (Test-Path ".venv\Scripts\python.exe") {
    $PYTHON_EXE = ".venv\Scripts\python.exe"
} else {
    $PYTHON_EXE = "python"
}

Write-Host "Using python: $PYTHON_EXE" -ForegroundColor Green

Write-Host "`n=========================================="
Write-Host "1. DELAUNAY/VORONOI GRAPH CONSTRUCTION"
Write-Host "=========================================="
& $PYTHON_EXE src/graph_construction/graph_construction.py

Write-Host "`n=========================================="
Write-Host "2. DATA PREPROCESSING"
Write-Host "=========================================="
& $PYTHON_EXE src/preprocessing/pressure_surrogate_preprocessing.py

Write-Host "`n=========================================="
Write-Host "3. FEATURE ENGINEERING & BASELINES"
Write-Host "=========================================="
& $PYTHON_EXE src/feature_engineering/phase4_spatial_temporal_features.py

Write-Host "`n=========================================="
Write-Host "4. ST-GNN ROLLOUT EVALUATION & FIGURES 8-12"
Write-Host "=========================================="
& $PYTHON_EXE src/evaluation/evaluate_rollout_v3.py

Write-Host "`n=========================================="
Write-Host "5. AUXILIARY FIGURES GENERATION (FIGURES 1-7)"
Write-Host "=========================================="
& $PYTHON_EXE src/utils/generate_missing_plots.py
& $PYTHON_EXE src/utils/generate_fig7_stgnn.py

Write-Host "`n=========================================="
Write-Host "6. MONOLITHIC LATEX RECONSTRUCTION"
Write-Host "=========================================="
& $PYTHON_EXE src/utils/reconstruct_final_paper.py

Write-Host "`n=========================================="
Write-Host "7. LATEX INTEGRITY VALIDATION"
Write-Host "=========================================="
& $PYTHON_EXE src/utils/validate_latex.py

Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "REPRODUCTION PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "All figures exist in results/figures/ and paper_project/figures/" -ForegroundColor Green
Write-Host "All tables exist in results/tables/ and paper_project/tables/" -ForegroundColor Green
Write-Host "All metrics exist in results/metrics/" -ForegroundColor Green
Write-Host "Monolithic paper reconstructed at: final_paper.tex" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
