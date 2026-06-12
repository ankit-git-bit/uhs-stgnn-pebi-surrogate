#!/bin/bash
# Reproduce all results for the GNN UHS Surrogate model paper from start to finish.

# Exit on error
set -e

# Detect virtual environment python
if [ -d ".venv" ]; then
    if [ -f ".venv/Scripts/python" ]; then
        PYTHON_EXE=".venv/Scripts/python"
    elif [ -f ".venv/bin/python" ]; then
        PYTHON_EXE=".venv/bin/python"
    else
        PYTHON_EXE="python"
    fi
else
    PYTHON_EXE="python"
fi

echo "Using python: $PYTHON_EXE"

echo "=========================================="
echo "1. DELAUNAY/VORONOI GRAPH CONSTRUCTION"
echo "=========================================="
$PYTHON_EXE src/graph_construction/graph_construction.py

echo "=========================================="
echo "2. DATA PREPROCESSING"
echo "=========================================="
$PYTHON_EXE src/preprocessing/pressure_surrogate_preprocessing.py

echo "=========================================="
echo "3. FEATURE ENGINEERING & BASELINES"
echo "=========================================="
$PYTHON_EXE src/feature_engineering/phase4_spatial_temporal_features.py

echo "=========================================="
echo "4. ST-GNN ROLLOUT EVALUATION & FIGURES 8-12"
echo "=========================================="
$PYTHON_EXE src/evaluation/evaluate_rollout_v3.py

echo "=========================================="
echo "5. AUXILIARY FIGURES GENERATION (FIGURES 1-7)"
echo "=========================================="
$PYTHON_EXE src/utils/generate_missing_plots.py
$PYTHON_EXE src/utils/generate_fig7_stgnn.py

echo "=========================================="
echo "6. MONOLITHIC LATEX RECONSTRUCTION"
echo "=========================================="
$PYTHON_EXE src/utils/reconstruct_final_paper.py

echo "=========================================="
echo "7. LATEX INTEGRITY VALIDATION"
echo "=========================================="
$PYTHON_EXE src/utils/validate_latex.py

echo "=========================================="
echo "REPRODUCTION PIPELINE COMPLETE!"
echo "All figures exist in results/figures/ and paper_project/figures/"
echo "All tables exist in results/tables/ and paper_project/tables/"
echo "All metrics exist in results/metrics/"
echo "Monolithic paper reconstructed at: final_paper.tex"
echo "=========================================="
