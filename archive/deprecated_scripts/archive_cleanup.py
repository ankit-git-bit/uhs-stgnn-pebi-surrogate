import os
import shutil
import glob

def main():
    print("Starting archive cleanup...")
    
    # Create target directories
    os.makedirs('archive/deprecated_scripts', exist_ok=True)
    os.makedirs('archive/old_experiments', exist_ok=True)

    # Move deprecated scripts
    dep_scripts = [
        'create_paper_figure_folder.py',
        'data_check.py',
        'detailed_mapping_report.py',
        'display_summary.py',
        'evaluate_pignn_rollout.py',
        'fix_unicode.py',
        'generate_plots.py',
        'inspect_phase1.py',
        'load_and_use_dataset.py',
        'optimize_schedules.py',
        'phase1_5_validation.py',
        'phase2_baselines.py',
        'phase3_pressure_change.py',
        'train_pignn.py'
    ]

    for script in dep_scripts:
        if os.path.exists(script):
            shutil.move(script, 'archive/deprecated_scripts/')
            print(f"  Moved to archive/deprecated_scripts: {script}")

    # Move old experiments, reports, and redundant assets
    old_exps = [
        'combined_paper_draft.tex', 'conference_paper.tex', 'data_analysis_report.tex',
        'final_report.tex', 'full_project_report.tex',
        'conference_questions.md', 'conference_submission_readiness.md',
        'DATA_INSPECTION_REPORT.md', 'figures_required.md', 'figure_audit_table.md',
        'figure_quality_report.md', 'final_reviewer_revision_log.md', 'future_work_roadmap.md',
        'OVERLEAF_READY_CHECKLIST.md', 'paper_fig.md', 'paper_figures.md', 'paper_figures.txt',
        'paper_outline.md', 'pdf_quality_report.md', 'presentation_outline.md',
        'publication_readiness.md', 'README_PHASE_1.md', 'reviewer2_report.md', 'story_audit.md',
        'tables_required.md', 'VERSION_FREEZE.md', 'walkthrough.md',
        'data.docx', 'data_analysis.json', 'uhs_combined.zip',
        'dataset_overview.png', 'diagnostic_pressure_evolution.png', 'feature_distributions.png',
        'target_distributions.png',
        'PHASE_1_5_REPORT.md', 'PHASE_1_5_VALIDATION_SUMMARY.json', 'PHASE_1_GUIDE.md',
        'PHASE_2_BASELINE_SUMMARY.json', 'PHASE_2_REPORT.md', 'PHASE_2_REPORT_TEMP.json',
        'PHASE_3_PRESSURE_CHANGE_SUMMARY.json', 'PHASE_3_REPORT.md',
        'st_gnn_optimization_summary.json', 'st_gnn_pignn_checkpoint.pt',
        'st_gnn_pignn_rollout_summary.json', 'st_gnn_rollout_summary.json',
        'st_gnn_v2_rollout_summary.json', 'st_gnn_v3_rollout_summary.json'
    ]

    for item in old_exps:
        if os.path.exists(item):
            shutil.move(item, 'archive/old_experiments/')
            print(f"  Moved to archive/old_experiments: {item}")

    # Move obsolete directories
    old_dirs = [
        'uha_trial_1', 'uhs_extracted', 'phase2_plots', 'phase3_plots',
        'phase4_plots', 'phase6_plots', 'paper_figures', 'paper_figure'
    ]

    for d in old_dirs:
        if os.path.exists(d):
            # If target dir already exists, delete it first to avoid nesting
            target_dir = os.path.join('archive/old_experiments', d)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.move(d, 'archive/old_experiments/')
            print(f"  Moved directory to archive/old_experiments: {d}")

    # Delete temporary files
    temp_files = ['fix_log.txt', 'PHASE_2_REPORT_TEMP.json']
    for tf in temp_files:
        if os.path.exists(tf):
            os.remove(tf)
            print(f"  Deleted temp file: {tf}")

    # Move scratch folder contents
    if os.path.exists('scratch'):
        target_scratch = 'archive/old_experiments/scratch'
        if os.path.exists(target_scratch):
            shutil.rmtree(target_scratch)
        shutil.move('scratch', target_scratch)
        print("  Moved scratch folder to archive/old_experiments/scratch")

    # Delete __pycache__ folders recursively
    pycache_count = 0
    for root, dirs, files in list(os.walk('.')):
        # Skip .venv
        if '.venv' in root:
            continue
        for d in dirs:
            if d == '__pycache__':
                full_path = os.path.join(root, d)
                shutil.rmtree(full_path)
                pycache_count += 1
    print(f"  Cleaned {pycache_count} __pycache__ directories.")
    
    print("Archive cleanup complete!")

if __name__ == "__main__":
    main()
