from pathlib import Path

def patch_generate_missing_plots():
    path = Path('src/utils/generate_missing_plots.py')
    print(f"Patching {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Paths at the beginning of main()
    old_paths = """def main():
    data_dir = Path(__file__).parent
    fig_dir = data_dir / "paper_project" / "figures"
    fig_dir.mkdir(exist_ok=True)"""

    new_paths = """def main():
    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    results_figures_dir = project_root / "results" / "figures"
    results_metrics_dir = project_root / "results" / "metrics"
    fig_dir = project_root / "paper_project" / "figures"
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper to save to both locations
    def save_plot(fig, filename, dpi=600):
        fig.savefig(fig_dir / filename, dpi=dpi, bbox_inches='tight')
        fig.savefig(results_figures_dir / filename, dpi=dpi, bbox_inches='tight')
        
    def save_pdf(fig, filename):
        fig.savefig(fig_dir / filename, bbox_inches='tight')
        fig.savefig(results_figures_dir / filename, bbox_inches='tight')"""

    content = content.replace(old_paths, new_paths)

    # Load paths replacements
    content = content.replace('data_dir / "Case_0001_wj.mat"', 'data_raw_dir / "Case_0001_wj.mat"')
    content = content.replace('data_dir / "mesh_graph.npz"', 'data_processed_dir / "mesh_graph.npz"')
    content = content.replace("data_dir / 'PHASE_4_FEATURE_ENGINEERING_SUMMARY.json'", "results_metrics_dir / 'PHASE_4_FEATURE_ENGINEERING_SUMMARY.json'")

    # Fig 1 save
    content = content.replace(
        "fig.savefig(fig_dir / \"fig1_dataset_overview.png\", dpi=600, bbox_inches='tight')\n        fig.savefig(fig_dir / \"fig1_dataset_overview.pdf\", bbox_inches='tight')",
        "save_plot(fig, \"fig1_dataset_overview.png\", dpi=600)\n        save_pdf(fig, \"fig1_dataset_overview.pdf\")"
    )

    # Fig 2 save
    content = content.replace(
        "fig.savefig(fig_dir / \"fig2_mesh_topology.png\", dpi=600, bbox_inches='tight')\n        fig.savefig(fig_dir / \"fig2_mesh_topology.pdf\", bbox_inches='tight')",
        "save_plot(fig, \"fig2_mesh_topology.png\", dpi=600)\n        save_pdf(fig, \"fig2_mesh_topology.pdf\")"
    )

    # Fig 6 save
    content = content.replace(
        "fig.savefig(fig_dir / \"fig6_interpolation_error.png\", dpi=600, bbox_inches='tight')\n        fig.savefig(fig_dir / \"fig6_interpolation_error.pdf\", bbox_inches='tight')",
        "save_plot(fig, \"fig6_interpolation_error.png\", dpi=600)\n        save_pdf(fig, \"fig6_interpolation_error.pdf\")"
    )

    # Fig 4 save
    content = content.replace(
        "fig.savefig(fig_dir / \"fig4_baseline_comparison.png\", dpi=600, bbox_inches='tight')\n        fig.savefig(fig_dir / \"fig4_baseline_comparison.pdf\", bbox_inches='tight')",
        "save_plot(fig, \"fig4_baseline_comparison.png\", dpi=600)\n        save_pdf(fig, \"fig4_baseline_comparison.pdf\")"
    )

    # Fig 5 save
    content = content.replace(
        "fig.savefig(fig_dir / \"fig5_feature_importance.png\", dpi=600, bbox_inches='tight')\n        fig.savefig(fig_dir / \"fig5_feature_importance.pdf\", bbox_inches='tight')",
        "save_plot(fig, \"fig5_feature_importance.png\", dpi=600)\n        save_pdf(fig, \"fig5_feature_importance.pdf\")"
    )

    # Fig 3 save
    content = content.replace(
        "fig.savefig(fig_dir / \"fig3_pressure_evolution.png\", dpi=600, bbox_inches='tight')\n        fig.savefig(fig_dir / \"fig3_pressure_evolution.pdf\", bbox_inches='tight')",
        "save_plot(fig, \"fig3_pressure_evolution.png\", dpi=600)\n        save_pdf(fig, \"fig3_pressure_evolution.pdf\")"
    )

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  Done patching generate_missing_plots.py")

if __name__ == "__main__":
    patch_generate_missing_plots()
