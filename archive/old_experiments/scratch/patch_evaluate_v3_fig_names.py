from pathlib import Path

def patch_evaluate_v3_fig_names():
    path = Path('src/evaluation/evaluate_rollout_v3.py')
    print(f"Patching GNN figure names in {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fig 1 -> Fig 8
    content = content.replace('"fig1_training_curves.png"', '"fig8_training_curves.png"')
    content = content.replace('"fig1_training_curves.pdf"', '"fig8_training_curves.pdf"')
    content = content.replace('Saved -> fig1_training_curves.png', 'Saved -> fig8_training_curves.png')

    # Fig 2 -> Fig 9
    content = content.replace('"fig2_rollout_rmse.png"', '"fig9_rollout_rmse.png"')
    content = content.replace('"fig2_rollout_rmse.pdf"', '"fig9_rollout_rmse.pdf"')
    content = content.replace('Saved -> fig2_rollout_rmse.png & .pdf', 'Saved -> fig9_rollout_rmse.png & .pdf')

    # Fig 3 -> Fig 10
    content = content.replace('"fig3_pressure_tracking.png"', '"fig10_pressure_tracking.png"')
    content = content.replace('"fig3_pressure_tracking.pdf"', '"fig10_pressure_tracking.pdf"')
    content = content.replace('Saved -> fig3_pressure_tracking.png', 'Saved -> fig10_pressure_tracking.png')

    # Fig 4 -> Fig 11
    content = content.replace('"fig4_saturation_tracking.png"', '"fig11_saturation_tracking.png"')
    content = content.replace('"fig4_saturation_tracking.pdf"', '"fig11_saturation_tracking.pdf"')
    content = content.replace('Saved -> fig4_saturation_tracking.png', 'Saved -> fig11_saturation_tracking.png')

    # Fig 5 -> Fig 12
    content = content.replace('"fig5_bar_comparison.png"', '"fig12_bar_comparison.png"')
    content = content.replace('"fig5_bar_comparison.pdf"', '"fig12_bar_comparison.pdf"')
    content = content.replace('Saved -> fig5_bar_comparison.png & .pdf', 'Saved -> fig12_bar_comparison.png & .pdf')

    # Replace copy mappings in main
    old_mappings = """    mappings = [
        ("fig1_training_curves", "fig8_training_curves"),
        ("fig2_rollout_rmse", "fig9_rollout_rmse"),
        ("fig3_pressure_tracking", "fig10_pressure_tracking"),
        ("fig4_saturation_tracking", "fig11_saturation_tracking"),
        ("fig5_bar_comparison", "fig12_bar_comparison"),
    ]"""

    new_mappings = """    mappings = [
        ("fig8_training_curves", "fig8_training_curves"),
        ("fig9_rollout_rmse", "fig9_rollout_rmse"),
        ("fig10_pressure_tracking", "fig10_pressure_tracking"),
        ("fig11_saturation_tracking", "fig11_saturation_tracking"),
        ("fig12_bar_comparison", "fig12_bar_comparison"),
    ]"""
    content = content.replace(old_mappings, new_mappings)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  Done patching evaluate_rollout_v3.py")

if __name__ == "__main__":
    patch_evaluate_v3_fig_names()
