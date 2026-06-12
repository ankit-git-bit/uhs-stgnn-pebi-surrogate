import shutil
from pathlib import Path

def copy_figures():
    src_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models/paper_figures")
    dest_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models/paper_project/figures")
    dest_dir.mkdir(exist_ok=True)
    
    mappings = [
        ("fig1_training_curves", "fig8_training_curves"),
        ("fig2_rollout_rmse", "fig9_rollout_rmse"),
        ("fig3_pressure_tracking", "fig10_pressure_tracking"),
        ("fig4_saturation_tracking", "fig11_saturation_tracking"),
        ("fig5_bar_comparison", "fig12_bar_comparison"),
    ]
    
    print("Copying rollout figures to paper_project/figures/...")
    for src_base, dest_base in mappings:
        for ext in [".png", ".pdf"]:
            src_file = src_dir / f"{src_base}{ext}"
            dest_file = dest_dir / f"{dest_base}{ext}"
            if src_file.exists():
                shutil.copy2(src_file, dest_file)
                print(f"  Copied: {src_file.name} -> {dest_file.name}")
            else:
                print(f"  Warning: Source {src_file.name} not found!")

if __name__ == "__main__":
    copy_figures()
