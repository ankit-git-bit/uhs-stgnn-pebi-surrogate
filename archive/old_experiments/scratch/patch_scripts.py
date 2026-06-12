import os
import re
from pathlib import Path

def patch_st_gnn_v1_training():
    path = Path('src/training/train_st_gnn_v1.py')
    print(f"Patching {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Add sys.path insert
    sys_path_code = """import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
    if "sys.path.insert" not in content:
        content = sys_path_code + content
        
    # Replace imports
    content = re.sub(
        r'from st_gnn_model import UHSGraphDataset, ST_GNN',
        r'from src.models.stgnn_v1.st_gnn_model import UHSGraphDataset, ST_GNN',
        content
    )
    
    # Replace path definitions in main
    old_paths = """    data_dir = Path(__file__).parent
    graph_path = data_dir / "mesh_graph.npz\""""
    
    new_paths = """    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    checkpoint_dir = project_root / "checkpoints" / "stgnn_v1"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = data_processed_dir / "mesh_graph.npz\""""
    
    content = content.replace(old_paths, new_paths)
    
    # Replace UHSGraphDataset initialization paths
    content = content.replace("UHSGraphDataset(data_dir,", "UHSGraphDataset(data_raw_dir,")
    
    # Replace output files paths
    content = content.replace("open(data_dir / 'st_gnn_stats.json'", "open(data_processed_dir / 'st_gnn_stats.json'")
    content = content.replace("'st_gnn_checkpoint.pt'", "'st_gnn_checkpoint.pt'") # placeholder
    content = content.replace("data_dir / 'st_gnn_checkpoint.pt'", "checkpoint_dir / 'st_gnn_checkpoint.pt'")
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  Done patching train_st_gnn_v1.py")

def patch_st_gnn_v2_training():
    path = Path('src/training/train_st_gnn_v2.py')
    print(f"Patching {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    sys_path_code = """import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
    if "sys.path.insert" not in content:
        content = sys_path_code + content
        
    content = re.sub(
        r'from st_gnn_v2_model import UHSGraphDataset, ST_GNN_v2',
        r'from src.models.stgnn_v2.st_gnn_v2_model import UHSGraphDataset, ST_GNN_v2',
        content
    )
    
    # Paths in main
    old_paths = """    data_dir   = Path(__file__).parent
    graph_path = data_dir / "mesh_graph.npz\""""
    
    new_paths = """    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    checkpoint_dir = project_root / "checkpoints" / "stgnn_v2"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = data_processed_dir / "mesh_graph.npz\""""
    
    content = content.replace(old_paths, new_paths)
    content = content.replace("UHSGraphDataset(data_dir,", "UHSGraphDataset(data_raw_dir,")
    content = content.replace("open(data_dir / 'st_gnn_v2_stats.json'", "open(data_processed_dir / 'st_gnn_v2_stats.json'")
    content = content.replace("data_dir / 'st_gnn_v2_checkpoint.pt'", "checkpoint_dir / 'st_gnn_v2_checkpoint.pt'")
    content = content.replace("data_dir / 'st_gnn_v2_training_history.json'", "data_processed_dir / 'st_gnn_v2_training_history.json'")
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  Done patching train_st_gnn_v2.py")

def patch_st_gnn_v3_training():
    path = Path('src/training/train_st_gnn_v3.py')
    print(f"Patching {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    sys_path_code = """import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
    if "sys.path.insert" not in content:
        content = sys_path_code + content
        
    content = re.sub(
        r'from st_gnn_model    import UHSGraphDataset, ST_GNN',
        r'from src.models.stgnn_v1.st_gnn_model import UHSGraphDataset, ST_GNN',
        content
    )
    content = re.sub(
        r'from st_gnn_v2_model import ST_GNN_v2',
        r'from src.models.stgnn_v2.st_gnn_v2_model import ST_GNN_v2',
        content
    )
    
    # Paths in main
    old_paths = """    data_dir   = Path(__file__).parent
    graph_path = data_dir / "mesh_graph.npz\""""
    
    new_paths = """    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    checkpoint_dir = project_root / "checkpoints" / "stgnn_v3"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = data_processed_dir / "mesh_graph.npz\""""
    
    content = content.replace(old_paths, new_paths)
    content = content.replace("UHSGraphDataset(data_dir,", "UHSGraphDataset(data_raw_dir,")
    content = content.replace("case5 = loadmat(str(data_dir / 'Case_0005_wj.mat'))", "case5 = loadmat(str(data_raw_dir / 'Case_0005_wj.mat'))")
    content = content.replace("open(data_dir / 'st_gnn_v3_study_summary.json'", "open(data_processed_dir / 'st_gnn_v3_study_summary.json'")
    content = content.replace("data_dir / 'st_gnn_v3_checkpoint.pt'", "checkpoint_dir / 'st_gnn_v3_checkpoint.pt'")
    content = content.replace("data_dir / 'st_gnn_v3_training_history.json'", "data_processed_dir / 'st_gnn_v3_training_history.json'")
    content = content.replace("data_dir / 'st_gnn_v3_rollout_summary.json'", "data_processed_dir / 'st_gnn_v3_rollout_summary.json'")
    
    # In train_one_config stats save
    content = content.replace("open(data_dir / f'{label}_stats.json'", "open(data_processed_dir / f'{label}_stats.json'")
    content = content.replace("open(data_dir / 'st_gnn_v3_stats.json'", "open(data_processed_dir / 'st_gnn_v3_stats.json'")
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  Done patching train_st_gnn_v3.py")

def patch_evaluate_rollout_v1():
    path = Path('src/evaluation/evaluate_rollout_v1.py')
    print(f"Patching {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    sys_path_code = """import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
    if "sys.path.insert" not in content:
        content = sys_path_code + content
        
    content = re.sub(
        r'from st_gnn_model import ST_GNN, UHSGraphDataset',
        r'from src.models.stgnn_v1.st_gnn_model import ST_GNN, UHSGraphDataset',
        content
    )
    
    old_paths = """    data_dir = Path(__file__).parent
    graph_path = data_dir / "mesh_graph.npz"
    stats_path = data_dir / "st_gnn_stats.json"
    checkpoint_path = data_dir / "st_gnn_checkpoint.pt"
    plot_dir = data_dir / "phase6_plots\""""
    
    new_paths = """    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    checkpoint_dir = project_root / "checkpoints" / "stgnn_v1"
    results_figures_dir = project_root / "results" / "figures"
    results_metrics_dir = project_root / "results" / "metrics"
    
    results_figures_dir.mkdir(parents=True, exist_ok=True)
    results_metrics_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = data_processed_dir / "mesh_graph.npz"
    stats_path = data_processed_dir / "st_gnn_stats.json"
    checkpoint_path = checkpoint_dir / "st_gnn_checkpoint.pt"
    plot_dir = results_figures_dir"""
    
    content = content.replace(old_paths, new_paths)
    content = content.replace("case_path = data_dir / \"Case_0005_wj.mat\"", "case_path = data_raw_dir / \"Case_0005_wj.mat\"")
    content = content.replace("open(data_dir / \"st_gnn_rollout_summary.json\"", "open(results_metrics_dir / \"st_gnn_rollout_summary.json\"")
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  Done patching evaluate_rollout_v1.py")

def patch_evaluate_rollout_v2():
    path = Path('src/evaluation/evaluate_rollout_v2.py')
    print(f"Patching {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    sys_path_code = """import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
    if "sys.path.insert" not in content:
        content = sys_path_code + content
        
    content = re.sub(
        r'from st_gnn_model import UHSGraphDataset, ST_GNN',
        r'from src.models.stgnn_v1.st_gnn_model import UHSGraphDataset, ST_GNN',
        content
    )
    content = re.sub(
        r'from st_gnn_v2_model import ST_GNN_v2',
        r'from src.models.stgnn_v2.st_gnn_v2_model import ST_GNN_v2',
        content
    )
    
    old_paths = """    data_dir = Path(__file__).parent
    graph_path = data_dir / "mesh_graph.npz"
    plot_dir = data_dir / "phase6_plots\""""
    
    new_paths = """    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    checkpoint_dir_v1 = project_root / "checkpoints" / "stgnn_v1"
    checkpoint_dir_v2 = project_root / "checkpoints" / "stgnn_v2"
    results_figures_dir = project_root / "results" / "figures"
    results_metrics_dir = project_root / "results" / "metrics"
    
    results_figures_dir.mkdir(parents=True, exist_ok=True)
    results_metrics_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = data_processed_dir / "mesh_graph.npz"
    plot_dir = results_figures_dir"""
    
    content = content.replace(old_paths, new_paths)
    
    # replace stats and checkpoints lookups
    content = content.replace('stats_v1_path = data_dir / "st_gnn_stats.json"', 'stats_v1_path = data_processed_dir / "st_gnn_stats.json"')
    content = content.replace('ckpt_v1_path  = data_dir / "st_gnn_checkpoint.pt"', 'ckpt_v1_path  = checkpoint_dir_v1 / "st_gnn_checkpoint.pt"')
    content = content.replace('stats_v2_path = data_dir / "st_gnn_v2_stats.json"', 'stats_v2_path = data_processed_dir / "st_gnn_v2_stats.json"')
    content = content.replace('ckpt_v2_path  = data_dir / "st_gnn_v2_checkpoint.pt"', 'ckpt_v2_path  = checkpoint_dir_v2 / "st_gnn_v2_checkpoint.pt"')
    
    content = content.replace('case_path = data_dir / "Case_0005_wj.mat"', 'case_path = data_raw_dir / "Case_0005_wj.mat"')
    content = content.replace('open(data_dir / "st_gnn_v2_rollout_summary.json"', 'open(results_metrics_dir / "st_gnn_v2_rollout_summary.json"')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  Done patching evaluate_rollout_v2.py")

def patch_evaluate_rollout_v3():
    path = Path('src/evaluation/evaluate_rollout_v3.py')
    print(f"Patching {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    sys_path_code = """import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
    if "sys.path.insert" not in content:
        content = sys_path_code + content
        
    content = re.sub(
        r'from st_gnn_model\s+import ST_GNN',
        r'from src.models.stgnn_v1.st_gnn_model import ST_GNN',
        content
    )
    content = re.sub(
        r'from st_gnn_v2_model import ST_GNN_v2',
        r'from src.models.stgnn_v2.st_gnn_v2_model import ST_GNN_v2',
        content
    )
    
    # Paths in main
    old_main_paths = """    data_dir = Path(__file__).parent
    plot_dir = data_dir / "paper_figures"
    plot_dir.mkdir(exist_ok=True)"""
    
    new_main_paths = """    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    results_figures_dir = project_root / "results" / "figures"
    results_metrics_dir = project_root / "results" / "metrics"
    
    # Make sure we also have the paper_project/figures path so Overleaf matches
    paper_figs_dir = project_root / "paper_project" / "figures"
    
    results_figures_dir.mkdir(parents=True, exist_ok=True)
    results_metrics_dir.mkdir(parents=True, exist_ok=True)
    paper_figs_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = data_processed_dir # read stats from data/processed
    plot_dir = results_figures_dir"""
    
    content = content.replace(old_main_paths, new_main_paths)
    
    # Replace GNN check paths for v1, v2, v3 checkpoints
    content = content.replace(
        "'v1': ('st_gnn_checkpoint.pt',    'st_gnn_stats.json',    'ST_GNN'),",
        "'v1': (project_root / 'checkpoints/stgnn_v1/st_gnn_checkpoint.pt',    'st_gnn_stats.json',    'ST_GNN'),"
    )
    content = content.replace(
        "'v2': ('st_gnn_v2_checkpoint.pt', 'st_gnn_v2_stats.json', 'ST_GNN_v2'),",
        "'v2': (project_root / 'checkpoints/stgnn_v2/st_gnn_v2_checkpoint.pt', 'st_gnn_v2_stats.json', 'ST_GNN_v2'),"
    )
    content = content.replace(
        "'v3': ('st_gnn_v3_checkpoint.pt', 'st_gnn_v3_stats.json', 'ST_GNN_v2'),",
        "'v3': (project_root / 'checkpoints/stgnn_v3/st_gnn_v3_checkpoint.pt', 'st_gnn_v3_stats.json', 'ST_GNN_v2'),"
    )
    
    # Replace GNN file existence checks
    content = content.replace(
        "if (data_dir / ckpt).exists() and (data_dir / stats).exists():",
        "if Path(ckpt).exists() and (data_processed_dir / stats).exists():"
    )
    content = content.replace(
        "available[name] = (ckpt, stats, arch)",
        "available[name] = (Path(ckpt), stats, arch)"
    )
    content = content.replace(
        "print(f\"  [OK] {name}: {ckpt}\")",
        "print(f\"  [OK] {name}: {Path(ckpt).name}\")"
    )
    content = content.replace(
        "print(f\"  [MISSING] {name}: {ckpt} or {stats} not found -- skipping\")",
        "print(f\"  [MISSING] {name}: {Path(ckpt).name} or {stats} not found -- skipping\")"
    )
    
    # Checkpoint load path
    content = content.replace(
        "model.load_state_dict(torch.load(data_dir / ckpt, map_location=device))",
        "model.load_state_dict(torch.load(ckpt, map_location=device))"
    )
    
    # Case 5 load path
    content = content.replace(
        "case_data  = sio.loadmat(str(data_dir / \"Case_0005_wj.mat\"))",
        "case_data  = sio.loadmat(str(data_raw_dir / \"Case_0005_wj.mat\"))"
    )
    
    # mesh_graph load path
    content = content.replace(
        "graph_data = np.load(data_dir / \"mesh_graph.npz\")",
        "graph_data = np.load(data_processed_dir / \"mesh_graph.npz\")"
    )
    
    # metrics output save path
    content = content.replace(
        "with open(data_dir / 'v3_evaluation_results.json', 'w') as f:",
        "with open(results_metrics_dir / 'v3_evaluation_results.json', 'w') as f:"
    )
    content = content.replace(
        "Saved -> v3_evaluation_results.json",
        "Saved -> results/metrics/v3_evaluation_results.json"
    )
    
    # Also save copies to paper_project/figures/ under their final mapped names
    # Let's add a copy section inside generate all 5 figures block
    figure_save_mapping_code = """
    # Copy generated figures to paper_project/figures/ with final paper numbering
    import shutil
    paper_figs_dir = project_root / "paper_project" / "figures"
    paper_figs_dir.mkdir(exist_ok=True)
    mappings = [
        ("fig1_training_curves", "fig8_training_curves"),
        ("fig2_rollout_rmse", "fig9_rollout_rmse"),
        ("fig3_pressure_tracking", "fig10_pressure_tracking"),
        ("fig4_saturation_tracking", "fig11_saturation_tracking"),
        ("fig5_bar_comparison", "fig12_bar_comparison"),
    ]
    for src_base, dest_base in mappings:
        for ext in [".png", ".pdf"]:
            src_file = results_figures_dir / f"{src_base}{ext}"
            dest_file = paper_figs_dir / f"{dest_base}{ext}"
            if src_file.exists():
                shutil.copy2(src_file, dest_file)
    print("  Copied rollout figures to paper_project/figures/ with paper mapping.")
"""
    # Insert this before "print(\"\\n[DONE] All figures saved to paper_figures/\")"
    content = content.replace(
        "print(\"\\n[DONE] All figures saved to paper_figures/\")",
        figure_save_mapping_code + "\n    print(\"\\n[DONE] All figures saved to results/figures/ and paper_project/figures/\")"
    )

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  Done patching evaluate_rollout_v3.py")

def main():
    patch_st_gnn_v1_training()
    patch_st_gnn_v2_training()
    patch_st_gnn_v3_training()
    patch_evaluate_rollout_v1()
    patch_evaluate_rollout_v2()
    patch_evaluate_rollout_v3()
    print("All scripts patched successfully!")

if __name__ == "__main__":
    main()
