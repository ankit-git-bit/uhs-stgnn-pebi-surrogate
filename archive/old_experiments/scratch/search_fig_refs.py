import os
from pathlib import Path

def search_fig_refs():
    workspace_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models")
    py_files = list(workspace_dir.glob("**/*.py"))
    
    target = "fig8_training_curves"
    print(f"Searching for '{target}' in {len(py_files)} Python files...")
    
    for pf in py_files:
        if "check_generalization" in pf.name or "check_pressure_notation" in pf.name:
            continue
        try:
            with open(pf, 'r', encoding='utf-8') as f:
                content = f.read()
            if target in content:
                print(f"Found in: {pf.relative_to(workspace_dir)}")
        except Exception:
            pass

if __name__ == "__main__":
    search_fig_refs()
