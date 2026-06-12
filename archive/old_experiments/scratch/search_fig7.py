import os
from pathlib import Path

def search_fig7():
    workspace_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models")
    py_files = list(workspace_dir.glob("*.py")) + list((workspace_dir / "scratch").glob("*.py"))
    
    target = "fig7_stgnn_architecture"
    print(f"Searching for '{target}' in {len(py_files)} Python files...")
    
    for pf in py_files:
        try:
            with open(pf, 'r', encoding='utf-8') as f:
                content = f.read()
            if target in content:
                print(f"Found in: {pf.relative_to(workspace_dir)}")
        except Exception:
            pass

if __name__ == "__main__":
    search_fig7()
