import os
from pathlib import Path

def search_broken_latex():
    project_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models/paper_project")
    tex_files = list(project_dir.glob("**/*.tex"))
    
    print(f"Scanning {len(tex_files)} files for broken LaTeX commands...")
    for tf in tex_files:
        with open(tf, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.splitlines()
        for idx, line in enumerate(lines):
            if "begintable" in line:
                print(f"Broken command in {tf.relative_to(project_dir)}:L{idx+1}: {line}")
            if "endtable" in line:
                print(f"Broken command in {tf.relative_to(project_dir)}:L{idx+1}: {line}")

if __name__ == "__main__":
    search_broken_latex()
