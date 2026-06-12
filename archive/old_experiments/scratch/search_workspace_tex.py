import re
from pathlib import Path

def search_workspace_tex():
    workspace_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models")
    tex_files = list(workspace_dir.glob("*.tex"))
    
    pattern = re.compile(r'\bEq(?:uation)?\.?\s*~?\s*\d+\b', re.IGNORECASE)
    
    print(f"Scanning {len(tex_files)} root LaTeX files for raw equation references...")
    found = False
    for tf in tex_files:
        with open(tf, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.splitlines()
        for idx, line in enumerate(lines):
            if line.strip().startswith('%'):
                continue
            matches = pattern.findall(line)
            if matches:
                print(f"{tf.name}:L{idx+1}: {line.strip()}")
                found = True
    if not found:
        print("No matches found in root LaTeX files.")

if __name__ == "__main__":
    search_workspace_tex()
