import re
from pathlib import Path

def find_equations_text():
    project_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models/paper_project")
    tex_files = list(project_dir.glob("**/*.tex"))
    
    # Matches patterns like Eq. 4, Eq.~4, Equation~5, Eq~9, etc.
    pattern = re.compile(r'\b(Eq\.|Equation|Eq)(?:\s+|~)(\d+)\b', re.IGNORECASE)
    
    print(f"Scanning LaTeX files for hardcoded equation references...")
    found_any = False
    for tf in tex_files:
        with open(tf, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.splitlines()
        for idx, line in enumerate(lines):
            if line.strip().startswith('%'):
                continue
            matches = pattern.findall(line)
            if matches:
                print(f"{tf.relative_to(project_dir)}:L{idx+1}: {line.strip()}")
                found_any = True
                
    if not found_any:
        print("No hardcoded equation references found.")

if __name__ == "__main__":
    find_equations_text()
