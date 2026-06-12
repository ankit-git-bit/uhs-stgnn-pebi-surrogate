import re
from pathlib import Path

def scan_generalization():
    project_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models/paper_project")
    tex_files = list(project_dir.glob("**/*.tex"))
    
    pattern = re.compile(r'\b(generaliz[a-z]*)\b', re.IGNORECASE)
    
    print(f"Scanning {len(tex_files)} LaTeX files for generalization claims...")
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
        print("No generalization claims found.")

if __name__ == "__main__":
    scan_generalization()
