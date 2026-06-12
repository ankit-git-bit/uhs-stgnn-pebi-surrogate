import re
from pathlib import Path

def scan_notation():
    project_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models/paper_project")
    tex_files = list(project_dir.glob("**/*.tex"))
    
    # regex to search for potential pressure change notations
    # excluding variables in code block templates if any
    pattern = re.compile(r'\b(dP|delta_P|delta P|\\delta P)\b', re.IGNORECASE)
    
    print(f"Scanning {len(tex_files)} LaTeX files for mixed pressure change notation...")
    found_any = False
    for tf in tex_files:
        with open(tf, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        for idx, line in enumerate(lines):
            # Skip comments
            if line.strip().startswith('%'):
                continue
            matches = pattern.findall(line)
            if matches:
                # Filter out standard math symbols or python references if they are commented
                print(f"{tf.relative_to(project_dir)}:L{idx+1}: {line.strip()}")
                found_any = True
                
    if not found_any:
        print("No inconsistent pressure change notation found.")

if __name__ == "__main__":
    scan_notation()
