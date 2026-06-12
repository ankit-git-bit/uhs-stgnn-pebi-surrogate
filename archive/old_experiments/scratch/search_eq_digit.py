import re
from pathlib import Path

def search_eq_digit():
    project_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models/paper_project")
    tex_files = list(project_dir.glob("**/*.tex"))
    
    # regex for "Eq" followed by a number (possibly with spaces, tildes, dots, etc.)
    pattern = re.compile(r'\bEq(?:uation)?\.?\s*~?\s*\d+\b', re.IGNORECASE)
    
    print("Searching for 'Eq.' followed by a number...")
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
        print("No matches found.")

if __name__ == "__main__":
    search_eq_digit()
