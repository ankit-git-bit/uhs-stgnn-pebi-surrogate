import re
from pathlib import Path

def search_uhs_trial_eqs():
    project_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models/uhs_trial_1")
    if not project_dir.exists():
        print("uhs_trial_1 folder does not exist!")
        return
        
    tex_files = list(project_dir.glob("**/*.tex"))
    pattern = re.compile(r'\bEq(?:uation)?\.?\s*~?\s*\d+\b', re.IGNORECASE)
    
    print(f"Scanning {len(tex_files)} files in uhs_trial_1 for raw equation references...")
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
                print(f"{tf.relative_to(project_dir)}:L{idx+1}: {line.strip()}")
                found = True
    if not found:
        print("No matches found in uhs_trial_1.")

if __name__ == "__main__":
    search_uhs_trial_eqs()
