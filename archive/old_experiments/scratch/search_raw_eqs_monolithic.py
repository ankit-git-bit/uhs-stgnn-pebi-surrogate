import re
from pathlib import Path

def search_monolithic():
    final_paper = Path("c:/Users/user/Desktop/KTU intern/sciml models/final_paper.tex")
    if not final_paper.exists():
        print("final_paper.tex not found!")
        return
        
    pattern = re.compile(r'\bEq(?:uation)?\.?\s*~?\s*\d+\b', re.IGNORECASE)
    
    with open(final_paper, 'r', encoding='utf-8') as f:
        content = f.read()
        
    lines = content.splitlines()
    found = False
    for idx, line in enumerate(lines):
        if line.strip().startswith('%'):
            continue
        matches = pattern.findall(line)
        if matches:
            print(f"L{idx+1}: {line.strip()}")
            found = True
    if not found:
        print("No raw equation references found in final_paper.tex.")

if __name__ == "__main__":
    search_monolithic()
