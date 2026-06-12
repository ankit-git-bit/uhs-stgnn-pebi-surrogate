import os
import re

sections_dir = 'paper_project/sections'
# Broad search for dP (case sensitive) and delta_P
patterns = [
    (r'dP', 'dP'),
    (r'delta_P', 'delta_P'),
    (r'delta', 'delta')
]

for file in os.listdir(sections_dir):
    if not file.endswith('.tex'):
        continue
    path = os.path.join(sections_dir, file)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for pat, label in patterns:
        matches = re.findall(pat, content, re.IGNORECASE if label == 'delta' else 0)
        if matches:
            # Filter matches to find actual mathematical usage of dP
            actual_matches = []
            with open(path, 'r', encoding='utf-8') as f_read:
                for i, line in enumerate(f_read):
                    # Check if it's not a comment
                    if line.strip().startswith('%'):
                        continue
                    if re.search(pat, line, re.IGNORECASE if label == 'delta' else 0):
                        # print if it contains dP, delta_p, etc.
                        if label == 'dP':
                            # check if it's not part of words like 'independent', 'depend', 'rapid'
                            if re.search(r'\b[a-zA-Z]*dP[a-zA-Z]*\b', line):
                                actual_matches.append((i+1, line.strip()))
                        elif label == 'delta_P':
                            actual_matches.append((i+1, line.strip()))
                        elif label == 'delta':
                            # check if it's not part of 'dataset'
                            if 'delta' in line.lower() and 'dataset' not in line.lower():
                                actual_matches.append((i+1, line.strip()))
            
            if actual_matches:
                print(f"File: {file} | Pattern: {label}")
                for line_no, line in actual_matches:
                    print(f"  Line {line_no}: {line}")
                print("-" * 50)
