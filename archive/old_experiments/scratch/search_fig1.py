import os

sections_dir = 'paper_project/sections'
for file in sorted(os.listdir(sections_dir)):
    if not file.endswith('.tex'):
        continue
    path = os.path.join(sections_dir, file)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'permeability realization' in line.lower():
            print(f"=== File: {file} (Line {i+1}) ===")
            print(f"  Line: {line.strip()}")
            if i > 0:
                print(f"  Prev: {lines[i-1].strip()}")
            if i < len(lines) - 1:
                print(f"  Next: {lines[i+1].strip()}")
            print("-" * 50)
