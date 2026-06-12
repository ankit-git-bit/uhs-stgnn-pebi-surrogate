import os
import re

sections_dir = 'paper_project/sections'
files = sorted(os.listdir(sections_dir))

img_pattern = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')

print(f"Auditing all images in: {sections_dir}\n")

for file in files:
    if not file.endswith('.tex'):
        continue
    path = os.path.join(sections_dir, file)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        match = img_pattern.search(line)
        if match:
            print(f"=== File: {file} (Line {i+1}) ===")
            print(f"  Line: {line.strip()}")
            # find caption around line i
            caption = "NOT FOUND"
            for offset in range(-15, 16):
                idx = i + offset
                if 0 <= idx < len(lines):
                    cap_match = re.search(r'\\caption\{([^}]+)\}', lines[idx])
                    if cap_match:
                        caption = cap_match.group(1)
                        break
            # find label around line i
            label = "NOT FOUND"
            for offset in range(-15, 16):
                idx = i + offset
                if 0 <= idx < len(lines):
                    lbl_match = re.search(r'\\label\{([^}]+)\}', lines[idx])
                    if lbl_match:
                        label = lbl_match.group(1)
                        break
            print(f"  Caption: {caption}")
            print(f"  Label: {label}")
            print("-" * 50)
