import os
import re

sections_dir = 'paper_project/sections'
pattern = re.compile(r'\\ref\{[^}]*limitations[^}]*\}|\\ref\{[^}]*implications[^}]*\}', re.IGNORECASE)

for file in os.listdir(sections_dir):
    if not file.endswith('.tex'):
        continue
    path = os.path.join(sections_dir, file)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    matches = pattern.findall(content)
    if matches:
        print(f"File: {file} | Matches: {matches}")
