import os
import re

for file in sorted(os.listdir('.')):
    if not file.endswith('.py'):
        continue
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    matches = re.findall(r'.*dpi.*', content, re.IGNORECASE)
    if matches:
        print(f"=== File: {file} ===")
        for match in matches:
            print(f"  {match.strip()}")
        print("-" * 50)
