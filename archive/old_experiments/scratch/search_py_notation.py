import os
import re

for file in os.listdir('.'):
    if not file.endswith('.py'):
        continue
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find string literals containing dP, delta_P, or delta_p
    matches = re.findall(r"['\"][^'\"]*(?:dP|delta_P|delta_p)[^'\"]*['\"]", content)
    if matches:
        print(f"File: {file}")
        for match in matches:
            print(f"  {match}")
        print("-" * 50)
