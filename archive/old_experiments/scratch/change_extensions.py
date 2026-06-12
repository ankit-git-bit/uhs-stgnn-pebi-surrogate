import os
import re

directories = ['paper_project/sections', 'paper_project/supplementary', 'paper_project']

img_ext_pattern = re.compile(r'(\\includegraphics(?:\[[^\]]*\])?\{figures/[^}]+)\.png\}')

for directory in directories:
    if not os.path.exists(directory):
        continue
    for file in os.listdir(directory):
        if not file.endswith('.tex'):
            continue
        path = os.path.join(directory, file)
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        new_content, count = img_ext_pattern.subn(r'\1}', content)
        if count > 0:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {path}: stripped extension from {count} graphics commands")
