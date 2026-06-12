import os
import re

def analyze_file(filepath):
    print(f"\nAnalyzing: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check imports
    imports = re.findall(r'^\s*(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\., ]+))', content, re.MULTILINE)
    print("  Imports found:")
    for imp in imports:
        # imp is a tuple (from_part, import_part)
        if imp[0]:
            print(f"    - from {imp[0]} import ...")
        if imp[1]:
            print(f"    - import {imp[1]}")
            
    # Check paths
    paths = re.findall(r'(".*?\.(?:mat|npz|pt|json|png|pdf)"|\'.*?\.(?:mat|npz|pt|json|png|pdf)\')', content)
    if paths:
        print("  File path literals found:")
        for p in paths:
            print(f"    - {p}")
            
    # Check Path(__file__) usage
    if "__file__" in content:
        print("  Contains __file__ reference.")

def main():
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                analyze_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
