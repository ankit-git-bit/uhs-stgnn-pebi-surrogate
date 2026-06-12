import os

terms = ['massive speedup', 'deployment', 'operational deployment', 'production-ready']

found_any = False
for root, dirs, files in os.walk('paper_project'):
    for file in files:
        if file.endswith('.tex'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            for term in terms:
                if term in content.lower():
                    print(f"Found '{term}' in {path}")
                    found_any = True

if not found_any:
    print("No overclaim terms found.")
