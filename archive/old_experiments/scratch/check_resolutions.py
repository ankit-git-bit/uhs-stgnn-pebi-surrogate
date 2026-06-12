import os
from PIL import Image

for folder in ["paper_figures", "paper_project/figures"]:
    print(f"\n=== Folder: {folder} ===")
    if not os.path.exists(folder):
        print("Directory does not exist")
        continue
    for file in sorted(os.listdir(folder)):
        if not file.endswith('.png'):
            continue
        path = os.path.join(folder, file)
        try:
            with Image.open(path) as img:
                print(f"  File: {file} | Size: {img.size} | Format: {img.format} | DPI Info: {img.info.get('dpi')}")
        except Exception as e:
            print(f"  Error reading {file}: {e}")
