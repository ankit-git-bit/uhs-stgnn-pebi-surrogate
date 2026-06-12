from pathlib import Path

def fix_citation():
    file_path = Path("c:/Users/user/Desktop/KTU intern/sciml models/paper_project/sections/discussion.tex")
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    target = "li2021fourier"
    if target in content:
        print(f"Found '{target}' in {file_path.name}. Replacing with 'li2020fno'...")
        new_content = content.replace(target, "li2020fno")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Replacement complete!")
    else:
        print(f"'{target}' not found in {file_path.name}. Let's search for similar citations...")
        # Check all cite commands
        import re
        cites = re.findall(r'\\cite\{([^}]+)\}', content)
        print("Citations found in discussion.tex:")
        for c in cites:
            print(f"  {c}")

if __name__ == "__main__":
    fix_citation()
